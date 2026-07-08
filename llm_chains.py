import os

from prompt_templates import memory_prompt_template

from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings
)

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import chromadb
import streamlit as st
import yaml

# ─────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Default size of the conversation window (number of past turns, not
# messages) used when building the {history} string for the prompt.
# llm_chains.py used `k=3` previously via ConversationBufferWindowMemory.
DEFAULT_MEMORY_K = config.get("memory_window_turns", 3)

# Default Groq model + generation params (see config.yaml). The
# sidebar "Active Model" selector in app.py can override the model at
# runtime by passing model_name= through load_chain() -> create_llm().
DEFAULT_GROQ_MODEL = config.get("groq_model", "llama-3.1-8b-instant")
_MODEL_CFG = config.get("model_config", {}) or {}
DEFAULT_TEMPERATURE = _MODEL_CFG.get("temperature", 0)
DEFAULT_MAX_TOKENS = _MODEL_CFG.get("max_tokens", 512)


def _get_groq_api_key():
    """Resolve the Groq API key, preferring secure sources over the
    plaintext config.yaml value.

    Order of precedence:
      1. GROQ_API_KEY environment variable
      2. st.secrets["GROQ_API_KEY"] (Streamlit Cloud secrets.toml)
      3. config.yaml -> groq_api_key (fallback so the app still runs
         out of the box; for real deployments prefer options 1 or 2
         and remove the key from config.yaml).
    """
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        api_key = config.get("groq_api_key")

    return api_key


# ─────────────────────────────────────────────
# CREATE LLM
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Groq…")
def create_llm(
    model_name=None,
    temperature=None,
    max_tokens=None
):
    """Build the Groq-backed chat LLM (replaces the old local
    CTransformers/GGUF model).

    Cached with st.cache_resource: constructing the client is cheap,
    but caching keeps a single instance per (model_name, temperature,
    max_tokens) combination alive across Streamlit reruns instead of
    recreating it on every message/button click. load_chain() (called
    once per rerun) calls this via create_llm()/create_llm(model_name=...)
    every time, and st.cache_resource keys on those arguments, so the
    SAME model client is reused whenever the args match.

    Raises a clear error instead of a raw exception if the Groq API
    key is missing, since this used to fail with a confusing traceback
    on a typo'd/missing config value.
    """
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "No Groq API key found. Set the GROQ_API_KEY environment "
            "variable, add it to .streamlit/secrets.toml, or set "
            "'groq_api_key' in config.yaml."
        )

    model_name = model_name or DEFAULT_GROQ_MODEL
    temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    llm = ChatGroq(
        groq_api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

    return llm


# ─────────────────────────────────────────────
# CREATE EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embeddings model…")
def create_embeddings(embeddings_path=None):

    if embeddings_path is None:
        embeddings_path = config["embeddings_path"]

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embeddings_path
    )

    return embeddings


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)


# ─────────────────────────────────────────────
# VECTOR DATABASE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to document store…")
def load_vectordb(_embeddings):
    """Create/connect to the persistent Chroma vector store.

    Cached with st.cache_resource — this is the real, root-cause fix
    for the chromadb singleton error described below: with caching,
    chromadb.PersistentClient(path="chroma_db") is now only ever
    called ONCE per process, no matter how many times a PDFChatChain
    gets constructed across reruns. The defensive try/except for the
    ValueError is kept as a safety net for any path that still bypasses
    the cache (e.g. a different process, or cache invalidation), but
    with caching in place it should no longer trigger in normal use.

    NOTE: the leading underscore on `_embeddings` tells st.cache_resource
    not to try hashing that argument (embeddings objects aren't
    reliably hashable) — caching is keyed on the other args only, which
    is fine here since there's normally a single embeddings model.

    NOTE 1 — telemetry: anonymized_telemetry=False disables ChromaDB's
    built-in PostHog telemetry. Without this, some chromadb versions
    call posthog.capture(user_id, event_name, properties) with 3
    positional arguments, but newer posthog releases made capture()
    keyword-only — producing a harmless but noisy "Failed to send
    telemetry event ...: capture() takes 1 positional argument but 3
    were given" log line every time a PersistentClient is created or
    queried. It never crashes the app on its own, but disabling
    telemetry avoids the noise and any network call to PostHog.

    NOTE 2 — singleton settings mismatch (now prevented by caching,
    kept as a fallback): chromadb.PersistentClient is a per-path
    SINGLETON within a single Python process (tracked internally by
    SharedSystemClient._identifer_to_system). If a PersistentClient for
    this same `path` was ever created earlier in this process with
    DIFFERENT settings — e.g. before caching was added, or from any
    other code path that touched this same chroma_db folder first —
    chromadb raises:
        ValueError: An instance of Chroma already exists for
        <identifier> with different settings
    instead of reusing the existing instance. Since the underlying
    store is already open and perfectly usable, we catch that specific
    error and fall back to connecting WITHOUT forcing our settings,
    rather than crashing the whole app. A full restart of the
    Streamlit server (not just a code save/hot-reload) clears any
    stale state from before this fix.
    """
    try:
        persistent_client = chromadb.PersistentClient(
            path="chroma_db",
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
    except ValueError as e:
        if "already exists" in str(e):
            persistent_client = chromadb.PersistentClient(path="chroma_db")
        else:
            raise

    vector_db = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=_embeddings
    )

    return vector_db


# ─────────────────────────────────────────────
# HISTORY FORMATTING
#
# memory_prompt_template expects {history} as plain TEXT (it's a raw
# Mistral-style "<s>[INST] ... Previous conversation: {history} ...
# [/INST]" string template, not a ChatPromptTemplate placeholder for a
# list of message objects). So we render the last K turns into a
# "Human: ...\nAI: ...\n" transcript here ourselves.
#
# NOTE: the [INST]/<s> tokens are Mistral-specific and are kept as-is
# per the "preserve all prompt templates" requirement. Llama models
# served by Groq don't use these tokens, but they arrive as ordinary
# text inside the single human message LangChain builds from this
# rendered prompt string, so Llama simply treats them as harmless
# literal text rather than special tokens — generation still works
# normally, it's just not doing anything useful for a Llama model.
# ─────────────────────────────────────────────
def _format_history_text(messages, k=DEFAULT_MEMORY_K):
    """Render the last *k* human/AI turns from *messages* as plain text.

    `k` counts turns (a human+ai pair), matching the old
    ConversationBufferWindowMemory(k=3) behavior, not raw message count.
    """
    if not messages:
        return ""

    windowed = messages[-(k * 2):] if k > 0 else messages

    lines = []
    for msg in windowed:
        msg_type = getattr(msg, "type", None)
        if msg_type == "human":
            lines.append(f"Human: {msg.content}")
        elif msg_type == "ai":
            lines.append(f"AI: {msg.content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN WRAPPERS  (used by app.py)
# ─────────────────────────────────────────────
def load_normal_chain(chat_history, memory_k=None, model_name=None):
    return ChatChain(
        chat_history, memory_k=memory_k or DEFAULT_MEMORY_K, model_name=model_name
    )


def load_pdf_chat_chain(chat_history, memory_k=None, model_name=None):
    return PDFChatChain(
        chat_history, memory_k=memory_k or DEFAULT_MEMORY_K, model_name=model_name
    )


# ─────────────────────────────────────────────
# NORMAL CHAT CHAIN  (LCEL, supports real .stream())
# ─────────────────────────────────────────────
class ChatChain:
    """Wraps a `prompt | llm | StrOutputParser` LCEL pipeline with
    message-history injection, exposing both .stream() and .run() so
    app.py's `for chunk in llm_chain.stream(...)` actually streams
    tokens instead of silently falling back to a single blocking call.

    `llm` is now a ChatGroq chat model instead of a local CTransformers
    LLM. StrOutputParser transparently converts the AIMessage/
    AIMessageChunk objects a chat model produces into plain strings,
    so .stream()/.run() still return/yield plain str exactly as before
    — nothing downstream in app.py needed to change for this.

    IMPORTANT: `chat_history` is the SAME StreamlitChatMessageHistory
    object app.py holds. Its `.messages` setter writes directly into
    st.session_state["history"] (the FULL conversation) — so this class
    must never assign to chat_history.messages itself, and must do its
    own windowing on a local copy instead of mutating the shared object.
    Doing otherwise previously caused the windowed (truncated) message
    list to silently overwrite the full saved history.
    """

    def __init__(self, chat_history, memory_k=DEFAULT_MEMORY_K, model_name=None):
        self.chat_history = chat_history
        self.memory_k = memory_k

        llm = create_llm(model_name=model_name)
        prompt = create_prompt_from_template(memory_prompt_template)

        # history is filled in right before invoke/stream from
        # self.chat_history.messages (read-only access — never written).
        self.chain = prompt | llm | StrOutputParser()

    def _build_inputs(self, user_input: str) -> dict:
        history_text = _format_history_text(
            self.chat_history.messages, k=self.memory_k
        )
        return {"history": history_text, "human_input": user_input}

    def stream(self, user_input: str):
        """Real token-by-token streaming via the underlying LCEL chain."""
        inputs = self._build_inputs(user_input)
        for chunk in self.chain.stream(inputs):
            yield chunk

    def run(self, user_input: str) -> str:
        """Non-streaming fallback — used by app.py if .stream() raises."""
        inputs = self._build_inputs(user_input)
        return self.chain.invoke(inputs)


# ─────────────────────────────────────────────
# PDF CHAT CHAIN  (LCEL retrieval + memory, supports real .stream())
# ─────────────────────────────────────────────
class PDFChatChain:
    """Retrieval-augmented chat over the PDF vector store.

    Unlike the old RetrievalQA-based version, this one actually uses
    conversation history: retrieved context AND the recent
    conversation transcript are both injected into the prompt.
    """

    PDF_TEMPLATE = (
        "<s>[INST] You are an AI chatbot answering questions about the "
        "user's uploaded documents. Use the context below to answer; if "
        "the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Previous conversation:\n{history}\n\n"
        "Human: {human_input}\n"
        "AI: [/INST]"
    )

    def __init__(self, chat_history, memory_k=DEFAULT_MEMORY_K, k_documents=3, model_name=None):
        self.chat_history = chat_history
        self.memory_k = memory_k

        self.llm = create_llm(model_name=model_name)
        self.vector_db = load_vectordb(create_embeddings())
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": k_documents}
        )

        prompt = create_prompt_from_template(self.PDF_TEMPLATE)
        self.chain = prompt | self.llm | StrOutputParser()

    def _retrieve_context(self, user_input: str) -> str:
        docs = self.retriever.invoke(user_input)
        if not docs:
            return "(no relevant documents found)"
        return "\n\n".join(d.page_content for d in docs)

    def _build_inputs(self, user_input: str) -> dict:
        history_text = _format_history_text(
            self.chat_history.messages, k=self.memory_k
        )
        context_text = self._retrieve_context(user_input)
        return {
            "context": context_text,
            "history": history_text,
            "human_input": user_input,
        }

    def stream(self, user_input: str):
        inputs = self._build_inputs(user_input)
        for chunk in self.chain.stream(inputs):
            yield chunk

    def run(self, user_input: str) -> str:
        inputs = self._build_inputs(user_input)
        return self.chain.invoke(inputs)
