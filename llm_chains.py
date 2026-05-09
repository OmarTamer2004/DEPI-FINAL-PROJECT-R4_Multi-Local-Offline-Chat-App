from prompt_templates import memory_prompt_template

from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings
)

from langchain_community.llms import CTransformers

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory

from langchain.chains import (
    LLMChain,
    RetrievalQA
)

import chromadb
import yaml

# ---------------- LOAD CONFIG ----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------------- CREATE LLM ----------------
def create_llm(
    model_path=None,
    model_type=None,
    model_config=None
):

    if model_path is None:
        model_path = config["model_path"]["large"]

    if model_type is None:
        model_type = config["model_type"]

    if model_config is None:
        model_config = config["model_config"]

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        config=model_config
    )

    return llm

# ---------------- CREATE EMBEDDINGS ----------------
def create_embeddings(embeddings_path=None):

    if embeddings_path is None:
        embeddings_path = config["embeddings_path"]

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embeddings_path
    )

    return embeddings

# ---------------- CHAT MEMORY ----------------
def create_chat_memory(chat_history):

    memory = ConversationBufferWindowMemory(
        memory_key="history",
        chat_memory=chat_history,
        k=3,
        return_messages=True
    )

    return memory

# ---------------- PROMPT TEMPLATE ----------------
def create_prompt_from_template(template):

    prompt = PromptTemplate.from_template(template)

    return prompt

# ---------------- NORMAL LLM CHAIN ----------------
def create_llm_chain(llm, chat_prompt, memory):

    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory
    )

    return chain

# ---------------- VECTOR DATABASE ----------------
def load_vectordb(embeddings):

    persistent_client = chromadb.PersistentClient(
        path="chroma_db"
    )

    vector_db = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings
    )

    return vector_db

# ---------------- RETRIEVAL QA CHAIN ----------------
def load_retrieval_chain(llm, vector_db):

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_kwargs={"k": 3}
        ),
        return_source_documents=True
    )

    return retrieval_chain

# ---------------- MAIN WRAPPERS ----------------
def load_normal_chain(chat_history):

    return ChatChain(chat_history)

def load_pdf_chat_chain(chat_history):

    return PDFChatChain(chat_history)

# ---------------- PDF CHAT CHAIN ----------------
class PDFChatChain:

    def __init__(self, chat_history):

        self.memory = create_chat_memory(
            chat_history
        )

        self.vector_db = load_vectordb(
            create_embeddings()
        )

        self.llm = create_llm()

        self.qa_chain = load_retrieval_chain(
            self.llm,
            self.vector_db
        )

    def run(self, user_input):

        print("PDF Chat Chain Running...")

        result = self.qa_chain({
            "query": user_input
        })

        return result["result"]

# ---------------- NORMAL CHAT CHAIN ----------------
class ChatChain:

    def __init__(self, chat_history):

        self.memory = create_chat_memory(
            chat_history
        )

        llm = create_llm()

        chat_prompt = create_prompt_from_template(
            memory_prompt_template
        )

        self.llm_chain = create_llm_chain(
            llm,
            chat_prompt,
            self.memory
        )

    def run(self, user_input):

        response = self.llm_chain.run(
            user_input
        )

        return response