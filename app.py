import streamlit as st
import os
import yaml
import json
import psutil
import threading
import glob
from datetime import datetime
from pathlib import Path

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder

from llm_chains import load_normal_chain, load_pdf_chat_chain, create_llm
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from utils import save_chat_history_json, load_chat_history_json, get_timestamp, sanitize_filename

# ─────────────────────────────────────────────
#  Optional GPU monitoring (graceful fallback)
# ─────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

history_path     = config.get("chat_history_path", "chat_history")
models_dir       = config.get("models_dir", "models")
available_models = config.get("available_models", ["default"])

DEFAULT_SYSTEM_PROMPT = config.get(
    "system_prompt",
    "You are a helpful, honest, and concise AI assistant."
)

DEFAULT_MEMORY_WINDOW = config.get("memory_window", 10)

os.makedirs(history_path, exist_ok=True)
os.makedirs(models_dir,   exist_ok=True)

# ─────────────────────────────────────────────
#  DARK UI + CUSTOM STYLING
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>

    /* ── Base dark theme ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b25 !important;
        border-right: 1px solid #2a2f3e;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"] {
        background-color: #1c2130;
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        position: relative;
    }

    /* ── Copy button inside chat bubble ── */
    .copy-btn-wrapper {
        display: flex;
        justify-content: flex-end;
        margin-top: 6px;
    }
    .copy-btn {
        background: #1a2540;
        border: 1px solid #2a3550;
        border-radius: 6px;
        color: #6e9fd4 !important;
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 3px 10px;
        transition: all 0.18s ease;
        user-select: none;
    }
    .copy-btn:hover {
        background: #223060;
        color: #93c5fd !important;
        border-color: #3b5998;
    }

    /* ── Input box ── */
    .stTextInput > div > div > input {
        background-color: #1c2130;
        color: #e0e0e0;
        border: 1px solid #30363d;
        border-radius: 8px;
        font-size: 15px;
        padding: 10px 14px;
    }

    /* ── Textarea (system prompt) ── */
    .stTextArea > div > div > textarea {
        background-color: #1c2130 !important;
        color: #e0e0e0 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-family: 'Consolas', monospace !important;
    }

    /* ── All buttons base ── */
    .stButton > button {
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 15px;
        padding: 10px 0;
        transition: all 0.2s ease;
        width: 100%;
        letter-spacing: 0.3px;
    }

    /* ── Send button ── */
    .send-btn > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(35,134,54,0.35);
    }
    .send-btn > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950) !important;
        box-shadow: 0 4px 14px rgba(35,134,54,0.5);
        transform: translateY(-1px);
    }
    .send-btn > button:active { transform: translateY(0px); }

    /* ── Stop button ── */
    .stop-btn > button {
        background: linear-gradient(135deg, #b91c1c, #da3633) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(218,54,51,0.35);
    }
    .stop-btn > button:hover {
        background: linear-gradient(135deg, #da3633, #f85149) !important;
        box-shadow: 0 4px 14px rgba(218,54,51,0.5);
        transform: translateY(-1px);
    }
    .stop-btn > button:active { transform: translateY(0px); }

    /* ── Mic button ── */
    .mic-btn > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35);
        border-radius: 10px;
        font-weight: 700;
        font-size: 14px;
    }
    .mic-btn > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        transform: translateY(-1px);
    }

    /* ── Action row wrapper ── */
    .action-row {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 8px;
        padding: 10px 12px;
        background-color: #161b25;
        border: 1px solid #2a2f3e;
        border-radius: 14px;
    }

    /* ── Code blocks ── */
    code, pre {
        background-color: #161b22 !important;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-size: 0.85em;
    }

    /* ── Monitor card ── */
    .monitor-card {
        background: linear-gradient(145deg, #1a2035, #1c2540);
        border: 1px solid #2a3550;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 4px;
    }
    .monitor-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6e7f9f !important;
        margin-bottom: 10px;
    }
    .monitor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .monitor-label { font-size: 12px; color: #8b9ab5 !important; }
    .monitor-value { font-size: 13px; font-weight: 700; color: #e2e8f0 !important; }
    .progress-bar-bg {
        width: 100%; background-color: #1e2d45;
        border-radius: 99px; height: 6px; margin-bottom: 8px; overflow: hidden;
    }
    .progress-bar-fill-green  { height:6px;border-radius:99px;background:linear-gradient(90deg,#22c55e,#4ade80); }
    .progress-bar-fill-blue   { height:6px;border-radius:99px;background:linear-gradient(90deg,#3b82f6,#60a5fa); }
    .progress-bar-fill-purple { height:6px;border-radius:99px;background:linear-gradient(90deg,#a855f7,#c084fc); }
    .progress-bar-fill-orange { height:6px;border-radius:99px;background:linear-gradient(90deg,#f97316,#fb923c); }
    .monitor-divider { border:none;border-top:1px solid #2a3550;margin:10px 0; }
    .gpu-badge    { display:inline-block;background:#16213a;border:1px solid #2a3550;border-radius:6px;padding:2px 8px;font-size:10px;color:#60a5fa !important;font-weight:600; }
    .no-gpu-badge { display:inline-block;background:#1a1a2e;border:1px solid #3a3a4a;border-radius:6px;padding:2px 8px;font-size:10px;color:#6e7f9f !important; }

    /* ── Title ── */
    h1 { color: #58a6ff !important; }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #1c2130;
        color: #e0e0e0;
        border: 1px solid #30363d;
    }

    /* ── Toggle ── */
    .stToggle { color: #58a6ff !important; }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] { padding: 4px 0; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    /* ── Remove iframe white border (st.components.v1.html) ── */
    [data-testid="stSidebar"] iframe,
    [data-testid="stSidebar"] iframe:focus {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        display: block;
    }

    /* ── Typing animation ── */
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
    .typing-cursor::after {
        content: '▋';
        animation: blink 0.7s infinite;
        color: #58a6ff;
    }

    /* ── System prompt expander badge ── */
    .sys-badge {
        display: inline-block;
        background: #1a2540;
        border: 1px solid #2a3550;
        border-radius: 6px;
        padding: 2px 9px;
        font-size: 10px;
        color: #f59e0b !important;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-left: 6px;
    }

    </style>

    <!-- ── Copy-to-clipboard JS ── -->
    <script>
    function copyText(id) {
        const el = document.getElementById(id);
        if (!el) return;
        navigator.clipboard.writeText(el.innerText).then(() => {
            const btn = document.querySelector('[onclick="copyText(\\''+id+'\\')"]');
            if (btn) { btn.innerText = '✅ Copied!'; setTimeout(()=>{ btn.innerText='📋 Copy'; }, 1500); }
        });
    }
    </script>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HARDWARE MONITOR
#  NOTE: No `with st.sidebar:` here — this function
#  must be called from INSIDE a `with st.sidebar:` block.
#  The old nested sidebar context caused raw HTML to
#  leak into the main chat area.
# ─────────────────────────────────────────────
def get_system_stats():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    gpu_util = gpu_mem = gpu_name = None

    if GPU_AVAILABLE:
        try:
            handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem  = round(mem_info.used / mem_info.total * 100, 1)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode()
        except Exception:
            pass

    return cpu, ram, gpu_util, gpu_mem, gpu_name


def render_monitor():
    """Render the hardware monitor with a clean card using st.markdown.
    HTML is written with NO leading-space indentation so Markdown's
    4-space code-block rule never triggers.
    Must be called inside a `with st.sidebar:` block.
    """
    cpu, ram, gpu_util, gpu_mem, gpu_name = get_system_stats()

    # ── helpers ──────────────────────────────────────────────────────
    def row(icon, label, value, bar_color, pct):
        # Each tag starts at column 0 — safe from markdown code-block rule
        return (
'<div style="margin-bottom:10px;">' +
'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
f'<span style="font-size:13px;color:#c9d1d9;">{icon} {label}</span>' +
f'<span style="font-size:13px;font-weight:700;color:#e2e8f0;">{value}%</span>' +
'</div>' +
'<div style="background:#1e2d45;border-radius:99px;height:7px;overflow:hidden;">' +
f'<div style="width:{pct}%;height:7px;border-radius:99px;background:{bar_color};"></div>' +
'</div>' +
'</div>'
        )

    # ── build GPU block ───────────────────────────────────────────────
    if gpu_util is not None:
        short = (gpu_name[:20] + "…") if gpu_name and len(gpu_name) > 22 else (gpu_name or "GPU")
        gpu_block = (
'<div style="border-top:1px solid #2a3550;margin:8px 0 10px;"></div>' +
f'<div style="font-size:11px;font-weight:700;color:#6e7f9f;letter-spacing:.8px;margin-bottom:8px;">🎮 GPU &nbsp;<span style="background:#16213a;border:1px solid #2a3550;border-radius:5px;padding:1px 7px;font-size:10px;color:#60a5fa;">{short}</span></div>' +
row("⚡", "Utilization", gpu_util, "linear-gradient(90deg,#a855f7,#c084fc)", gpu_util) +
row("🗂", "VRAM", gpu_mem, "linear-gradient(90deg,#f97316,#fb923c)", gpu_mem)
        )
    else:
        gpu_block = (
'<div style="border-top:1px solid #2a3550;margin:8px 0 10px;"></div>' +
'<div style="display:flex;justify-content:space-between;align-items:center;">' +
'<span style="font-size:13px;color:#c9d1d9;">🎮 GPU</span>' +
'<span style="background:#1a1a2e;border:1px solid #3a3a4a;border-radius:5px;padding:2px 8px;font-size:10px;color:#6e7f9f;">Not Available</span>' +
'</div>'
        )

    # ── assemble card — every line starts at column 0 ─────────────────
    card = (
'<div style="background:linear-gradient(145deg,#1a2035,#1c2540);border:1px solid #2a3550;border-radius:14px;padding:14px 16px;margin-bottom:4px;">' +
'<div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#6e7f9f;margin-bottom:12px;">🖥 System Monitor</div>' +
row("🧠", "CPU", cpu, "linear-gradient(90deg,#22c55e,#4ade80)", cpu) +
row("💾", "RAM", ram, "linear-gradient(90deg,#3b82f6,#60a5fa)", ram) +
gpu_block +
'</div>'
    )

    st.markdown(card, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  GGUF MODEL SCANNER
# ─────────────────────────────────────────────
def get_gguf_models():
    files  = glob.glob(os.path.join(models_dir, "**/*.gguf"), recursive=True)
    files += glob.glob(os.path.join(models_dir, "*.gguf"))
    return [os.path.basename(p) for p in files] or ["No GGUF models found"]


# ─────────────────────────────────────────────
#  CHAIN LOADER
# ─────────────────────────────────────────────
def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)


# ─────────────────────────────────────────────
#  SESSION HELPERS
# ─────────────────────────────────────────────
def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input    = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def generate_session_name(question):
    try:
        llm    = create_llm()
        prompt = (
            "Generate a short chat title (3 to 5 words only).\n"
            f"User Message:\n{question}\n"
            "Rules:\n- Return title only\n- No quotes\n- No special characters"
        )
        raw   = llm.invoke(prompt).strip().replace(" ", "_")
        title = sanitize_filename(raw) or get_timestamp()
        base  = title
        i = 1
        while os.path.exists(os.path.join(history_path, title + ".json")):
            title = f"{base}_{i}"; i += 1
        return title
    except Exception:
        return get_timestamp()


# ─────────────────────────────────────────────
#  SAVE / LOAD HISTORY
# ─────────────────────────────────────────────
def save_chat_history():
    """Persist the current chat history to disk.

    IMPORTANT: never write to st.session_state.session_key directly —
    that key is owned by the selectbox widget and Streamlit raises
    StreamlitAPIException if you modify it from code.
    Instead we update only session_index_tracker, which controls the
    selectbox's `index=` parameter.  The widget will pick up the new
    value on the next rerun and set session_key itself.
    """
    if not st.session_state.history:
        return

    if st.session_state.session_key == "new_session":
        if st.session_state.new_session_key is None:
            return                          # title not generated yet — skip
        filename = st.session_state.new_session_key
        if not filename.endswith(".json"):
            filename += ".json"
        # Store normalised filename so next rerun the selectbox lands on it.
        st.session_state.new_session_key       = filename
        # ── Do NOT touch session_key (widget-owned) ──
        st.session_state.session_index_tracker = filename
    else:
        filename = st.session_state.session_key
        if not filename.endswith(".json"):
            filename += ".json"

    path = os.path.join(history_path, filename)
    os.makedirs(history_path, exist_ok=True)
    save_chat_history_json(st.session_state.history, path)


# ─────────────────────────────────────────────
#  EXPORT CHAT
# ─────────────────────────────────────────────
def export_chat_as_text():
    lines = []
    for msg in st.session_state.history:
        role = "You" if msg.type == "human" else "Assistant"
        lines.append(f"[{role}]\n{msg.content}\n")
    return "\n".join(lines)

def export_chat_as_json():
    data = [{"role": m.type, "content": m.content} for m in st.session_state.history]
    return json.dumps(data, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
#  STREAMING RESPONSE
# ─────────────────────────────────────────────
def stream_response(llm_chain, question, container, stop_event):
    full_text   = ""
    placeholder = container.empty()

    system_prompt = st.session_state.get("system_prompt_text", "").strip()
    if system_prompt:
        question = f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n{question}"

    try:
        for chunk in llm_chain.stream(question):
            if stop_event.is_set():
                break
            token      = chunk if isinstance(chunk, str) else getattr(chunk, "content", str(chunk))
            full_text += token
            placeholder.markdown(
                f'<div class="typing-cursor">{full_text}</div>',
                unsafe_allow_html=True,
            )
        placeholder.markdown(full_text)

    except (AttributeError, NotImplementedError):
        with st.spinner("Thinking..."):
            full_text = llm_chain.run(question)
        placeholder.markdown(full_text)

    return full_text


# ─────────────────────────────────────────────
#  RENDER MESSAGE  (markdown + code + copy btn)
# ─────────────────────────────────────────────
_copy_counter = 0

def render_message(msg):
    global _copy_counter
    with st.chat_message(msg.type):
        parts = msg.content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lines = part.split("\n", 1)
                lang  = lines[0].strip() if lines else ""
                code  = lines[1] if len(lines) > 1 else part
                st.code(code, language=lang or None)
            else:
                if part.strip():
                    st.markdown(part)

        # Copy button — only for assistant messages
        if msg.type == "ai":
            _copy_counter += 1
            uid = f"msg_copy_{_copy_counter}"
            st.markdown(
                f"""
                <span id="{uid}" style="display:none">{msg.content.replace('"', '&quot;').replace('<', '&lt;')}</span>
                <div class="copy-btn-wrapper">
                  <button class="copy-btn" onclick="copyText('{uid}')">📋 Copy</button>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global _copy_counter
    _copy_counter = 0

    st.set_page_config(
        page_title="Pro Chat App",
        page_icon="🤖",
        layout="wide",
    )

    inject_css()
    st.title("🤖 Multi-Model Offline Chat")

    chat_container = st.container()

    # ── Session state defaults ──────────────────
    defaults = {
        "send_input":            False,
        "user_question":         "",
        "new_session_key":       None,
        "session_key":           "new_session",
        "session_index_tracker": "new_session",
        "history":               [],
        "pdf_chat":              False,
        "pdf_processed":         False,
        "stop_generation":       False,
        "selected_model":        available_models[0] if available_models else "default",
        "selected_gguf":         None,
        "rename_mode":           False,
        "system_prompt_text":    DEFAULT_SYSTEM_PROMPT,
        "memory_window":         DEFAULT_MEMORY_WINDOW,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ══════════════════════════════════════════
    #  SIDEBAR
    #  render_monitor() is called here — do NOT
    #  wrap it again inside its own function.
    # ══════════════════════════════════════════
    with st.sidebar:
        st.title("⚙️ Settings")

        # ── Hardware monitor lives entirely here ──
        render_monitor()    # <-- single sidebar context; no nested `with st.sidebar`
        st.divider()

        # ── Model selector ───────────────────────
        st.markdown("### 🧠 Model")
        st.selectbox("Active Model", available_models, key="selected_model")

        gguf_models = get_gguf_models()
        if gguf_models[0] != "No GGUF models found":
            st.selectbox("GGUF File", gguf_models, key="selected_gguf")
        else:
            st.caption("📁 Drop .gguf files into `models/` to enable GGUF switching.")

        st.divider()

        # ── SYSTEM PROMPT ─────────────────────
        st.markdown("### 🗒 System Prompt")
        with st.expander("Edit System Prompt", expanded=False):
            st.text_area(
                label="System Prompt",
                key="system_prompt_text",
                height=160,
                help="This prompt is prepended to every message you send.",
                label_visibility="collapsed",
            )
            col_reset, col_save = st.columns(2)
            with col_reset:
                if st.button("↩ Reset", use_container_width=True):
                    st.session_state.system_prompt_text = DEFAULT_SYSTEM_PROMPT
                    st.rerun()
            with col_save:
                if st.button("💾 Save to YAML", use_container_width=True):
                    try:
                        with open("config.yaml", "r") as f:
                            cfg = yaml.safe_load(f)
                        cfg["system_prompt"] = st.session_state.system_prompt_text
                        with open("config.yaml", "w") as f:
                            yaml.dump(cfg, f, allow_unicode=True)
                        st.success("Saved!")
                    except Exception as e:
                        st.error(str(e))

        st.divider()

        # ── MEMORY WINDOW ─────────────────────
        st.markdown("### 🧩 Memory Window")
        st.slider(
            "Messages to keep in context",
            min_value=2,
            max_value=50,
            step=2,
            key="memory_window",
            help="Limits how many past messages the model sees.",
        )
        st.caption(
            f"🔵 Keeping last **{st.session_state.memory_window}** messages in context."
        )

        st.divider()

        # ── PDF Chat toggle ──────────────────────
        st.toggle("📄 PDF Chat Mode", key="pdf_chat", value=False)
        st.divider()

        # ── Session list ─────────────────────────
        st.markdown("### 💬 Chat Sessions")

        session_files = sorted(
            [
                (f, os.path.getmtime(os.path.join(history_path, f)))
                for f in os.listdir(history_path)
                if f.endswith(".json")
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        chat_sessions = ["new_session"] + [f for f, _ in session_files]

        # ── FIX: clamp tracker to valid options ──────────────────────────
        tracker = st.session_state.session_index_tracker
        index   = chat_sessions.index(tracker) if tracker in chat_sessions else 0

        st.selectbox(
            "Select session",
            chat_sessions,
            index=index,
            key="session_key",
            on_change=track_index,
        )

        # ── Session actions ──────────────────────
        if st.session_state.session_key != "new_session":
            col_r, col_d = st.columns(2)

            with col_r:
                if st.button("✏️ Rename", use_container_width=True):
                    st.session_state.rename_mode = not st.session_state.rename_mode

            with col_d:
                if st.button("🗑️ Delete", use_container_width=True):
                    path = os.path.join(history_path, st.session_state.session_key)
                    if os.path.exists(path):
                        os.remove(path)
                    st.session_state.session_key           = "new_session"
                    st.session_state.session_index_tracker = "new_session"
                    st.session_state.new_session_key       = None
                    st.session_state.history               = []
                    st.rerun()

            if st.session_state.rename_mode:
                new_name = st.text_input(
                    "New name",
                    value=st.session_state.session_key.replace(".json", ""),
                )
                if st.button("✅ Confirm Rename"):
                    old_path = os.path.join(history_path, st.session_state.session_key)
                    new_key  = new_name.strip().replace(" ", "_") + ".json"
                    new_path = os.path.join(history_path, new_key)
                    if old_path != new_path and not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        st.session_state.session_key           = new_key
                        st.session_state.session_index_tracker = new_key
                    st.session_state.rename_mode = False
                    st.rerun()

        st.divider()

        # ── Export ───────────────────────────────
        if st.session_state.history:
            st.markdown("### 📤 Export Chat")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.download_button(
                    "📝 TXT",
                    data=export_chat_as_text(),
                    file_name="chat_export.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_e2:
                st.download_button(
                    "📦 JSON",
                    data=export_chat_as_json(),
                    file_name="chat_export.json",
                    mime="application/json",
                    use_container_width=True,
                )
            st.divider()

        # ── File uploads ─────────────────────────
        uploaded_audio = st.file_uploader(
            "🎵 Upload Audio", type=["wav", "mp3", "ogg"]
        )
        uploaded_pdf = st.file_uploader(
            "📄 Upload PDFs",
            accept_multiple_files=True,
            key="pdf_upload",
            type=["pdf"],
        )

    if uploaded_pdf and not st.session_state.pdf_processed:
        with st.spinner("Processing PDF…"):
            try:
                add_documents_to_db(uploaded_pdf)
                st.session_state.pdf_processed = True
                st.sidebar.success("✅ PDF added!")
            except Exception as e:
                st.sidebar.error(str(e))

    # ══════════════════════════════════════════
    #  LOAD HISTORY
    # ══════════════════════════════════════════
    # Determine which file to load.
    # After saving a brand-new session, session_key is still "new_session"
    # (widget-owned, can't be changed in code) but new_session_key holds the
    # real filename.  Use new_session_key as a fallback in that case.
    _active_key = st.session_state.session_key
    if _active_key == "new_session" and st.session_state.new_session_key:
        _active_key = st.session_state.new_session_key

    if _active_key != "new_session":
        try:
            st.session_state.history = load_chat_history_json(
                os.path.join(history_path, _active_key)
            )
        except Exception:
            st.session_state.history = []
    else:
        # Truly new conversation — only reset if nothing is in memory yet.
        if not st.session_state.get("history"):
            st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")

    # ── Memory window: give LLM only the last N messages ──
    window = st.session_state.memory_window
    _full_history_backup = list(st.session_state.history)
    if window > 0 and len(st.session_state.history) > window:
        st.session_state.history = st.session_state.history[-window:]

    llm_chain = load_chain(chat_history)

    # ══════════════════════════════════════════
    #  RENDER CHAT
    # ══════════════════════════════════════════
    with chat_container:
        for msg in _full_history_backup:   # always render the FULL history
            render_message(msg)

    # ══════════════════════════════════════════
    #  INPUT BOX
    # ══════════════════════════════════════════
    st.text_input(
        "Type your message…",
        key="user_input",
        on_change=set_send_input,
        placeholder="Ask anything…",
    )

    # ══════════════════════════════════════════
    #  ACTION ROW  🎤 | 📨 Send | 🛑 Stop
    # ══════════════════════════════════════════
    st.markdown('<div class="action-row">', unsafe_allow_html=True)
    col_mic, col_send, col_stop = st.columns([1.4, 1, 1])

    with col_mic:
        st.markdown('<div class="mic-btn">', unsafe_allow_html=True)
        voice_recording = mic_recorder(
            start_prompt="🎤  Start Recording",
            stop_prompt="⏹  Stop Recording",
            just_once=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_send:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send_button = st.button(
            "📨  Send",
            on_click=clear_input_field,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stop:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        stop_button = st.button("🛑  Stop", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if stop_button:
            st.session_state.stop_generation = True

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Stop event ────────────────────────────
    stop_event = threading.Event()
    if st.session_state.stop_generation:
        stop_event.set()
        st.session_state.stop_generation = False

    # ══════════════════════════════════════════
    #  AUDIO FILE HANDLER
    # ══════════════════════════════════════════
    if uploaded_audio:
        try:
            with st.spinner("Transcribing audio…"):
                transcribed = transcribe_audio(uploaded_audio.getvalue())
            with chat_container:
                st.chat_message("human").write(transcribed)
            with chat_container:
                resp_box = st.chat_message("assistant").empty()
            ai_reply = stream_response(llm_chain, "Summarize this text:\n" + transcribed, resp_box, stop_event)
            from langchain_core.messages import HumanMessage, AIMessage
            st.session_state.history = (
                _full_history_backup
                + [HumanMessage(content=transcribed), AIMessage(content=ai_reply)]
            )
            save_chat_history()
        except Exception as e:
            st.error(str(e))

    # ══════════════════════════════════════════
    #  MIC RECORDING HANDLER
    # ══════════════════════════════════════════
    if voice_recording:
        try:
            with st.spinner("Transcribing voice…"):
                transcribed = transcribe_audio(voice_recording["bytes"])
            with chat_container:
                st.chat_message("human").write(transcribed)
            with chat_container:
                resp_box = st.chat_message("assistant").empty()
            ai_reply = stream_response(llm_chain, transcribed, resp_box, stop_event)
            from langchain_core.messages import HumanMessage, AIMessage
            st.session_state.history = (
                _full_history_backup
                + [HumanMessage(content=transcribed), AIMessage(content=ai_reply)]
            )
            save_chat_history()
        except Exception as e:
            st.error(str(e))

    # ══════════════════════════════════════════
    #  TEXT CHAT HANDLER
    # ══════════════════════════════════════════
    if send_button or st.session_state.send_input:
        question = st.session_state.user_question.strip()
        if question:
            st.session_state.send_input = False

            # Generate a persistent session name on first message
            if (
                st.session_state.session_key == "new_session"
                and st.session_state.new_session_key is None
            ):
                title = generate_session_name(question)
                if not title.endswith(".json"):
                    title += ".json"
                st.session_state.new_session_key = title

            with chat_container:
                st.chat_message("human").write(question)

            with chat_container:
                resp_box = st.chat_message("assistant").empty()

            ai_reply = stream_response(llm_chain, question, resp_box, stop_event)

            # Restore full history + append the new exchange
            from langchain_core.messages import HumanMessage, AIMessage
            st.session_state.history = (
                _full_history_backup
                + [HumanMessage(content=question), AIMessage(content=ai_reply)]
            )

            # ── save_chat_history now also promotes session_key ──
            save_chat_history()

            st.session_state.user_question = ""
            st.rerun()


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
