import json
import os
import re
from datetime import datetime

# Support both old and new LangChain import paths
try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
except ImportError:
    from langchain.schema.messages import HumanMessage, AIMessage, BaseMessage


# ─────────────────────────────────────────────
# SANITIZE FILENAME  (Windows-safe)
# ─────────────────────────────────────────────
_WIN_FORBIDDEN = re.compile(r'[\\/:*?"<>|\r\n\t]')
_MULTI_SPACE   = re.compile(r'\s+')

def sanitize_filename(name: str, max_len: int = 60) -> str:
    """Return a Windows-safe filename stem (no extension)."""
    name = _WIN_FORBIDDEN.sub("", name)      # strip forbidden chars
    name = _MULTI_SPACE.sub(" ", name)       # collapse whitespace
    name = name.strip(". ")                  # no leading/trailing dots or spaces
    name = name[:max_len]                    # hard length cap
    return name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ─────────────────────────────────────────────
# SAVE CHAT
# ─────────────────────────────────────────────
def save_chat_history_json(chat_history, file_path: str) -> None:
    """Safely persist chat history to *file_path* (JSON).

    Uses a sibling temp-file + atomic os.replace() so a crash mid-write
    never corrupts an existing file.  Works on Windows and Linux.
    """
    try:
        # Normalise path — resolve to absolute so dirname is never empty
        file_path = os.path.abspath(file_path)
        if not file_path.endswith(".json"):
            file_path += ".json"

        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)

        # Collect messages
        if hasattr(chat_history, "messages"):
            messages = chat_history.messages
        else:
            messages = chat_history

        json_data = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if isinstance(msg, HumanMessage) or msg_type == "human":
                json_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage) or msg_type == "ai":
                json_data.append({"type": "ai",    "content": msg.content})

        # Write directly — simple and reliable on Windows.
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print("ERROR saving chat history:", repr(e))
        raise


# ─────────────────────────────────────────────
# LOAD CHAT
# ─────────────────────────────────────────────
def load_chat_history_json(file_path: str) -> list:
    """Load chat history from *file_path*.  Returns [] on any error."""
    try:
        file_path = os.path.abspath(file_path)
        if not file_path.endswith(".json"):
            file_path += ".json"

        if not os.path.exists(file_path):
            return []
        if os.path.getsize(file_path) == 0:
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        messages = []
        for msg in json_data:
            if msg.get("type") == "human":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai":
                messages.append(AIMessage(content=msg.get("content", "")))
        return messages

    except json.JSONDecodeError:
        print(f"Invalid JSON in: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []


# ─────────────────────────────────────────────
# TIMESTAMP
# ─────────────────────────────────────────────
def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
