import json
import os
from datetime import datetime
from langchain.schema.messages import HumanMessage, AIMessage

# ---------------- SAVE CHAT ----------------
def save_chat_history_json(chat_history, file_path):

    # create folder if not exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    json_data = []

    # لو object فيه messages
    if hasattr(chat_history, "messages"):
        messages = chat_history.messages
    else:
        messages = chat_history

    # convert messages to json
    for message in messages:

        if isinstance(message, HumanMessage):

            json_data.append({
                "type": "human",
                "content": message.content
            })

        elif isinstance(message, AIMessage):

            json_data.append({
                "type": "ai",
                "content": message.content
            })

    # save json
    with open(file_path, "w", encoding="utf-8") as f:

        json.dump(
            json_data,
            f,
            ensure_ascii=False,
            indent=2
        )

# ---------------- LOAD CHAT ----------------
def load_chat_history_json(file_path):

    # file not exists
    if not os.path.exists(file_path):
        return []

    # empty file
    if os.path.getsize(file_path) == 0:
        return []

    try:

        with open(file_path, "r", encoding="utf-8") as f:

            json_data = json.load(f)

        messages = []

        for message in json_data:

            if message.get("type") == "human":

                messages.append(
                    HumanMessage(
                        content=message.get("content", "")
                    )
                )

            elif message.get("type") == "ai":

                messages.append(
                    AIMessage(
                        content=message.get("content", "")
                    )
                )

        return messages

    except json.JSONDecodeError:

        print(f"Invalid JSON file: {file_path}")
        return []

    except Exception as e:

        print(f"Error loading chat history: {e}")
        return []

# ---------------- TIMESTAMP ----------------
def get_timestamp():

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")