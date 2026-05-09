import streamlit as st

from llm_chains import (
    load_normal_chain,
    load_pdf_chat_chain,
    create_llm
)

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory
)

from streamlit_mic_recorder import mic_recorder

from audio_handler import transcribe_audio

from pdf_handler import add_documents_to_db

from utils import (
    save_chat_history_json,
    load_chat_history_json,
    get_timestamp
)

import os
import yaml

# ---------------- CONFIG ----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

history_path = config["chat_history_path"]

os.makedirs(history_path, exist_ok=True)

# ---------------- LOAD CHAIN ----------------
def load_chain(chat_history):

    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)

    return load_normal_chain(chat_history)

# ---------------- HELPERS ----------------
def clear_input_field():

    st.session_state.user_question = (
        st.session_state.user_input
    )

    st.session_state.user_input = ""

def set_send_input():

    st.session_state.send_input = True

    clear_input_field()

def track_index():

    st.session_state.session_index_tracker = (
        st.session_state.session_key
    )

# ---------------- SESSION NAME ----------------
def generate_session_name(question):

    llm = create_llm()

    prompt = f"""
    Generate a short chat title (3 to 5 words only).

    User Message:
    {question}

    Rules:
    - Return title only
    - No quotes
    - No special characters
    """

    try:

        response = llm.invoke(prompt)

        title = response.strip()

        invalid_chars = [
            '\\', '/', ';', ':', '*',
            '?', '"', '<', '>', '|'
        ]

        for c in invalid_chars:
            title = title.replace(c, "")

        title = title[:40]

        if title == "":
            title = get_timestamp()

        base_title = title

        counter = 1

        while os.path.exists(
            os.path.join(
                history_path,
                title + ".json"
            )
        ):

            title = f"{base_title}_{counter}"

            counter += 1

        return title

    except:

        return get_timestamp()

# ---------------- SAVE CHAT ----------------
def save_chat_history():

    if len(st.session_state.history) == 0:
        return

    if st.session_state.session_key == "new_session":

        if st.session_state.new_session_key is None:

            st.session_state.new_session_key = (
                get_timestamp() + ".json"
            )

        save_path = os.path.join(
            history_path,
            st.session_state.new_session_key
        )

    else:

        save_path = os.path.join(
            history_path,
            st.session_state.session_key
        )

    save_chat_history_json(
        st.session_state.history,
        save_path
    )

# ---------------- MAIN ----------------
def main():

    st.set_page_config(
        page_title="Offline Chat App",
        page_icon="💬",
        layout="wide"
    )

    st.title("Multi Local Offline Chat App 💬")

    chat_container = st.container()

    # ---------------- SESSION STATE ----------------
    if "send_input" not in st.session_state:
        st.session_state.send_input = False

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    if "new_session_key" not in st.session_state:
        st.session_state.new_session_key = None

    if "session_key" not in st.session_state:
        st.session_state.session_key = "new_session"

    if "session_index_tracker" not in st.session_state:
        st.session_state.session_index_tracker = (
            "new_session"
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    if "pdf_chat" not in st.session_state:
        st.session_state.pdf_chat = False

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Chat Sessions")

    chat_sessions = ["new_session"]

    # ---------------- SORT SESSIONS ----------------
    session_files = []

    for file in os.listdir(history_path):

        if file.endswith(".json"):

            full_path = os.path.join(
                history_path,
                file
            )

            session_files.append(
                (
                    file,
                    os.path.getmtime(full_path)
                )
            )

    # newest first
    session_files.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for file, _ in session_files:
        chat_sessions.append(file)

    # ---------------- SESSION SELECT ----------------
    if (
        st.session_state.session_index_tracker
        in chat_sessions
    ):

        index = chat_sessions.index(
            st.session_state.session_index_tracker
        )

    else:
        index = 0

    st.sidebar.selectbox(
        "Select a chat session",
        chat_sessions,
        index=index,
        key="session_key",
        on_change=track_index
    )

    # ---------------- PDF CHAT MODE ----------------
    st.sidebar.toggle(
        "PDF Chat Mode",
        key="pdf_chat",
        value=False
    )

    # ---------------- LOAD HISTORY ----------------
    if st.session_state.session_key != "new_session":

        try:

            st.session_state.history = (
                load_chat_history_json(
                    os.path.join(
                        history_path,
                        st.session_state.session_key
                    )
                )
            )

        except:

            st.session_state.history = []

    else:

        st.session_state.history = []

    # ---------------- CHAT HISTORY ----------------
    chat_history = StreamlitChatMessageHistory(
        key="history"
    )

    llm_chain = load_chain(chat_history)

    # ---------------- DISPLAY HISTORY ----------------
    with chat_container:

        for message in st.session_state.history:

            st.chat_message(
                message.type
            ).write(
                message.content
            )

    # ---------------- TEXT INPUT ----------------
    st.text_input(
        "Type your message here...",
        key="user_input",
        on_change=set_send_input
    )

    col1, col2 = st.columns(2)

    # ---------------- MIC ----------------
    with col1:

        voice_recording = mic_recorder(
            start_prompt="🎤 Start recording",
            stop_prompt="⏹ Stop recording",
            just_once=True
        )

    # ---------------- SEND BUTTON ----------------
    with col2:

        send_button = st.button(
            "Send",
            on_click=clear_input_field
        )

    # ---------------- AUDIO UPLOAD ----------------
    uploaded_audio = st.sidebar.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "ogg"]
    )

    # ---------------- PDF UPLOAD ----------------
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload PDF Files",
        accept_multiple_files=True,
        key="pdf_upload",
        type=["pdf"]
    )

    # ---------------- PDF PROCESSING ----------------
    if (
        uploaded_pdf
        and not st.session_state.pdf_processed
    ):

        with st.spinner("Processing PDF..."):

            try:

                add_documents_to_db(uploaded_pdf)

                st.session_state.pdf_processed = True

                st.sidebar.success(
                    "PDF Added Successfully"
                )

            except Exception as e:

                st.sidebar.error(str(e))

    # ---------------- AUDIO FILE ----------------
    if uploaded_audio:

        try:

            with st.spinner(
                "Transcribing Audio..."
            ):

                transcribed_audio = (
                    transcribe_audio(
                        uploaded_audio.getvalue()
                    )
                )

            with chat_container:

                st.chat_message(
                    "user"
                ).write(
                    transcribed_audio
                )

            with st.spinner(
                "Generating Response..."
            ):

                response = llm_chain.run(
                    "Summarize this text:\n"
                    + transcribed_audio
                )

            with chat_container:

                st.chat_message(
                    "assistant"
                ).write(
                    response
                )

            save_chat_history()

        except Exception as e:

            st.error(str(e))

    # ---------------- MIC RECORDING ----------------
    if voice_recording:

        try:

            with st.spinner(
                "Transcribing Voice..."
            ):

                transcribed_text = (
                    transcribe_audio(
                        voice_recording["bytes"]
                    )
                )

            with chat_container:

                st.chat_message(
                    "user"
                ).write(
                    transcribed_text
                )

            with st.spinner("Thinking..."):

                response = llm_chain.run(
                    transcribed_text
                )

            with chat_container:

                st.chat_message(
                    "assistant"
                ).write(
                    response
                )

            save_chat_history()

        except Exception as e:

            st.error(str(e))

    # ---------------- TEXT CHAT ----------------
    if send_button or st.session_state.send_input:

        question = (
            st.session_state.user_question.strip()
        )

        if question != "":

            st.session_state.send_input = False

            # ---------------- AUTO SESSION TITLE ----------------
            if (
                st.session_state.session_key
                == "new_session"
                and st.session_state.new_session_key
                is None
            ):

                session_title = (
                    generate_session_name(
                        question
                    )
                )

                st.session_state.new_session_key = (
                    session_title + ".json"
                )

            with chat_container:

                st.chat_message(
                    "user"
                ).write(
                    question
                )

            with st.spinner("Thinking..."):

                response = llm_chain.run(
                    question
                )

            with chat_container:

                st.chat_message(
                    "assistant"
                ).write(
                    response
                )

            save_chat_history()

            if (
                st.session_state.session_key
                == "new_session"
            ):

                st.session_state.session_index_tracker = (
                    st.session_state.new_session_key
                )

            st.session_state.user_question = ""

# ---------------- RUN ----------------
if __name__ == "__main__":

    main()
