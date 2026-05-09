# 💬 Multi Local Offline Chat App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-AI%20App-red?style=for-the-badge\&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)
![Offline](https://img.shields.io/badge/100%25-Offline-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-orange?style=for-the-badge)

## 🔒 Privacy-First AI Assistant Running Fully Offline

*A modern local AI chatbot powered entirely by local LLMs — no internet, no APIs, no cloud dependency.*

</div>

---

# ✨ Overview

**Multi Local Offline Chat App** is a fully offline AI-powered assistant built for the **DEPI Final Project (Round 4)**.

The application combines:

* 🤖 Local AI Chat
* 🎙️ Voice-to-Text Interaction
* 📄 PDF Question Answering (RAG)
* 💾 Persistent Chat History
* 🔊 Audio File Summarization
* 🔒 Complete Offline Privacy

All processing happens directly on your machine using locally hosted models.

---

# 🎯 Key Features

<div align="center">

| Feature                        | Description                                                              |
| ------------------------------ | ------------------------------------------------------------------------ |
| 🤖 **Local LLM Chat**          | Chat with locally hosted AI models without internet access               |
| 🎙️ **Voice Input**            | Record audio directly from the browser microphone                        |
| 🔊 **Audio Upload**            | Upload `.wav`, `.mp3`, or `.ogg` files for transcription & summarization |
| 📄 **PDF Chat (RAG)**          | Ask questions about uploaded PDF documents                               |
| 💾 **Persistent Chat History** | Automatically save and revisit conversations                             |
| 🏷️ **Auto Session Naming**    | AI-generated titles for chat sessions                                    |
| 🔒 **Fully Offline**           | No APIs, cloud services, or external data sharing                        |

</div>

---

# 🖼️ Application Workflow

```text
User Input
   │
   ├── Text Message
   ├── Voice Recording
   ├── Audio File
   └── PDF Document
            │
            ▼
    Local Processing Engine
            │
    ├── Local LLM
    ├── Whisper / Audio Processing
    ├── ChromaDB Vector Store
    └── LangChain Pipeline
            │
            ▼
      AI Generated Response
```

---

# 🗂️ Project Structure

```bash
.
├── app.py                  # Main Streamlit application & UI logic
├── llm_chains.py           # LangChain chain definitions
├── audio_handler.py        # Audio transcription logic
├── pdf_handler.py          # PDF processing & vector storage
├── image_handler.py        # Image utilities
├── prompt_templates.py     # Prompt engineering templates
├── utils.py                # Helper utilities
├── config.yaml             # Application configuration
├── requirements.txt        # Project dependencies
└── test.py                 # Testing scripts
```

---

# 🛠️ Tech Stack

<div align="center">

| Layer               | Technology             |
| ------------------- | ---------------------- |
| 🎨 Frontend UI      | Streamlit              |
| 🧠 LLM Inference    | CTransformers          |
| 🔗 AI Orchestration | LangChain              |
| 📚 Vector Database  | ChromaDB               |
| 🧬 Embeddings       | InstructorEmbedding    |
| 🎵 Audio Processing | librosa                |
| 📄 PDF Parsing      | pypdfium2              |
| 🔥 Deep Learning    | PyTorch + Transformers |

</div>

---

# ⚙️ Installation Guide

## 📌 Prerequisites

Before starting, make sure you have:

* Python **3.9 – 3.11**
* pip installed
* A GGUF / GGML compatible local model

  * Example: LLaMA 2, Mistral, Phi, TinyLlama

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/OmarTamer2004/DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App.git
cd DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch Note:**
>
> The provided requirements use CPU mode by default.
> For GPU acceleration, install the correct CUDA version from PyTorch official website.

---

## 4️⃣ Configure Application

Edit `config.yaml`:

```yaml
chat_history_path: "chat_history"
model_path: "models/your-model.gguf"
model_type: "llama"
```

---

## 5️⃣ Run the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🚀 Usage Guide

## 💬 Text Chat

Type your message and interact with the local AI model in real time.

---

## 🎙️ Voice Input

1. Click **Start Recording**
2. Speak into the microphone
3. Stop recording
4. The speech is automatically transcribed and sent to the model

---

## 🔊 Audio File Upload

Upload:

* `.wav`
* `.mp3`
* `.ogg`

The application will:

* Transcribe the audio
* Generate an AI summary

---

## 📄 PDF Chat (RAG)

1. Enable **PDF Chat** mode
2. Upload one or more PDF files
3. Ask questions about document content

The system retrieves relevant chunks using ChromaDB and answers using the local LLM.

---

## 💾 Chat History

All conversations are stored locally as JSON files.

Users can:

* Reopen old chats
* Continue conversations
* Maintain persistent memory between sessions

---

# 📦 Main Dependencies

```txt
langchain==0.0.354
langchain-community==0.0.20
chromadb==0.4.22
ctransformers==0.2.27
InstructorEmbedding==1.0.1
sentence-transformers==2.2.2
torch==2.1.2
transformers==4.35.2
huggingface-hub==0.20.3
pypdfium2==4.24.0
librosa==0.10.1
PyYAML==6.0.1
streamlit==1.29.0
streamlit-mic-recorder==0.0.4
python==3.10.11
```

---

# 🔒 Privacy & Security

<div align="center">

## Your Data Never Leaves Your Device

</div>

This application is completely offline.

* ❌ No API keys
* ❌ No cloud processing
* ❌ No third-party servers
* ❌ No internet dependency

All conversations, audio recordings, and uploaded documents remain on your local machine.

---

# 🌟 Future Improvements

* 🌐 Multi-language support
* 🧠 Better memory management
* 🖼️ Image understanding support
* ⚡ GPU optimization
* 📱 Mobile responsive UI
* 🔍 Semantic search improvements

---

# 🤝 Contributing

Contributions are welcome.

If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

# 👨‍💻 Developed By

## Omar Tamer Elazab Melegy

DEPI Final Project — Round 4

---

# 📄 License

This project was developed for educational purposes as part of the **Digital Egypt Pioneers Initiative (DEPI)**.

You are free to use and modify it for learning and educational projects.

---

<div align="center">

# ⭐ If you like this project, give it a star on GitHub ⭐

</div>
