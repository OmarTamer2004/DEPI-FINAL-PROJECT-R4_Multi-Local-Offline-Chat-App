# 💬 Multi Local Offline Chat App

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-AI%20App-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)
![Offline](https://img.shields.io/badge/100%25-Offline-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-orange?style=for-the-badge)

# 🔒 Privacy-First Offline AI Workspace

*A modern fully offline AI platform powered entirely by local LLMs — no internet, no APIs, and no cloud dependency.*

</div>

---

# ✨ Overview

**Multi Local Offline Chat App** is a fully offline AI-powered assistant developed for the **DEPI Final Project (Round 4)**.

The platform combines multiple AI capabilities into a single local workspace:

- 🤖 Local AI Chat
- 🧠 Multi-Model Support
- ⚡ Real-Time Streaming Responses
- 🎙️ Voice-to-Text Interaction
- 🔊 Audio File Summarization
- 📄 PDF Question Answering (RAG)
- 💾 Persistent Chat History
- 📤 Chat Export System
- 🖥️ Real-Time Hardware Monitoring
- 🔒 Complete Offline Privacy

All processing happens directly on your machine using locally hosted models.

---

# 🎯 Key Features

<div align="center">

| Feature | Description |
|---|---|
| 🤖 **Local LLM Chat** | Chat with locally hosted AI models without internet access |
| 🧠 **Multi-Model Support** | Dynamically switch between multiple GGUF local models |
| ⚡ **Streaming Responses** | Real-time token streaming with typing animation |
| 🎙️ **Voice Input** | Record audio directly from the browser microphone |
| 🔊 **Audio Upload** | Upload `.wav`, `.mp3`, or `.ogg` files for transcription & summarization |
| 📄 **PDF Chat (RAG)** | Ask questions about uploaded PDF documents |
| 💾 **Persistent Chat History** | Automatically save and revisit conversations |
| ✏️ **Session Management** | Rename, delete, and organize chat sessions |
| 📤 **Chat Export** | Export conversations as TXT or JSON |
| 🧩 **Memory Window Control** | Dynamically control conversation memory size |
| 🗒️ **Custom System Prompt** | Customize assistant behavior using editable prompts |
| 🛑 **Stop Generation** | Stop AI streaming responses instantly |
| 📋 **Copy Responses** | One-click copy for assistant replies |
| 🖥️ **Hardware Monitor** | Real-time CPU, RAM, GPU, and VRAM monitoring |
| 🏷️ **Auto Session Naming** | AI-generated titles for chat sessions |
| 🔒 **Fully Offline** | No APIs, cloud services, or external data sharing |

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
    ├── LangChain Pipeline
    └── Streaming Response Engine
            │
            ▼
      AI Generated Response
```

---

# 🎨 Modern UI Features

- 🌙 Fully customized dark theme
- ⚡ Streaming typing animation
- 📋 One-click copy buttons
- 💬 Styled AI chat bubbles
- 🧠 Sidebar hardware monitor
- 📱 Clean responsive layout
- 🛑 Interactive generation controls
- 🎛️ Dynamic settings sidebar

---

# 🧠 Supported Models

The application supports local GGUF/GGML models including:

- LLaMA 2
- Mistral
- TinyLlama
- Phi
- Gemma
- DeepSeek
- Any CTransformers-compatible GGUF model

Simply place `.gguf` files inside the `models/` directory.

---

# 🗂️ Project Structure

```bash
.
├── app.py                  # Main Streamlit application & UI logic
├── llm_chains.py           # LangChain chain definitions & model loading
├── audio_handler.py        # Audio transcription utilities
├── pdf_handler.py          # PDF parsing & ChromaDB ingestion
├── image_handler.py        # Image utilities
├── prompt_templates.py     # Prompt engineering templates
├── utils.py                # Helper & persistence utilities
├── config.yaml             # Application configuration
├── models/                 # Local GGUF models
├── chat_history/           # Stored conversation history
├── requirements.txt        # Project dependencies
└── test.py                 # Testing utilities
```

---

# 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| 🎨 Frontend UI | Streamlit |
| 🧠 LLM Inference | CTransformers |
| 🔗 AI Orchestration | LangChain |
| 📚 Vector Database | ChromaDB |
| 🧬 Embeddings | InstructorEmbedding |
| 🎵 Audio Processing | librosa + Whisper |
| 📄 PDF Parsing | pypdfium2 |
| 🔥 Deep Learning | PyTorch + Transformers |
| 🖥️ System Monitoring | psutil + pynvml |

</div>

---

# ⚙️ Installation Guide

## 📌 Prerequisites

Before starting, make sure you have:

- Python **3.9 – 3.11**
- pip installed
- A GGUF / GGML compatible local model

Example supported models:

- Mistral
- LLaMA
- TinyLlama
- Phi
- Gemma

---

# 1️⃣ Clone Repository

```bash
git clone https://github.com/OmarTamer2004/DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App.git
cd DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App
```

---

# 2️⃣ Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

# 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ PyTorch Note:
>
> The provided requirements use CPU mode by default.
> For GPU acceleration, install the correct CUDA version from the PyTorch official website.

---

# 4️⃣ Add Local Models

Create a `models/` folder and place your `.gguf` models inside it.

Example:

```text
models/
 ├── mistral-7b-instruct.gguf
 ├── phi-2.gguf
 └── tinyllama.gguf
```

---

# 5️⃣ Configure Application

Edit `config.yaml`:

```yaml
chat_history_path: "chat_history"
models_dir: "models"

available_models:
  - "Mistral"
  - "Phi"
  - "TinyLlama"

system_prompt: "You are a helpful AI assistant."

memory_window: 10
```

---

# 6️⃣ Run the Application

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🚀 Usage Guide

# 💬 Text Chat

Type your message and interact with the local AI model in real time.

---

# 🎙️ Voice Input

1. Click **Start Recording**
2. Speak into the microphone
3. Stop recording
4. The speech is automatically transcribed and sent to the model

---

# 🔊 Audio File Upload

Supported formats:

- `.wav`
- `.mp3`
- `.ogg`

The application will:

- Transcribe the audio
- Generate an AI summary

---

# 📄 PDF Chat (RAG)

1. Enable **PDF Chat Mode**
2. Upload one or more PDF files
3. Ask questions about the document content

The system retrieves relevant chunks using ChromaDB and generates responses using the local LLM.

---

# 💾 Chat History

All conversations are stored locally as JSON files.

Users can:

- Reopen old chats
- Continue conversations
- Maintain persistent memory between sessions

---

# ✏️ Session Management

The application supports:

- Automatic AI-generated session names
- Rename chat sessions
- Delete sessions
- Organized session history

---

# 📤 Export Conversations

Export chats in multiple formats:

- TXT
- JSON

Useful for:

- Dataset creation
- Research
- Backup
- Conversation analysis

---

# 🧠 Advanced Features

# 🖥️ Hardware Monitoring

The application includes a real-time system monitoring panel showing:

- CPU usage
- RAM utilization
- GPU utilization
- VRAM consumption

GPU monitoring uses NVIDIA NVML with graceful fallback if no GPU is available.

---

# 🧩 Dynamic Memory Window

Users can control how many previous messages remain inside the LLM context window.

Benefits:

- Reduced RAM usage
- Faster inference
- Better context management
- Improved performance on low-end devices

---

# 🗒️ Custom System Prompts

The application supports editable system prompts.

Users can:

- Modify assistant personality
- Save prompts directly into `config.yaml`
- Reset prompts to default

This enables prompt engineering experimentation fully offline.

---

# 🛑 Interruptible Generation

Streaming responses can be stopped in real time using the Stop button.

This improves:

- UX responsiveness
- Long generation control
- Resource efficiency

---

# 📋 Copy Assistant Responses

Every assistant response includes a one-click copy button for improved usability.

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
psutil
pynvml
python==3.10.11
```

---

# 🔒 Privacy & Security

<div align="center">

# Your Data Never Leaves Your Device

</div>

This application is completely offline.

- ❌ No API keys
- ❌ No cloud processing
- ❌ No third-party servers
- ❌ No internet dependency

All conversations, audio recordings, and uploaded documents remain entirely on your local machine.

---

# 🌟 Future Improvements

- 🌐 Multi-language support
- 🧠 Better memory management
- 🖼️ Image understanding support
- ⚡ GPU optimization
- 📱 Mobile responsive UI
- 🔍 Semantic search improvements
- 🧩 Plugin system
- 📊 Usage analytics dashboard
- 🤝 Multi-agent support

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

# Omar Tamer Elazab Melegy

DEPI Final Project — Round 4

---

# 📄 License

This project was developed for educational purposes as part of the **Digital Egypt Pioneers Initiative (DEPI)**.

You are free to use and modify it for learning and educational projects.

---

<div align="center">

# ⭐ If you like this project, give it a star on GitHub ⭐

</div>
