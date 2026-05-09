💬 Multi Local Offline Chat App

A fully offline, privacy-first AI chat application powered by local LLMs — no internet connection or API keys required.

Built as the DEPI Final Project (Round 4), this app combines text, voice, and document interaction in a clean Streamlit interface, all running entirely on your own machine.

✨ Features
FeatureDescription🤖 Local LLM ChatConverse with a locally hosted language model — zero cloud dependency🎙️ Voice InputRecord your message via microphone directly in the browser🔊 Audio File UploadUpload .wav, .mp3, or .ogg files and get automatic transcription + summary📄 PDF Chat (RAG)Upload PDFs and ask questions about their content using Retrieval-Augmented Generation💾 Persistent Chat HistorySessions are saved as JSON files and can be revisited at any time🏷️ Auto Session NamingThe LLM automatically generates a short title for each new chat session🔒 Fully OfflineAll models and embeddings run locally — your data never leaves your machine

🗂️ Project Structure
.
├── app.py                  # Main Streamlit application & UI logic
├── llm_chains.py           # LangChain chain definitions (normal & PDF chat)
├── audio_handler.py        # Audio transcription logic
├── pdf_handler.py          # PDF ingestion & ChromaDB vector storage
├── image_handler.py        # Image processing utilities
├── prompt_templates.py     # LangChain prompt templates
├── utils.py                # Helper functions (JSON save/load, timestamps)
├── config.yaml             # App configuration (paths, model settings)
├── requirements.txt        # Python dependencies
└── test.py                 # Unit / integration tests

🛠️ Tech Stack
LayerLibraryUIStreamlitLLM InferenceCTransformers (GGML local models)LLM OrchestrationLangChainVector StoreChromaDBEmbeddingsInstructorEmbedding + sentence-transformersAudiolibrosa + streamlit-mic-recorderPDF Parsingpypdfium2Deep LearningPyTorch + Transformers

⚙️ Installation
Prerequisites

Python 3.9 – 3.11
pip
A GGML-compatible local model file (e.g. LLaMA 2, Mistral, etc.)

1. Clone the repository
bashgit clone https://github.com/OmarTamer2004/DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App.git
cd DEPI-FINAL-PROJECT-R4_Multi-Local-Offline-Chat-App
2. Create a virtual environment (recommended)
bashpython -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt

⚠️ PyTorch note: The version in requirements.txt targets CPU. For GPU acceleration, visit pytorch.org and install the appropriate CUDA build first.

4. Configure the app
Edit config.yaml to point to your local model file and set the chat history directory:
yaml# config.yaml (example)
chat_history_path: "chat_history"
model_path: "models/your-model.gguf"   # path to your GGML model
model_type: "llama"                     # ctransformers model type
5. Run the app
bashstreamlit run app.py
Open your browser at http://localhost:8501.

🚀 Usage
Text Chat
Type a message in the input field and press Send (or hit Enter). The LLM responds locally in real time.
Voice Input
Click 🎤 Start recording, speak your message, then click ⏹ Stop recording. The audio is transcribed and sent to the model automatically.
Audio File Upload
Use the Upload Audio File sidebar panel to upload a pre-recorded audio file. The app transcribes it and returns an AI-generated summary.
PDF Chat

Toggle pdf chat in the sidebar.
Upload one or more PDF files via Upload PDF Files.
Ask questions — the app retrieves relevant chunks from the document and answers using the LLM.

Chat History
All sessions are persisted locally as JSON files. Use the Select a chat session dropdown in the sidebar to switch between past conversations.

📦 Dependencies
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
python == 3.10.11

🔐 Privacy
This application is 100% local and offline. No data — including your messages, uploaded files, or voice recordings — is ever sent to an external server or third-party API.

🤝 Contributing
Contributions are welcome! To get started:




📄 License
This project was developed as part of the DEPI (Digital Egypt Pioneers Initiative) program. Feel free to use and adapt it for educational purposes.
