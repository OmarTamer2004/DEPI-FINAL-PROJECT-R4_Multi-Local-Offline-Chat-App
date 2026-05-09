import torch
from transformers import pipeline
import librosa
import io

# ---------------- CONVERT AUDIO ----------------
def convert_bytes_to_array(audio_bytes):

    audio_bytes = io.BytesIO(audio_bytes)

    audio, sample_rate = librosa.load(
        audio_bytes,
        sr=16000
    )

    print(sample_rate)

    return audio

# ---------------- TRANSCRIBE AUDIO ----------------
def transcribe_audio(audio_bytes):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    audio_array = convert_bytes_to_array(audio_bytes)

    # speech to text
    prediction = pipe(
        audio_array,
        batch_size=1
    )["text"]

    return prediction