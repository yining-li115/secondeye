from pathlib import Path
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def transcribe_audio_to_text(audio_path: Path) -> str:
    """
    Convert audio file to text using OpenAI Whisper API.
    Supports multiple audio formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.
    Automatically detects language (supports 99+ languages including English, Chinese, German, etc.)
    """
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
            # Omit language parameter for automatic language detection
        )
    return transcript.text
