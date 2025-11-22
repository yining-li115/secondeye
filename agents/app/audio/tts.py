from pathlib import Path
from openai import OpenAI
from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def text_to_speech(text: str, output_path: Path, voice: str = "nova") -> Path:
    """
    Convert text to speech using OpenAI TTS API.

    Args:
        text: The text to convert to speech
        output_path: Path where the audio file will be saved
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
               'nova' is recommended for a gentle, patient tone

    Returns:
        Path to the generated audio file
    """
    # Validate input text
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for TTS conversion")

    response = client.audio.speech.create(
        model="tts-1",  # Use "tts-1-hd" for higher quality
        voice=voice,
        input=text.strip(),  # Remove leading/trailing whitespace
        speed=0.9  # Slightly slower for better comprehension
    )

    response.stream_to_file(output_path)
    return output_path


def text_to_speech_gentle(text: str, output_path: Path) -> Path:
    """
    Convert text to speech with a gentle, patient voice suitable for
    visually impaired and mobility-impaired users.

    Uses 'nova' voice with slightly slower speed for better comprehension.
    """
    return text_to_speech(text, output_path, voice="nova")
