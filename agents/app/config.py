import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Chat model name (supports multimodal capabilities)
    llm_model: str = "gpt-4o-mini"       # Text-only model for intent recognition
    vlm_model: str = "gpt-4o-mini"       # Vision-language model (gpt-4o-mini supports vision)

    # Backend service URL (e.g., your vision service)
    vision_backend_url: str = os.getenv("VISION_BACKEND_URL", "")

    # Unified gentle voice style description (for LLM system prompts)
    gentle_voice_style: str = (
        "You are a voice assistant specifically designed to help visually impaired and mobility-impaired users. "
        "Your tone should be gentle, patient, and your speech pace should be slightly slower. "
        "Keep sentences short and explain things step by step."
    )


settings = Settings()
