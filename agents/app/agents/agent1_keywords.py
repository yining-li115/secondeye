from pathlib import Path
from typing import Dict

from app.audio.stt import transcribe_audio_to_text
from app.nlp.keyword_extractor import extract_keywords_and_prompt


def process_audio_for_vlm_prompt(audio_file_path: Path) -> Dict:
    """
    Agent 1 main workflow:
    1. Audio -> Text
    2. Text -> Keywords + VLM prompt
    3. Return structured result
    """
    # 1. Speech to text
    transcript = transcribe_audio_to_text(audio_file_path)

    # 2. Extract keywords + generate prompt
    result = extract_keywords_and_prompt(transcript)

    # 3. Return unified format
    return {
        "transcript": transcript,
        "keywords": result["keywords"],
        "vlm_prompt": result["vlm_prompt"],
    }
