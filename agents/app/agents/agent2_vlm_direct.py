from pathlib import Path
from typing import Dict, List, Optional
import base64
import json
from io import BytesIO
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from app.audio.stt import transcribe_audio_to_text
from app.audio.tts import text_to_speech_gentle
from app.config import settings


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode image file to base64 string for OpenAI Vision API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_pil_image_to_base64(image: Image.Image) -> str:
    """
    Encode PIL Image to base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_keywords_and_intent(transcript: str) -> Dict:
    """
    Extract keywords and determine user intent from transcribed audio.
    Internal function integrated from Agent 1.

    Args:
        transcript: User's speech converted to text

    Returns:
        Dict with:
        - keywords: List of extracted keywords
        - intent: User intent (describe/find/navigate/general)
        - target_object: Main object of interest (if any)
        - enhanced_query: Optimized query for VLM
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    system_prompt = (
        "You are an assistant specialized in understanding user requests for visual assistance. "
        "Analyze the user's speech and extract:\n"
        "1. Key keywords (1-5 important words)\n"
        "2. User intent: 'describe' (scene description), 'find' (locate object), or 'general'\n"
        "3. Target object (if user wants to find something)\n"
        "4. An enhanced query that can be sent to a vision model\n\n"
        "CRITICAL LANGUAGE RULE:\n"
        "- If input is in English, ALL output fields MUST be in English\n"
        "- If input is in Chinese, ALL output fields MUST be in Chinese\n"
        "- If input is in German, ALL output fields MUST be in German\n"
        "- NEVER translate the input to a different language\n"
        "- The enhanced_query MUST be in the SAME language as the input\n\n"
        "Output MUST be valid JSON with these fields:\n"
        "{{\"keywords\": [...], \"intent\": \"...\", \"target_object\": \"...\", \"enhanced_query\": \"...\"}}\n\n"
        f"Context: {settings.gentle_voice_style}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "User speech: {transcript}")
    ])

    chain = prompt | llm
    response = chain.invoke({"transcript": transcript})

    try:
        data = json.loads(response.content)
    except Exception:
        # Fallback if JSON parsing fails
        data = {
            "keywords": [],
            "intent": "general",
            "target_object": "",
            "enhanced_query": transcript
        }

    # Ensure all fields exist
    data.setdefault("keywords", [])
    data.setdefault("intent", "general")
    data.setdefault("target_object", "")
    data.setdefault("enhanced_query", transcript)

    return data


def analyze_scene_with_vlm(
    image_path: Path,
    user_query: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 150  # Limit response length for conciseness
) -> str:
    """
    Use Vision-Language Model (VLM) to analyze an image based on user query.

    Args:
        image_path: Path to the image file
        user_query: User's question about the image
        system_prompt: Optional system prompt to guide the model's behavior
        max_tokens: Maximum tokens in response (default: 150, about 2-3 sentences)

    Returns:
        VLM's text response
    """
    llm = ChatOpenAI(
        model=settings.vlm_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
        max_tokens=max_tokens,  # Limit response length
    )

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Build messages
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    else:
        messages.append(SystemMessage(content=settings.gentle_voice_style))

    # Add user message with image
    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        )
    )

    # Get response
    response = llm.invoke(messages)
    return response.content


def search_object_in_frames(
    frame_paths: List[Path],
    target_object: str,
    duration_seconds: int = 5
) -> Dict:
    """
    Search for a target object across multiple video frames.

    Args:
        frame_paths: List of paths to frame images
        target_object: Name of the object to find (e.g., "apple")
        duration_seconds: How long to search (for context)

    Returns:
        Dict with detection results and description
    """
    llm = ChatOpenAI(
        model=settings.vlm_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    system_prompt = (
        f"You are a vision assistant helping to find a '{target_object}' in the scene. "
        f"Carefully examine the image and determine if the object is present.\n\n"
        f"IMPORTANT: Respond with valid JSON only:\n"
        f"{{\n"
        f'  "found": true/false,\n'
        f'  "description": "your description here"\n'
        f"}}\n\n"
        f"If found=true: Describe the location RELATIVE TO THE CAMERA (from the viewer's perspective):\n"
        f"  - Use directions: 'to your left', 'to your right', 'in front of you', 'above you', 'below you'\n"
        f"  - Add distance if clear: 'close by', 'far away', 'within reach'\n"
        f"  - Example: 'To your right, close by' or 'In front of you, on the upper part of the view'\n"
        f"  - DO NOT reference other objects (no 'on the table', 'next to the phone', etc.)\n"
        f"If found=false: explain you don't see it and suggest moving the camera.\n\n"
        f"CRITICAL: The 'description' field MUST be in the SAME language as the input."
    )

    found = False
    location_description = ""
    best_frame_idx = -1

    # Check each frame
    for idx, frame_path in enumerate(frame_paths):
        base64_image = encode_image_to_base64(frame_path)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Can you see a {target_object} in this image? If yes, where is it located?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON response from VLM
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            if result.get("found", False):
                found = True
                location_description = result.get("description", "Object found.")
                best_frame_idx = idx
                break
        except json.JSONDecodeError:
            # Fallback: if VLM doesn't return JSON, use simple heuristic
            # Check for clear negative indicators
            response_lower = response_text.lower()
            is_negative = any(neg in response_lower for neg in [
                "don't see", "do not see", "not see", "cannot see", "can't see",
                "no " + target_object.lower(), "not visible"
            ])

            # If response contains target object and no negative, assume found
            if target_object.lower() in response_lower and not is_negative:
                found = True
                location_description = response_text
                best_frame_idx = idx
                break

    if found:
        return {
            "found": True,
            "frame_index": best_frame_idx,
            "description": location_description,
            "voice_response": location_description
        }
    else:
        response_text = f"I've been looking for {duration_seconds} seconds, but I don't see a {target_object} yet. Try moving your camera around slowly."
        return {
            "found": False,
            "frame_index": -1,
            "description": response_text,
            "voice_response": response_text
        }


def process_audio_and_video(
    audio_path: Path,
    image_path: Path,
    output_audio_path: Path
) -> Dict:
    """
    Agent 2 main workflow:
    1. Audio (user query) -> Text via STT
    2. Extract keywords and understand intent (integrated from Agent 1)
    3. Analyze current video frame with VLM based on intent
    4. Generate voice response via TTS

    Args:
        audio_path: Path to user's audio question
        image_path: Path to current video frame
        output_audio_path: Path where TTS output will be saved

    Returns:
        Dict with transcript, keywords, intent, VLM response, and audio output path
    """
    # 1. Speech to text
    user_query = transcribe_audio_to_text(audio_path)

    # 2. Extract keywords and understand intent (Agent 1 functionality integrated)
    intent_data = extract_keywords_and_intent(user_query)
    intent = intent_data["intent"]
    target_object = intent_data["target_object"]
    enhanced_query = intent_data["enhanced_query"]
    keywords = intent_data["keywords"]

    # 3. Use VLM based on intent
    if intent == "describe":
        # Scene description task
        system_prompt = (
            f"{settings.gentle_voice_style}\n\n"
            "Describe what you see in 2-3 short sentences. "
            "Focus ONLY on the most important objects. "
            "Be concise and clear.\n\n"
            "IMPORTANT: Respond in the SAME language as the user's query."
        )
        vlm_response = analyze_scene_with_vlm(image_path, enhanced_query, system_prompt, max_tokens=100)

    elif intent == "find" and target_object:
        # Object finding task
        system_prompt = (
            f"{settings.gentle_voice_style}\n\n"
            f"Look for '{target_object}' in the scene. "
            f"If found, describe its location in 1-2 sentences (left/right/center). "
            f"Be brief and clear.\n\n"
            f"IMPORTANT: Respond in the SAME language as the user's query."
        )
        vlm_response = analyze_scene_with_vlm(image_path, enhanced_query, system_prompt, max_tokens=80)

    else:
        # General query
        system_prompt = (
            f"{settings.gentle_voice_style}\n\n"
            "Answer briefly in 1-2 sentences.\n\n"
            "IMPORTANT: Respond in the SAME language as the user's query."
        )
        vlm_response = analyze_scene_with_vlm(image_path, enhanced_query, system_prompt, max_tokens=100)

    # 4. Convert response to speech
    text_to_speech_gentle(vlm_response, output_audio_path)

    return {
        "transcript": user_query,
        "keywords": keywords,
        "intent": intent,
        "target_object": target_object,
        "vlm_response": vlm_response,
        "audio_output": str(output_audio_path)
    }
