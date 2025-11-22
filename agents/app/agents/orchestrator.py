"""
Main orchestrator for SecondEye agents.
Coordinates the workflow between STT, intent recognition, VLM, navigation, and TTS.
"""

from pathlib import Path
from typing import Dict, List, Optional
import httpx
import time
from datetime import datetime

from app.audio.stt import transcribe_audio_to_text
from app.audio.tts import text_to_speech_gentle
from app.agents.agent2_vlm_direct import (
    extract_keywords_and_intent,
    analyze_scene_with_vlm,
    search_object_in_frames
)
from app.agents.agent3_navigation import (
    navigate_to_target,
    Position3D,
    Orientation
)
from app.config import settings


def log_timestamp(step_name: str, start_time: float = None) -> float:
    """
    Log timestamp for a step. Returns current time for timing purposes.

    Args:
        step_name: Name of the step to log
        start_time: Start time of the step (optional, for duration calculation)

    Returns:
        Current timestamp
    """
    current_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    if start_time is not None:
        duration = current_time - start_time
        print(f"[{timestamp}] ✓ {step_name} (took {duration:.2f}s)")
    else:
        print(f"[{timestamp}] ▶ {step_name}")

    return current_time


async def call_3d_reconstruction_service(target_object: str, frame_path: Path) -> Optional[Dict]:
    """
    Call external 3D reconstruction service to get object and camera positions.

    Args:
        target_object: Name of the object to locate
        frame_path: Path to the frame where object was found

    Returns:
        Dict with:
        - target_position: {x, y, z}
        - camera_position: {x, y, z}
        - camera_orientation: {yaw, pitch}
        Or None if service unavailable
    """
    if not settings.vision_backend_url:
        # Fallback: return mock positions for testing
        return {
            "target_position": {"x": 2.0, "y": 0.0, "z": 3.0},
            "camera_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "camera_orientation": {"yaw": 0.0, "pitch": 0.0}
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Send frame to 3D reconstruction service
            with open(frame_path, "rb") as f:
                files = {"image": f}  # 3D service expects "file" parameter
                data = {"target_object": target_object}

                response = await client.post(
                    f"{settings.vision_backend_url}/locate",
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"3D service returned status {response.status_code}: {response.text}")
                    return None
    except Exception as e:
        print(f"3D reconstruction service error: {e}")
        return None


async def process_user_request(
    audio_path: Path,
    video_frames: List[Path],
    output_audio_path: Path,
    max_search_duration: int = 5
) -> Dict:
    """
    Main orchestrator: processes user audio and video to provide appropriate response.

    Workflow:
    1. STT: Convert audio to text
    2. Intent recognition: Understand what user wants
    3. Execute based on intent:
       - describe: Scene description
       - find: Search object → Get 3D position → Navigate
       - general: Answer question
    4. TTS: Convert response to speech

    Args:
        audio_path: Path to user's audio file
        video_frames: List of video frame paths (for search operations)
        output_audio_path: Where to save TTS output
        max_search_duration: Max seconds to search for object

    Returns:
        Dict with:
        - intent: User intent
        - transcript: What user said
        - action_taken: What the system did
        - response_text: Text response
        - audio_output: Path to audio response
        - additional_data: Any extra info (positions, etc.)
    """
    workflow_start = log_timestamp("Starting workflow")

    # Step 1: Speech to text
    step_start = log_timestamp("Step 1: Speech to Text (STT)")
    transcript = transcribe_audio_to_text(audio_path)
    log_timestamp(f"STT complete - Transcript: '{transcript[:50]}...'", step_start)

    # Step 2: Extract intent and keywords
    step_start = log_timestamp("Step 2: Intent Recognition")
    intent_data = extract_keywords_and_intent(transcript)
    intent = intent_data["intent"]
    target_object = intent_data["target_object"]
    enhanced_query = intent_data["enhanced_query"]
    print(f"DEBUG - Original transcript: {transcript}")
    print(f"DEBUG - Enhanced query: {enhanced_query}")
    log_timestamp(f"Intent: {intent}, Target: {target_object}", step_start)

    response_text = ""
    action_taken = ""
    additional_data = {}
    # Step 3: Execute based on intent
    step_start = log_timestamp(f"Step 3: Executing action for intent '{intent}'")

    if intent == "describe":
        # Scene description
        action_taken = "scene_description"

        # Use the first/latest frame for description
        current_frame = video_frames[0] if video_frames else None

        if current_frame:
            vlm_start = log_timestamp("VLM: Analyzing scene")
            system_prompt = (
                "Describe the scene in 1-2 SHORT sentences only. "
                "Focus on the main objects and their general locations. "
                "Be direct and concise. No detailed descriptions.\n\n"
                "CRITICAL: Respond in the EXACT SAME language as the user's input. "
                "DO NOT translate or change the language.\n\n"
                "Example: 'I see a desk with a laptop and a coffee cup. There is a window on the left.'"
            )
            response_text = analyze_scene_with_vlm(
                current_frame,
                enhanced_query,
                system_prompt,
                max_tokens=60
            )
            log_timestamp("VLM: Scene analysis complete", vlm_start)
        else:
            response_text = "I need a video frame to describe the scene."

    elif intent == "find" and target_object:
        # Object finding workflow
        action_taken = "object_search"

        # Search for object in video frames
        search_start = log_timestamp(f"VLM: Searching for '{target_object}' in {len(video_frames)} frame(s)")
        search_result = search_object_in_frames(
            video_frames,
            target_object,
            max_search_duration
        )
        log_timestamp(f"Search complete - Found: {search_result['found']}", search_start)

        if search_result["found"]:
            # Object found! Now get 3D position
            action_taken = "object_found_navigation"
            found_frame = video_frames[search_result["frame_index"]]

            # Call 3D reconstruction service
            service_start = log_timestamp("Calling 3D reconstruction service")
            position_data = await call_3d_reconstruction_service(
                target_object,
                found_frame
            )
            
            log_timestamp("3D service response received", service_start)
            print(f"3D service returned data: {position_data}")

            # Validate 3D service response has required fields
            has_valid_format = (
                position_data and
                "target_position" in position_data and
                "camera_position" in position_data and
                "camera_orientation" in position_data
            )

            if has_valid_format:
                # Generate navigation instructions
                nav_start = log_timestamp("Generating navigation instructions")
                current_pos = Position3D(
                    position_data["camera_position"]["x"],
                    position_data["camera_position"]["y"],
                    position_data["camera_position"]["z"]
                )
                current_orientation = Orientation(
                    position_data["camera_orientation"]["yaw"],
                    position_data["camera_orientation"]["pitch"]
                )
                target_pos = Position3D(
                    position_data["target_position"]["x"],
                    position_data["target_position"]["y"],
                    position_data["target_position"]["z"]
                )

                # Create temporary navigation audio path
                nav_audio_path = output_audio_path.parent / "navigation_temp.mp3"

                nav_result = navigate_to_target(
                    target_object,
                    current_pos,
                    current_orientation,
                    target_pos,
                    nav_audio_path
                )
                log_timestamp("Navigation instructions generated", nav_start)

                response_text = nav_result["instructions"]
                additional_data = {
                    "object_found": True,
                    "navigation_metrics": nav_result["navigation_metrics"],
                    "positions": position_data
                }
            else:
                # 3D service unavailable, just confirm we found it
                response_text = search_result["description"]
                additional_data = {"object_found": True}
        else:
            # Object not found
            action_taken = "object_not_found"
            response_text = (
                f"I've been looking for the {target_object} for {max_search_duration} seconds, "
                f"but I haven't found it yet. Please move your camera around slowly so I can see more of the area."
            )
            additional_data = {"object_found": False}

    else:
        # General query
        action_taken = "general_query"

        current_frame = video_frames[0] if video_frames else None

        if current_frame:
            vlm_start = log_timestamp("VLM: Answering general query")
            system_prompt = (
                "Answer the user's question about the image in 1-2 SHORT sentences. "
                "Be direct and concise. No explanations or lists.\n\n"
                "CRITICAL: You MUST respond in the EXACT SAME language as the user's question. "
                "If the question is in English, respond in English. "
                "If the question is in Chinese, respond in Chinese. "
                "DO NOT translate or change the language."
            )
            response_text = analyze_scene_with_vlm(
                current_frame,
                enhanced_query,
                system_prompt,
                max_tokens=50
            )
            log_timestamp("VLM: General query complete", vlm_start)
        else:
            response_text = "I need a video frame to answer your question."

    log_timestamp(f"Action '{action_taken}' complete", step_start)
    
    print("response_text:",{response_text})

    # Step 4: Convert response to speech
    tts_start = log_timestamp("Step 4: Text to Speech (TTS)")
    text_to_speech_gentle(response_text, output_audio_path)
    log_timestamp("TTS complete", tts_start)

    log_timestamp(f"✓ Workflow complete", workflow_start)

    return {
        "intent": intent,
        "transcript": transcript,
        "target_object": target_object,
        "action_taken": action_taken,
        "response_text": response_text,
        "audio_output": str(output_audio_path),
        "additional_data": additional_data
    }
