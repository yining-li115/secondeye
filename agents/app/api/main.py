from pathlib import Path
import tempfile
from typing import List
import base64

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.agents.orchestrator import process_user_request

app = FastAPI(
    title="SecondEye Agents API",
    description="Unified AI agent for visually impaired and mobility-impaired users",
    version="2.0.0"
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/process")
async def process_request(
    audio: UploadFile = File(...),
    frames: List[UploadFile] = File(...),
    max_search_duration: int = Form(5)
):
    """
    Main unified endpoint for processing user requests.

    Workflow:
    1. STT: Convert audio to text
    2. Understand intent (describe/find/general)
    3. Execute appropriate action:
       - describe: Scene description
       - find: Search object → Get 3D position from backend service → Navigate
       - general: Answer question
    4. Return audio response

    Receives:
    - audio: User's audio question
    - frames: Video frames (1+ frames, use multiple for search)
    - max_search_duration: Max seconds to search for object (default: 5)

    Returns:
    - JSON with intent, action taken, response text, and audio output
    """
    # Save audio
    audio_suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    tmp_audio_dir = tempfile.mkdtemp()
    tmp_audio_path = Path(tmp_audio_dir) / f"input{audio_suffix}"
    print("step1: save audio")
    with open(tmp_audio_path, "wb") as f:
        audio_content = await audio.read()
        f.write(audio_content)

    # Save all video frames
    frame_paths = []
    tmp_frames_dir = tempfile.mkdtemp()

    for idx, frame in enumerate(frames):
        suffix = Path(frame.filename).suffix if frame.filename else ".jpg"
        frame_path = Path(tmp_frames_dir) / f"frame_{idx}{suffix}"

        with open(frame_path, "wb") as f:
            content = await frame.read()
            f.write(content)

        frame_paths.append(frame_path)
    print("step2: save all video frames")
    # Output audio path
    tmp_output_dir = tempfile.mkdtemp()
    output_audio_path = Path(tmp_output_dir) / "response.mp3"
    print("step3: output audio path")
    # Process request through orchestrator
    result = await process_user_request(
        tmp_audio_path,
        frame_paths,
        output_audio_path,
        max_search_duration
    )
    print("already processed user request")
    
    # Read the generated audio file and encode as base64
    if output_audio_path.exists():
        with open(output_audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            result["audio_base64"] = audio_base64
            result["audio_format"] = "mp3"

    return JSONResponse(content=result)
