import os
import uuid
import json
import tempfile
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import whisper

app = FastAPI(title="Podcast Transcriber")

UPLOAD_DIR = Path("uploads")
TRANSCRIPTION_DIR = Path("transcriptions")
UPLOAD_DIR.mkdir(exist_ok=True)
TRANSCRIPTION_DIR.mkdir(exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".opus"}
MAX_FILE_SIZE_MB = 500

# Load Whisper model lazily
_model = None


def get_model(model_size: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(model_size)
    return _model


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form(""),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id = str(uuid.uuid4())
    audio_path = UPLOAD_DIR / f"{job_id}{ext}"

    with open(audio_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(400, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
        f.write(content)

    result = _run_transcription(str(audio_path), model_size, language or None)

    _save_transcription(job_id, file.filename, result)

    audio_path.unlink(missing_ok=True)

    return JSONResponse({
        "job_id": job_id,
        "filename": file.filename,
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"],
    })


@app.post("/transcribe/url")
async def transcribe_url(
    url: str = Form(...),
    model_size: str = Form("base"),
    language: str = Form(""),
):
    job_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(title)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", output_template,
            "--no-playlist",
            url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise HTTPException(408, "Download timed out after 5 minutes")
        except subprocess.CalledProcessError as e:
            raise HTTPException(400, f"Failed to download audio: {e.stderr[:500]}")

        audio_files = list(Path(tmpdir).glob("*"))
        if not audio_files:
            raise HTTPException(400, "No audio file was downloaded")

        audio_path = str(audio_files[0])
        source_name = audio_files[0].stem

        result = _run_transcription(audio_path, model_size, language or None)

    _save_transcription(job_id, source_name, result)

    return JSONResponse({
        "job_id": job_id,
        "filename": source_name,
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"],
    })


@app.get("/transcriptions")
async def list_transcriptions():
    transcriptions = []
    for f in sorted(TRANSCRIPTION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        with open(f) as fh:
            data = json.load(fh)
            transcriptions.append({
                "job_id": f.stem,
                "filename": data.get("filename", ""),
                "language": data.get("language", ""),
            })
    return JSONResponse(transcriptions)


@app.get("/transcriptions/{job_id}")
async def get_transcription(job_id: str):
    path = TRANSCRIPTION_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "Transcription not found")
    with open(path) as f:
        return JSONResponse(json.load(f))


def _run_transcription(audio_path: str, model_size: str, language: str | None) -> dict:
    model = get_model(model_size)

    options = {}
    if language:
        options["language"] = language

    try:
        result = model.transcribe(audio_path, **options)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language", "unknown"),
    }


def _save_transcription(job_id: str, filename: str, result: dict):
    output = {
        "job_id": job_id,
        "filename": filename,
        **result,
    }
    with open(TRANSCRIPTION_DIR / f"{job_id}.json", "w") as f:
        json.dump(output, f, indent=2)
