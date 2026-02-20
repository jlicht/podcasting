import os
import uuid
import json
import tempfile
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import feedparser
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
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
_model_lock = threading.Lock()


def get_model(model_size: str = "base"):
    global _model
    with _model_lock:
        if _model is None:
            _model = whisper.load_model(model_size)
    return _model


templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Transcription queue
# ---------------------------------------------------------------------------

class QueueItem:
    def __init__(self, job_id: str, title: str, audio_url: str, model_size: str, language: str | None):
        self.job_id = job_id
        self.title = title
        self.audio_url = audio_url
        self.model_size = model_size
        self.language = language
        self.status = "pending"  # pending | downloading | transcribing | completed | failed
        self.error: str | None = None


# Ordered dict preserves insertion order for display
_queue: OrderedDict[str, QueueItem] = OrderedDict()
_queue_lock = threading.Lock()
_worker_running = False


def _start_worker():
    global _worker_running
    if _worker_running:
        return
    _worker_running = True
    t = threading.Thread(target=_queue_worker, daemon=True)
    t.start()


def _queue_worker():
    global _worker_running
    while True:
        item = None
        with _queue_lock:
            for qi in _queue.values():
                if qi.status == "pending":
                    qi.status = "downloading"
                    item = qi
                    break
        if item is None:
            # No more work
            _worker_running = False
            return

        try:
            _process_queue_item(item)
        except Exception as e:
            item.status = "failed"
            item.error = str(e)[:500]


def _process_queue_item(item: QueueItem):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(title)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", output_template,
            "--no-playlist",
            item.audio_url,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            item.status = "failed"
            item.error = "Download timed out after 10 minutes"
            return
        except subprocess.CalledProcessError as e:
            item.status = "failed"
            item.error = f"Download failed: {e.stderr[:500]}"
            return

        audio_files = list(Path(tmpdir).glob("*"))
        if not audio_files:
            item.status = "failed"
            item.error = "No audio file was downloaded"
            return

        audio_path = str(audio_files[0])

        item.status = "transcribing"

        try:
            result = _run_transcription(audio_path, item.model_size, item.language)
        except Exception as e:
            item.status = "failed"
            item.error = f"Transcription failed: {str(e)[:500]}"
            return

    _save_transcription(item.job_id, item.title, result)
    item.status = "completed"


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Routes — single-file transcription
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Routes — podcast feed
# ---------------------------------------------------------------------------

@app.post("/feed/fetch")
async def fetch_feed(url: str = Form(...)):
    feed = feedparser.parse(url)

    if feed.bozo and not feed.entries:
        raise HTTPException(400, f"Failed to parse feed: {str(feed.bozo_exception)[:200]}")

    show_info = {
        "title": feed.feed.get("title", "Unknown Show"),
        "description": feed.feed.get("summary", feed.feed.get("subtitle", "")),
        "image": "",
        "link": feed.feed.get("link", ""),
    }

    # Try to get the show artwork
    if hasattr(feed.feed, "image") and hasattr(feed.feed.image, "href"):
        show_info["image"] = feed.feed.image.href
    elif "itunes_image" in feed.feed:
        show_info["image"] = feed.feed.itunes_image.get("href", "")

    episodes = []
    for entry in feed.entries:
        audio_url = ""
        audio_type = ""
        audio_length = 0

        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio/") or link.get("rel") == "enclosure":
                audio_url = link.get("href", "")
                audio_type = link.get("type", "")
                try:
                    audio_length = int(link.get("length", 0))
                except (ValueError, TypeError):
                    audio_length = 0
                break

        if not audio_url:
            for enc in entry.get("enclosures", []):
                audio_url = enc.get("href", enc.get("url", ""))
                audio_type = enc.get("type", "")
                try:
                    audio_length = int(enc.get("length", 0))
                except (ValueError, TypeError):
                    audio_length = 0
                if audio_url:
                    break

        if not audio_url:
            continue

        # Duration — try itunes:duration first, then fallback
        duration = entry.get("itunes_duration", "")

        # Published date
        published = entry.get("published", entry.get("updated", ""))

        # Episode number
        ep_number = entry.get("itunes_episode", "")
        season = entry.get("itunes_season", "")

        episodes.append({
            "title": entry.get("title", "Untitled"),
            "published": published,
            "duration": duration,
            "description": _truncate(entry.get("summary", entry.get("subtitle", "")), 300),
            "audio_url": audio_url,
            "audio_type": audio_type,
            "audio_length_bytes": audio_length,
            "episode_number": ep_number,
            "season": season,
        })

    return JSONResponse({
        "show": show_info,
        "episodes": episodes,
        "total": len(episodes),
    })


class QueueRequest(BaseModel):
    episodes: list[dict]
    model_size: str = "base"
    language: str = ""


@app.post("/queue/add")
async def queue_add(req: QueueRequest):
    job_ids = []
    with _queue_lock:
        for ep in req.episodes:
            audio_url = ep.get("audio_url", "")
            title = ep.get("title", "Untitled")
            if not audio_url:
                continue
            job_id = str(uuid.uuid4())
            item = QueueItem(
                job_id=job_id,
                title=title,
                audio_url=audio_url,
                model_size=req.model_size,
                language=req.language or None,
            )
            _queue[job_id] = item
            job_ids.append({"job_id": job_id, "title": title})

    _start_worker()

    return JSONResponse({"queued": job_ids})


@app.get("/queue/status")
async def queue_status():
    with _queue_lock:
        items = []
        for qi in _queue.values():
            items.append({
                "job_id": qi.job_id,
                "title": qi.title,
                "status": qi.status,
                "error": qi.error,
            })
    return JSONResponse(items)


@app.post("/queue/clear")
async def queue_clear():
    with _queue_lock:
        to_remove = [k for k, v in _queue.items() if v.status in ("completed", "failed")]
        for k in to_remove:
            del _queue[k]
    return JSONResponse({"cleared": len(to_remove)})


# ---------------------------------------------------------------------------
# Routes — transcription history
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _truncate(text: str, length: int) -> str:
    text = text.strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."
