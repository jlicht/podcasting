import logging
import os
import re
import uuid
import json
import tempfile
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import feedparser
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlx_whisper

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Podcast Transcriber")

UPLOAD_DIR = Path("uploads")
TRANSCRIPTION_DIR = Path("transcriptions")
UPLOAD_DIR.mkdir(exist_ok=True)
TRANSCRIPTION_DIR.mkdir(exist_ok=True)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac", ".opus"}
MAX_FILE_SIZE_MB = 500

MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "turbo": "mlx-community/whisper-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",
}

# Serialize GPU access for transcription
_transcribe_lock = threading.Lock()


templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Transcription queue
# ---------------------------------------------------------------------------

class QueueItem:
    def __init__(self, job_id: str, title: str, audio_url: str, model_size: str, language: str | None,
                 season: str = "", episode_number: str = "",
                 show_context: dict | None = None):
        self.job_id = job_id
        self.title = title
        self.audio_url = audio_url
        self.model_size = model_size
        self.language = language
        self.season = season
        self.episode_number = episode_number
        self.show_context = show_context
        self.status = "pending"  # pending | downloading | transcribing | completed | failed
        self.error: str | None = None
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.step_detail: str = ""


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
                    qi.started_at = time.time()
                    qi.step_detail = "Downloading audio..."
                    item = qi
                    break
        if item is None:
            # No more work
            _worker_running = False
            return

        logger.info("Starting download: %s (%s)", item.title, item.audio_url)
        try:
            _process_queue_item(item)
        except Exception as e:
            item.status = "failed"
            item.completed_at = time.time()
            item.error = str(e)[:500]
            item.step_detail = f"Failed: {str(e)[:200]}"
            logger.error("Item failed unexpectedly: %s — %s", item.title, e)


def _process_queue_item(item: QueueItem):
    download_start = time.time()

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
            item.completed_at = time.time()
            item.error = "Download timed out after 10 minutes"
            item.step_detail = "Download timed out after 10 minutes"
            logger.warning("Download timed out: %s", item.title)
            return
        except subprocess.CalledProcessError as e:
            item.status = "failed"
            item.completed_at = time.time()
            item.error = f"Download failed: {e.stderr[:500]}"
            item.step_detail = "Download failed"
            logger.error("Download failed: %s — %s", item.title, e.stderr[:200])
            return

        audio_files = list(Path(tmpdir).glob("*"))
        if not audio_files:
            item.status = "failed"
            item.completed_at = time.time()
            item.error = "No audio file was downloaded"
            item.step_detail = "No audio file found after download"
            logger.warning("No audio file found: %s", item.title)
            return

        audio_path = str(audio_files[0])
        file_size = audio_files[0].stat().st_size
        download_elapsed = round(time.time() - download_start)
        size_mb = round(file_size / (1024 * 1024), 1)
        item.step_detail = f"Downloaded {size_mb} MB in {download_elapsed}s"
        logger.info("Download complete: %s — %.1f MB in %ds", item.title, size_mb, download_elapsed)

        item.status = "transcribing"
        item.step_detail = f"Transcribing with {item.model_size} model..."
        logger.info("Starting transcription: %s (model=%s)", item.title, item.model_size)

        transcribe_start = time.time()
        try:
            result = _run_transcription(audio_path, item.model_size, item.language)
        except Exception as e:
            item.status = "failed"
            item.completed_at = time.time()
            item.error = f"Transcription failed: {str(e)[:500]}"
            item.step_detail = "Transcription failed"
            logger.error("Transcription failed: %s — %s", item.title, e)
            return

    transcribe_elapsed = round(time.time() - transcribe_start)
    logger.info("Transcription complete: %s — %ds, language=%s", item.title, transcribe_elapsed, result.get("language", "unknown"))

    _save_transcription(item.job_id, item.title, result,
                        season=item.season, episode_number=item.episode_number)

    _try_format_transcript(item, result)

    item.status = "completed"
    item.completed_at = time.time()
    item.step_detail = f"Completed in {round(item.completed_at - item.started_at)}s" if item.started_at else "Completed"


def _try_format_transcript(item: QueueItem, result: dict):
    """Attempt to format the transcript with Claude. Never raises."""
    try:
        from formatter import (
            ShowContext,
            TranscriptData,
            FormatResult,
            format_transcript,
            save_formatted_output,
            get_anthropic_client,
        )

        client = get_anthropic_client()
        if client is None:
            logger.info("No ANTHROPIC_API_KEY set, skipping transcript formatting")
            return

        item.step_detail = "Formatting transcript with Claude..."

        transcript = TranscriptData(
            job_id=item.job_id,
            filename=item.title,
            text=result["text"],
            segments=result.get("segments", []),
            language=result.get("language", ""),
            season=item.season,
            episode_number=item.episode_number,
        )

        context = ShowContext.from_dict(item.show_context) if item.show_context else ShowContext()

        fmt_result = format_transcript(transcript, context, client=client)
        save_formatted_output(fmt_result, TRANSCRIPTION_DIR, item.job_id)
        logger.info("Formatted transcript saved for %s (tokens: %d in, %d out)",
                     item.title, fmt_result.input_tokens, fmt_result.output_tokens)
    except Exception as e:
        logger.warning("Auto-format failed for %s: %s", item.title, e)


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

_APPLE_PODCAST_RE = re.compile(
    r"https?://podcasts\.apple\.com/.+/podcast/.+/id(\d+)"
)


def _resolve_feed_url(url: str) -> str:
    """If *url* is an Apple Podcasts link, resolve it to the actual RSS feed URL
    via the iTunes Lookup API.  Otherwise return *url* unchanged."""
    m = _APPLE_PODCAST_RE.match(url.strip())
    if not m:
        return url
    podcast_id = m.group(1)
    lookup_url = f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast"
    try:
        with urlopen(lookup_url, timeout=10) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results and results[0].get("feedUrl"):
            return results[0]["feedUrl"]
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        pass
    raise HTTPException(
        400,
        "Could not resolve Apple Podcasts URL to an RSS feed. "
        "The podcast may not have a public RSS feed.",
    )


@app.post("/feed/fetch")
async def fetch_feed(url: str = Form(...)):
    feed = feedparser.parse(_resolve_feed_url(url))

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


@app.get("/config")
async def get_config():
    return JSONResponse({
        "default_feed_url": os.environ.get("DEFAULT_FEED_URL", ""),
    })


class QueueRequest(BaseModel):
    episodes: list[dict]
    model_size: str = "base"
    language: str = ""
    show_context: dict | None = None


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
                season=str(ep.get("season", "")),
                episode_number=str(ep.get("episode_number", "")),
                show_context=req.show_context,
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
                "started_at": qi.started_at,
                "completed_at": qi.completed_at,
                "step_detail": qi.step_detail,
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
                "has_formatted": (TRANSCRIPTION_DIR / f"{f.stem}.md").exists(),
            })
    return JSONResponse(transcriptions)


@app.get("/transcriptions/{job_id}")
async def get_transcription(job_id: str):
    path = TRANSCRIPTION_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "Transcription not found")
    with open(path) as f:
        return JSONResponse(json.load(f))


class FormatRequest(BaseModel):
    show_context: dict | None = None


@app.post("/transcriptions/{job_id}/format")
async def format_transcription(job_id: str, req: FormatRequest | None = None):
    """Format (or re-format) an existing transcription with Claude."""
    from formatter import (
        ShowContext,
        TranscriptData,
        format_transcript as fmt_transcript,
        save_formatted_output,
        get_anthropic_client,
    )

    json_path = TRANSCRIPTION_DIR / f"{job_id}.json"
    if not json_path.exists():
        raise HTTPException(404, "Transcription not found")

    client = get_anthropic_client()
    if client is None:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not configured")

    transcript = TranscriptData.from_json_file(json_path)
    show_ctx = req.show_context if req else None
    context = ShowContext.from_dict(show_ctx) if show_ctx else ShowContext()

    try:
        result = fmt_transcript(transcript, context, client=client)
    except Exception as e:
        raise HTTPException(502, f"Claude API error: {str(e)[:500]}")

    try:
        md_path, docx_path = save_formatted_output(result, TRANSCRIPTION_DIR, job_id)
    except Exception as e:
        logger.warning("DOCX conversion failed: %s", e)
        # Still save the markdown even if DOCX fails
        md_path = TRANSCRIPTION_DIR / f"{job_id}.md"
        md_path.write_text(result.markdown, encoding="utf-8")

    return JSONResponse({
        "job_id": job_id,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "md_path": str(md_path),
    })


@app.get("/transcriptions/{job_id}/formatted")
async def get_formatted_transcription(job_id: str):
    """Return the formatted markdown content."""
    md_path = TRANSCRIPTION_DIR / f"{job_id}.md"
    if not md_path.exists():
        raise HTTPException(404, "Formatted transcription not found")
    return JSONResponse({"job_id": job_id, "markdown": md_path.read_text(encoding="utf-8")})


@app.get("/transcriptions/{job_id}/download/{fmt}")
async def download_formatted(job_id: str, fmt: str):
    """Download the formatted transcript as .md or .docx."""
    from fastapi.responses import FileResponse
    from formatter import TranscriptData, build_output_stem

    if fmt not in ("md", "docx"):
        raise HTTPException(400, "Format must be 'md' or 'docx'")

    file_path = TRANSCRIPTION_DIR / f"{job_id}.{fmt}"
    if not file_path.exists():
        raise HTTPException(404, f"Formatted file not found ({fmt})")

    # Build a human-readable filename from episode metadata
    json_path = TRANSCRIPTION_DIR / f"{job_id}.json"
    if json_path.exists():
        transcript = TranscriptData.from_json_file(json_path)
        stem = build_output_stem(transcript)
    else:
        stem = job_id

    media_type = (
        "text/markdown" if fmt == "md"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        path=str(file_path),
        filename=f"{stem}.{fmt}",
        media_type=media_type,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_transcription(audio_path: str, model_size: str, language: str | None) -> dict:
    repo = MODEL_REPOS.get(model_size, MODEL_REPOS["base"])
    options: dict = {"path_or_hf_repo": repo}
    if language:
        options["language"] = language

    try:
        with _transcribe_lock:
            result = mlx_whisper.transcribe(audio_path, **options)
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


def _save_transcription(job_id: str, filename: str, result: dict, *,
                        season: str = "", episode_number: str = ""):
    output = {
        "job_id": job_id,
        "filename": filename,
        "season": season,
        "episode_number": episode_number,
        **result,
    }
    with open(TRANSCRIPTION_DIR / f"{job_id}.json", "w") as f:
        json.dump(output, f, indent=2)


def _truncate(text: str, length: int) -> str:
    text = text.strip()
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "..."
