# Podcast Transcriber

A web application for transcribing podcasts using OpenAI's Whisper model. Upload audio files or paste URLs to get accurate transcriptions with timestamps.

## Features

- **File upload** — Drag & drop or browse for audio files (MP3, WAV, M4A, OGG, FLAC, AAC, OPUS)
- **URL download** — Paste a podcast episode URL and the app downloads and transcribes it automatically
- **Multiple Whisper models** — Choose from tiny to large depending on your speed/accuracy needs
- **Timestamps** — View transcription with per-segment timestamps
- **Export** — Copy to clipboard, download as plain text, or download as SRT subtitles
- **History** — Browse and reload past transcriptions
- **Language detection** — Automatic language detection or manual override

## Quick start on a MacBook Air

These steps assume a fresh Mac with nothing installed. Open Terminal (Applications > Utilities > Terminal) and run the following.

### 1. Install Homebrew (if you don't have it)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions to add `brew` to your PATH.

### 2. Install system dependencies

```bash
brew install ffmpeg uv
```

`ffmpeg` is required by Whisper for audio decoding. `uv` manages Python and project dependencies automatically.

### 3. Clone and run

```bash
git clone <repo-url> podcasting
cd podcasting
mkdir -p uploads transcriptions
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

On the first run `uv` will download Python (if needed), create a virtual environment, and install all dependencies. This may take a few minutes.

Open http://localhost:8000 in your browser.

### 4. Whisper model sizes

The model is downloaded once on the first transcription request and cached in `~/.cache/whisper/`.

| Model  | Parameters | RAM needed | Relative speed | Best for                     |
|--------|-----------|------------|----------------|------------------------------|
| tiny   | 39M       | ~1 GB      | ~10x           | Quick drafts                 |
| base   | 74M       | ~1 GB      | ~7x            | Good everyday default        |
| small  | 244M      | ~2 GB      | ~4x            | Better accuracy              |
| medium | 769M      | ~5 GB      | ~2x            | High accuracy                |
| large  | 1550M     | ~10 GB     | 1x             | Best accuracy (16 GB+ Macs) |

A MacBook Air with 8 GB RAM can comfortably run `base` or `small`. Use `medium` only if you have 16 GB. `large` needs 16 GB+ and will be slow without a GPU.

### 5. Stopping the server

Press `Ctrl+C` in the Terminal window where the server is running.

## Development setup

```bash
brew install ffmpeg uv
git clone <repo-url> podcasting
cd podcasting
mkdir -p uploads transcriptions

# Run tests
uv run pytest tests/ -v

# Run server with auto-reload
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /transcribe/upload` — Upload an audio file for transcription
- `POST /transcribe/url` — Provide a URL to download and transcribe
- `GET /transcriptions` — List all past transcriptions
- `GET /transcriptions/{job_id}` — Get a specific transcription result
