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

## Prerequisites

- Python 3.10+
- FFmpeg (required by Whisper)

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (with chocolatey)
choco install ffmpeg
```

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

### Whisper Model Sizes

| Model  | Parameters | Speed    | Accuracy |
|--------|-----------|----------|----------|
| tiny   | 39M       | Fastest  | Lower    |
| base   | 74M       | Fast     | Good     |
| small  | 244M      | Moderate | Better   |
| medium | 769M      | Slow     | High     |
| large  | 1550M     | Slowest  | Best     |

The model is loaded on first transcription request. Larger models require more RAM/VRAM.

## API Endpoints

- `POST /transcribe/upload` — Upload an audio file for transcription
- `POST /transcribe/url` — Provide a URL to download and transcribe
- `GET /transcriptions` — List all past transcriptions
- `GET /transcriptions/{job_id}` — Get a specific transcription result
