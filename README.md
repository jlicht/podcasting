# Podcast Transcriber

A web application for transcribing podcasts using Whisper via [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), optimized for Apple Silicon. Upload audio files or paste URLs to get accurate transcriptions with timestamps.

> **Requires Apple Silicon** (M1 or later). `mlx-whisper` uses the MLX framework which only runs on Apple Silicon Macs.

## Features

- **File upload** — Drag & drop or browse for audio files (MP3, WAV, M4A, OGG, FLAC, AAC, OPUS)
- **URL download** — Paste a podcast episode URL and the app downloads and transcribes it automatically
- **Multiple Whisper models** — Choose from tiny to large depending on your speed/accuracy needs
- **Timestamps** — View transcription with per-segment timestamps
- **Export** — Copy to clipboard, download as plain text, or download as SRT subtitles
- **History** — Browse and reload past transcriptions
- **Language detection** — Automatic language detection or manual override
- **Claude formatting** — Optionally post-process transcripts with Claude to add speaker labels, paragraphs, and section headers (requires `ANTHROPIC_API_KEY`)

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

`ffmpeg` is required by Whisper for audio decoding. `uv` manages Python and project dependencies automatically. This app requires an Apple Silicon Mac (M1 or later).

### 3. Clone and run

```bash
git clone <repo-url> podcasting
cd podcasting
mkdir -p uploads transcriptions
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

To enable Claude-powered transcript formatting, set your API key before starting:

```bash
export ANTHROPIC_API_KEY=your-key-here
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Formatting is optional — transcriptions work fine without an API key.

On the first run `uv` will download Python (if needed), create a virtual environment, and install all dependencies. This may take a few minutes.

Open http://localhost:8000 in your browser.

### 4. Whisper model sizes

Models are downloaded from HuggingFace on first use and cached in `~/.cache/huggingface/`.

| Model  | Parameters | Download size | Relative speed | Best for                     |
|--------|-----------|---------------|----------------|------------------------------|
| tiny   | 39M       | ~75 MB        | ~32x           | Quick drafts                 |
| base   | 74M       | ~150 MB       | ~16x           | Good everyday default        |
| small  | 244M      | ~500 MB       | ~8x            | Better accuracy              |
| medium | 769M      | ~1.5 GB       | ~4x            | High accuracy                |
| turbo  | 809M      | ~800 MB       | ~12x           | Fast + accurate (recommended)|
| large  | 1550M     | ~3 GB         | ~2x            | Best accuracy (16 GB+ Macs) |

`turbo` is the recommended model — it's nearly as accurate as `large` but much faster on Apple Silicon. A MacBook Air with 8 GB RAM can comfortably run `base`, `small`, or `turbo`.

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

## Transcript formatting with Claude

When `ANTHROPIC_API_KEY` is set, transcripts are automatically formatted after transcription. The formatter uses Claude Sonnet to:

- Identify and label speakers (using show context when available)
- Break the wall of text into paragraphs with section headers
- Clean up obvious speech-to-text errors and excessive filler words
- Add an episode title, metadata block, and summary

Formatted output is saved alongside the raw JSON as `.md` (Markdown) and `.docx` (Word) files.

### CLI tool

You can also format existing transcriptions from the command line:

```bash
# Basic usage
ANTHROPIC_API_KEY=your-key uv run python format_transcript.py transcriptions/<job_id>.json

# With show context for better speaker identification
ANTHROPIC_API_KEY=your-key uv run python format_transcript.py transcriptions/<job_id>.json --context show_context.json

# Custom output directory
ANTHROPIC_API_KEY=your-key uv run python format_transcript.py transcriptions/<job_id>.json --output-dir output/
```

### Show context file

Create a `show_context.json` to help Claude identify speakers:

```json
{
  "show_name": "My Podcast",
  "show_description": "A weekly show about technology",
  "hosts": ["Alice Smith", "Bob Jones"],
  "guests": []
}
```

## API Endpoints

- `POST /transcribe/upload` — Upload an audio file for transcription
- `POST /transcribe/url` — Provide a URL to download and transcribe
- `GET /transcriptions` — List all past transcriptions (includes `has_formatted` field)
- `GET /transcriptions/{job_id}` — Get a specific transcription result
- `POST /transcriptions/{job_id}/format` — Format/re-format a transcription with Claude
- `GET /transcriptions/{job_id}/formatted` — Get the formatted markdown content
- `GET /transcriptions/{job_id}/download/{md|docx}` — Download formatted file
