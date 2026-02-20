import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock the mlx_whisper module before app is imported so `import mlx_whisper` succeeds
# even when mlx-whisper is not installed.
_whisper_mock = MagicMock()
sys.modules.setdefault("mlx_whisper", _whisper_mock)

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _tmp_dirs(monkeypatch, tmp_path):
    """Redirect upload and transcription dirs to temp folders for every test."""
    upload_dir = tmp_path / "uploads"
    transcription_dir = tmp_path / "transcriptions"
    upload_dir.mkdir()
    transcription_dir.mkdir()

    import app as app_module

    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", transcription_dir)

    # Reset queue state between tests
    with app_module._queue_lock:
        app_module._queue.clear()
    monkeypatch.setattr(app_module, "_worker_running", False)

    yield


@pytest.fixture
def fake_transcription_result():
    return {
        "text": "Hello world this is a test transcription.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Hello world"},
            {"start": 2.5, "end": 5.0, "text": " this is a test transcription."},
        ],
        "language": "en",
    }


@pytest.fixture
def mock_whisper(fake_transcription_result):
    """Mock mlx_whisper.transcribe as a function."""
    with patch("app.mlx_whisper.transcribe", return_value=fake_transcription_result) as mock_transcribe:
        yield mock_transcribe


@pytest.fixture
def client(mock_whisper):
    from app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_anthropic():
    """Mock anthropic.Anthropic client returning a fake formatted response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="# Episode Title\n\n**Host:** Hello and welcome.\n\n## Topic One\n\n**Guest 1:** Thanks for having me.")]
    mock_response.usage.input_tokens = 1000
    mock_response.usage.output_tokens = 500
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client
