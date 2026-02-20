import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock the whisper module before app is imported so `import whisper` succeeds
# even when openai-whisper is not installed.
_whisper_mock = MagicMock()
sys.modules.setdefault("whisper", _whisper_mock)

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

    # Ensure model singleton doesn't leak across tests that mock it
    monkeypatch.setattr(app_module, "_model", None)


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
    """Mock whisper.load_model and model.transcribe."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = fake_transcription_result

    with patch("app.whisper") as mock_whisper_module:
        mock_whisper_module.load_model.return_value = mock_model
        # Reset model cache so our mock gets picked up
        import app as app_module
        app_module._model = None
        yield mock_model


@pytest.fixture
def client(mock_whisper):
    from app import app
    with TestClient(app) as c:
        yield c
