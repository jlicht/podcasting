"""Tests for helper functions."""

import json
from pathlib import Path

from app import _truncate, _save_transcription


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_text_truncated_at_word_boundary(self):
        result = _truncate("hello world foo bar", 12)
        assert result.endswith("...")
        assert len(result) <= 15  # 12 + "..."

    def test_strips_whitespace(self):
        assert _truncate("  hello  ", 100) == "hello"

    def test_empty_string(self):
        assert _truncate("", 10) == ""


class TestSaveTranscription:
    def test_saves_json_file(self, tmp_path, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        result = {
            "text": "test transcript",
            "segments": [{"start": 0.0, "end": 1.0, "text": "test"}],
            "language": "en",
        }
        _save_transcription("job-123", "episode.mp3", result)

        path = tmp_path / "job-123.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["job_id"] == "job-123"
        assert data["filename"] == "episode.mp3"
        assert data["text"] == "test transcript"
        assert data["language"] == "en"
        assert len(data["segments"]) == 1
        assert data["season"] == ""
        assert data["episode_number"] == ""

    def test_saves_season_and_episode(self, tmp_path, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        result = {
            "text": "test transcript",
            "segments": [],
            "language": "en",
        }
        _save_transcription("job-456", "ep.mp3", result, season="2", episode_number="5")

        data = json.loads((tmp_path / "job-456.json").read_text())
        assert data["season"] == "2"
        assert data["episode_number"] == "5"
