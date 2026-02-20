"""Tests for the transcription queue."""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess as sp

import pytest


class TestQueueAdd:
    def test_queue_episodes(self, client):
        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={
                    "episodes": [
                        {"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"},
                        {"title": "Ep 2", "audio_url": "https://example.com/ep2.mp3"},
                    ],
                    "model_size": "base",
                    "language": "",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["queued"]) == 2
        assert data["queued"][0]["title"] == "Ep 1"
        assert data["queued"][1]["title"] == "Ep 2"
        assert "job_id" in data["queued"][0]

    def test_queue_skips_episodes_without_url(self, client):
        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={
                    "episodes": [
                        {"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"},
                        {"title": "No URL", "audio_url": ""},
                        {"title": "Also No URL"},
                    ],
                    "model_size": "base",
                },
            )

        data = resp.json()
        assert len(data["queued"]) == 1

    def test_queue_empty_list(self, client):
        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={"episodes": [], "model_size": "base"},
            )

        data = resp.json()
        assert len(data["queued"]) == 0


class TestQueueStatus:
    def test_empty_queue(self, client):
        resp = client.get("/queue/status")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_status_after_adding(self, client):
        with patch("app._start_worker"):
            client.post(
                "/queue/add",
                json={
                    "episodes": [{"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"}],
                    "model_size": "base",
                },
            )

        resp = client.get("/queue/status")
        items = resp.json()
        assert len(items) == 1
        assert items[0]["title"] == "Ep 1"
        assert items[0]["status"] == "pending"
        assert items[0]["error"] is None
        assert items[0]["started_at"] is None
        assert items[0]["completed_at"] is None
        assert items[0]["step_detail"] == ""


class TestQueueClear:
    def test_clears_completed_items(self, client):
        import app as app_module

        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={
                    "episodes": [
                        {"title": "Done", "audio_url": "https://example.com/done.mp3"},
                        {"title": "Failed", "audio_url": "https://example.com/fail.mp3"},
                        {"title": "Pending", "audio_url": "https://example.com/pending.mp3"},
                    ],
                    "model_size": "base",
                },
            )

        # Manually set statuses
        with app_module._queue_lock:
            items = list(app_module._queue.values())
            items[0].status = "completed"
            items[1].status = "failed"
            items[2].status = "pending"

        resp = client.post("/queue/clear")
        assert resp.json()["cleared"] == 2

        resp = client.get("/queue/status")
        remaining = resp.json()
        assert len(remaining) == 1
        assert remaining[0]["title"] == "Pending"


class TestQueueShowContext:
    def test_show_context_passes_through(self, client):
        import app as app_module

        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={
                    "episodes": [{"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"}],
                    "model_size": "base",
                    "show_context": {"show_name": "Test Show", "hosts": ["Alice"]},
                },
            )

        assert resp.status_code == 200
        job_id = resp.json()["queued"][0]["job_id"]

        with app_module._queue_lock:
            item = app_module._queue[job_id]
            assert item.show_context == {"show_name": "Test Show", "hosts": ["Alice"]}

    def test_show_context_defaults_to_none(self, client):
        import app as app_module

        with patch("app._start_worker"):
            resp = client.post(
                "/queue/add",
                json={
                    "episodes": [{"title": "Ep 1", "audio_url": "https://example.com/ep1.mp3"}],
                    "model_size": "base",
                },
            )

        job_id = resp.json()["queued"][0]["job_id"]
        with app_module._queue_lock:
            item = app_module._queue[job_id]
            assert item.show_context is None


class TestProcessQueueItem:
    def test_successful_processing(self, client, mock_whisper, tmp_path, monkeypatch):
        import app as app_module

        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        item = app_module.QueueItem(
            job_id="test-job",
            title="Test Episode",
            audio_url="https://example.com/ep.mp3",
            model_size="base",
            language=None,
        )
        item.started_at = time.time()

        def fake_subprocess_run(cmd, **kwargs):
            output_template = cmd[cmd.index("--output") + 1]
            out_dir = Path(output_template).parent
            (out_dir / "episode.mp3").write_bytes(b"\x00" * 100)

        with patch("app.subprocess.run", side_effect=fake_subprocess_run):
            app_module._process_queue_item(item)

        assert item.status == "completed"
        assert (tmp_path / "test-job.json").exists()
        assert item.started_at is not None
        assert item.completed_at is not None
        assert item.completed_at >= item.started_at
        assert item.step_detail != ""

    def test_download_timeout(self, client, monkeypatch):
        import app as app_module

        item = app_module.QueueItem(
            job_id="test-timeout",
            title="Slow Episode",
            audio_url="https://example.com/slow.mp3",
            model_size="base",
            language=None,
        )

        with patch("app.subprocess.run", side_effect=sp.TimeoutExpired("yt-dlp", 600)):
            app_module._process_queue_item(item)

        assert item.status == "failed"
        assert "timed out" in item.error
        assert item.completed_at is not None
        assert "timed out" in item.step_detail.lower()

    def test_download_failure(self, client, monkeypatch):
        import app as app_module

        item = app_module.QueueItem(
            job_id="test-fail",
            title="Bad Episode",
            audio_url="https://example.com/bad.mp3",
            model_size="base",
            language=None,
        )

        with patch("app.subprocess.run", side_effect=sp.CalledProcessError(1, "yt-dlp", stderr="404 not found")):
            app_module._process_queue_item(item)

        assert item.status == "failed"
        assert "Download failed" in item.error
        assert item.completed_at is not None
        assert item.step_detail != ""

    def test_no_audio_file_downloaded(self, client, monkeypatch):
        import app as app_module

        item = app_module.QueueItem(
            job_id="test-empty",
            title="Empty Episode",
            audio_url="https://example.com/empty.mp3",
            model_size="base",
            language=None,
        )

        with patch("app.subprocess.run"):  # succeeds but writes no files
            app_module._process_queue_item(item)

        assert item.status == "failed"
        assert "No audio file" in item.error
        assert item.completed_at is not None
        assert "audio file" in item.step_detail.lower()


class TestAutoFormat:
    def test_auto_format_called_when_api_key_set(self, client, mock_whisper, mock_anthropic, tmp_path, monkeypatch):
        import app as app_module

        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        item = app_module.QueueItem(
            job_id="test-fmt",
            title="Test Episode",
            audio_url="https://example.com/ep.mp3",
            model_size="base",
            language=None,
            show_context={"show_name": "My Show"},
        )
        item.started_at = time.time()

        def fake_subprocess_run(cmd, **kwargs):
            output_template = cmd[cmd.index("--output") + 1]
            out_dir = Path(output_template).parent
            (out_dir / "episode.mp3").write_bytes(b"\x00" * 100)

        with patch("app.subprocess.run", side_effect=fake_subprocess_run), \
             patch("formatter.get_anthropic_client", return_value=mock_anthropic):
            app_module._process_queue_item(item)

        assert item.status == "completed"
        assert (tmp_path / "test-fmt.json").exists()
        assert (tmp_path / "test-fmt.md").exists()
        assert (tmp_path / "test-fmt.docx").exists()

    def test_auto_format_skipped_when_no_api_key(self, client, mock_whisper, tmp_path, monkeypatch):
        import app as app_module

        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        item = app_module.QueueItem(
            job_id="test-nofmt",
            title="Test Episode",
            audio_url="https://example.com/ep.mp3",
            model_size="base",
            language=None,
        )
        item.started_at = time.time()

        def fake_subprocess_run(cmd, **kwargs):
            output_template = cmd[cmd.index("--output") + 1]
            out_dir = Path(output_template).parent
            (out_dir / "episode.mp3").write_bytes(b"\x00" * 100)

        with patch("app.subprocess.run", side_effect=fake_subprocess_run), \
             patch("formatter.get_anthropic_client", return_value=None):
            app_module._process_queue_item(item)

        assert item.status == "completed"
        assert (tmp_path / "test-nofmt.json").exists()
        assert not (tmp_path / "test-nofmt.md").exists()

    def test_auto_format_failure_doesnt_fail_transcription(self, client, mock_whisper, tmp_path, monkeypatch):
        import app as app_module

        monkeypatch.setattr(app_module, "TRANSCRIPTION_DIR", tmp_path)

        item = app_module.QueueItem(
            job_id="test-fmterr",
            title="Test Episode",
            audio_url="https://example.com/ep.mp3",
            model_size="base",
            language=None,
        )
        item.started_at = time.time()

        def fake_subprocess_run(cmd, **kwargs):
            output_template = cmd[cmd.index("--output") + 1]
            out_dir = Path(output_template).parent
            (out_dir / "episode.mp3").write_bytes(b"\x00" * 100)

        # Mock client that raises an error
        bad_client = MagicMock()
        bad_client.messages.create.side_effect = Exception("API error")

        with patch("app.subprocess.run", side_effect=fake_subprocess_run), \
             patch("formatter.get_anthropic_client", return_value=bad_client):
            app_module._process_queue_item(item)

        # Transcription should still succeed
        assert item.status == "completed"
        assert (tmp_path / "test-fmterr.json").exists()
        # But no formatted files
        assert not (tmp_path / "test-fmterr.md").exists()
