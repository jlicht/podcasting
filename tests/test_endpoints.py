"""Tests for API endpoints."""

import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestIndexPage:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Podcast Transcriber" in resp.text


class TestTranscribeUpload:
    def test_upload_mp3(self, client, mock_whisper):
        audio_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 100  # fake mp3 header
        resp = client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["filename"] == "test.mp3"
        assert data["text"] == "Hello world this is a test transcription."
        assert data["language"] == "en"
        assert len(data["segments"]) == 2
        mock_whisper.assert_called_once()

    def test_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/transcribe/upload",
            files={"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")},
            data={"model_size": "base", "language": ""},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_accepts_all_audio_formats(self, client, mock_whisper):
        for ext in ["mp3", "wav", "m4a", "ogg", "flac", "aac", "opus"]:
            resp = client.post(
                "/transcribe/upload",
                files={"file": (f"test.{ext}", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
                data={"model_size": "base", "language": ""},
            )
            assert resp.status_code == 200, f"Failed for .{ext}"


class TestTranscribeUrl:
    def test_url_transcription(self, client, mock_whisper, tmp_path):
        def fake_subprocess_run(cmd, **kwargs):
            # Create a fake downloaded audio file in the tmpdir
            # yt-dlp writes to the --output template dir
            output_template = cmd[cmd.index("--output") + 1]
            out_dir = Path(output_template).parent
            fake_file = out_dir / "Episode Title.mp3"
            fake_file.write_bytes(b"\x00" * 100)

        with patch("app.subprocess.run", side_effect=fake_subprocess_run):
            resp = client.post(
                "/transcribe/url",
                data={"url": "https://example.com/episode.mp3", "model_size": "base", "language": ""},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "Episode Title"
        assert data["text"] == "Hello world this is a test transcription."

    def test_url_download_failure(self, client):
        import subprocess as sp

        with patch("app.subprocess.run", side_effect=sp.CalledProcessError(1, "yt-dlp", stderr="download error")):
            resp = client.post(
                "/transcribe/url",
                data={"url": "https://example.com/bad.mp3", "model_size": "base", "language": ""},
            )

        assert resp.status_code == 400
        assert "Failed to download" in resp.json()["detail"]

    def test_url_download_timeout(self, client):
        import subprocess as sp

        with patch("app.subprocess.run", side_effect=sp.TimeoutExpired("yt-dlp", 300)):
            resp = client.post(
                "/transcribe/url",
                data={"url": "https://example.com/slow.mp3", "model_size": "base", "language": ""},
            )

        assert resp.status_code == 408


class TestTranscriptionHistory:
    def test_list_empty(self, client):
        resp = client.get("/transcriptions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_upload(self, client, mock_whisper):
        # Create a transcription first
        client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )

        resp = client.get("/transcriptions")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["filename"] == "test.mp3"
        assert items[0]["language"] == "en"

    def test_list_includes_has_formatted_false(self, client, mock_whisper):
        client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )

        resp = client.get("/transcriptions")
        items = resp.json()
        assert items[0]["has_formatted"] is False

    def test_list_includes_has_formatted_true(self, client, mock_whisper):
        import app as app_module

        upload_resp = client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )
        job_id = upload_resp.json()["job_id"]

        # Create a fake .md file to simulate formatted output
        md_path = app_module.TRANSCRIPTION_DIR / f"{job_id}.md"
        md_path.write_text("# Formatted")

        resp = client.get("/transcriptions")
        items = resp.json()
        assert items[0]["has_formatted"] is True

    def test_get_specific_transcription(self, client, mock_whisper):
        upload_resp = client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )
        job_id = upload_resp.json()["job_id"]

        resp = client.get(f"/transcriptions/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["text"] == "Hello world this is a test transcription."

    def test_get_nonexistent_transcription(self, client):
        resp = client.get("/transcriptions/does-not-exist")
        assert resp.status_code == 404


class TestFormatEndpoints:
    def _create_transcription(self, client):
        """Helper to create a transcription and return the job_id."""
        resp = client.post(
            "/transcribe/upload",
            files={"file": ("test.mp3", io.BytesIO(b"\x00" * 50), "audio/mpeg")},
            data={"model_size": "base", "language": ""},
        )
        return resp.json()["job_id"]

    def test_format_success(self, client, mock_whisper, mock_anthropic):
        job_id = self._create_transcription(client)

        with patch("formatter.get_anthropic_client", return_value=mock_anthropic):
            resp = client.post(f"/transcriptions/{job_id}/format")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["input_tokens"] == 1000
        assert data["output_tokens"] == 500

    def test_format_with_show_context(self, client, mock_whisper, mock_anthropic):
        job_id = self._create_transcription(client)

        with patch("formatter.get_anthropic_client", return_value=mock_anthropic):
            resp = client.post(
                f"/transcriptions/{job_id}/format",
                json={"show_context": {"show_name": "My Show", "hosts": ["Alice"]}},
            )

        assert resp.status_code == 200

    def test_format_not_found(self, client):
        with patch("formatter.get_anthropic_client", return_value=MagicMock()):
            resp = client.post("/transcriptions/nonexistent/format")
        assert resp.status_code == 404

    def test_format_no_api_key(self, client, mock_whisper):
        job_id = self._create_transcription(client)

        with patch("formatter.get_anthropic_client", return_value=None):
            resp = client.post(f"/transcriptions/{job_id}/format")

        assert resp.status_code == 503

    def test_get_formatted_success(self, client, mock_whisper):
        import app as app_module

        job_id = self._create_transcription(client)
        md_path = app_module.TRANSCRIPTION_DIR / f"{job_id}.md"
        md_path.write_text("# Formatted Transcript\n\n**Host:** Hello.")

        resp = client.get(f"/transcriptions/{job_id}/formatted")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert "# Formatted Transcript" in data["markdown"]

    def test_get_formatted_not_found(self, client):
        resp = client.get("/transcriptions/nonexistent/formatted")
        assert resp.status_code == 404

    def test_download_md(self, client, mock_whisper):
        import app as app_module

        job_id = self._create_transcription(client)
        md_path = app_module.TRANSCRIPTION_DIR / f"{job_id}.md"
        md_path.write_text("# Test Markdown")

        resp = client.get(f"/transcriptions/{job_id}/download/md")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]

    def test_download_docx(self, client, mock_whisper):
        import app as app_module
        from formatter import markdown_to_docx

        job_id = self._create_transcription(client)
        docx_path = app_module.TRANSCRIPTION_DIR / f"{job_id}.docx"
        markdown_to_docx("# Test", docx_path)

        resp = client.get(f"/transcriptions/{job_id}/download/docx")
        assert resp.status_code == 200
        assert "wordprocessingml" in resp.headers["content-type"]

    def test_download_not_found(self, client):
        resp = client.get("/transcriptions/nonexistent/download/md")
        assert resp.status_code == 404

    def test_download_invalid_format(self, client):
        resp = client.get("/transcriptions/some-id/download/pdf")
        assert resp.status_code == 400
