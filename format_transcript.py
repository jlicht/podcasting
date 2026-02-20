#!/usr/bin/env python3
"""CLI tool to format raw Whisper transcriptions using Claude.

Usage:
    # Single file
    uv run python format_transcript.py transcriptions/<job_id>.json --context show_context.json

    # All unprocessed files in a directory (newest first, 4 workers)
    uv run python format_transcript.py transcriptions/ --context show_context.json

    # Custom parallelism
    uv run python format_transcript.py transcriptions/ --context show_context.json --workers 2
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from formatter import (
    ShowContext,
    TranscriptData,
    build_output_stem,
    format_transcript,
    save_formatted_output,
)


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _spinner(message, stop_event):
    """Show a spinning progress indicator until stop_event is set."""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    start = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start
        sys.stderr.write(f"\r{chars[i % len(chars)]} {message} ({elapsed:.0f}s)")
        sys.stderr.flush()
        i += 1
        stop_event.wait(0.1)
    elapsed = time.time() - start
    sys.stderr.write(f"\r✓ {message} ({elapsed:.0f}s)\n")
    sys.stderr.flush()


@dataclass
class FormatOneResult:
    path: Path
    label: str
    success: bool
    elapsed: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    md_path: Path | None = None
    docx_path: Path | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Unprocessed file discovery
# ---------------------------------------------------------------------------

def _is_processed(transcript_path: Path, output_dir: Path) -> bool:
    """Check whether a transcript JSON already has a formatted .md in the output dir."""
    transcript = TranscriptData.from_json_file(transcript_path)
    stem = build_output_stem(transcript)
    return (output_dir / f"{stem}.md").exists()


def _find_unprocessed(source_dir: Path, output_dir: Path) -> list[Path]:
    """Return unprocessed .json transcript files sorted newest-first."""
    all_json = sorted(
        source_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [p for p in all_json if not _is_processed(p, output_dir)]


# ---------------------------------------------------------------------------
# Single-file formatting (used for both single and batch modes)
# ---------------------------------------------------------------------------

def _format_one(transcript_path: Path, context: ShowContext, output_dir: Path) -> FormatOneResult:
    """Format a single transcript file. Returns a result object."""
    transcript = TranscriptData.from_json_file(transcript_path)
    label = build_output_stem(transcript)
    start = time.time()

    try:
        result = format_transcript(transcript, context)
    except Exception as e:
        return FormatOneResult(
            path=transcript_path, label=label, success=False,
            elapsed=time.time() - start, error=str(e),
        )

    md_path, docx_path = save_formatted_output(
        result, output_dir, transcript.job_id, transcript=transcript,
    )

    return FormatOneResult(
        path=transcript_path, label=label, success=True,
        elapsed=time.time() - start,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        md_path=md_path, docx_path=docx_path,
    )


# ---------------------------------------------------------------------------
# Single-file mode (interactive with spinner)
# ---------------------------------------------------------------------------

def _run_single(source: Path, context: ShowContext, output_dir: Path):
    transcript = TranscriptData.from_json_file(source)
    label = build_output_stem(transcript)
    print(f"Loaded transcript: {label} ({len(transcript.text):,} chars)")

    stop = threading.Event()
    spinner_thread = threading.Thread(
        target=_spinner, args=("Formatting with Claude...", stop), daemon=True,
    )
    spinner_thread.start()

    r = _format_one(source, context, output_dir)

    stop.set()
    spinner_thread.join()

    if not r.success:
        print(f"Error: {r.error}", file=sys.stderr)
        sys.exit(1)

    print(f"Tokens: {r.input_tokens:,} in / {r.output_tokens:,} out")
    print(f"Saved: {r.md_path}")
    print(f"Saved: {r.docx_path}")


# ---------------------------------------------------------------------------
# Batch mode (parallel with live progress)
# ---------------------------------------------------------------------------

def _run_batch(source: Path, context: ShowContext, output_dir: Path, workers: int):
    unprocessed = _find_unprocessed(source, output_dir)
    total_json = len(list(source.glob("*.json")))
    already_done = total_json - len(unprocessed)

    print(f"\nFound {total_json} transcript(s) in {source}/")
    if already_done:
        print(f"  {already_done} already formatted, {len(unprocessed)} remaining")
    if not unprocessed:
        print("Nothing to do.")
        return

    effective_workers = min(workers, len(unprocessed))
    print(f"  Processing with {effective_workers} parallel worker(s)\n")

    succeeded = 0
    failed = 0
    total_in = 0
    total_out = 0
    batch_start = time.time()

    # Status line state
    completed_count = 0
    active_labels: dict[int, str] = {}  # worker_index -> label
    status_lock = threading.Lock()
    stop_status = threading.Event()

    def _status_line():
        """Redraw a single live progress line on stderr."""
        while not stop_status.is_set():
            with status_lock:
                active = list(active_labels.values())
                n = completed_count
            elapsed = time.time() - batch_start
            active_str = ", ".join(active[:4]) if active else "waiting..."
            sys.stderr.write(
                f"\r⠹ [{n}/{len(unprocessed)}] ({elapsed:.0f}s) {active_str}    "
            )
            sys.stderr.flush()
            stop_status.wait(0.2)
        # Clear the status line
        sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()

    status_thread = threading.Thread(target=_status_line, daemon=True)
    status_thread.start()

    def _worker(path: Path, index: int) -> FormatOneResult:
        transcript = TranscriptData.from_json_file(path)
        label = build_output_stem(transcript)
        # Truncate long labels for the status line
        short_label = label if len(label) <= 30 else label[:27] + "..."
        with status_lock:
            active_labels[index] = short_label
        r = _format_one(path, context, output_dir)
        with status_lock:
            del active_labels[index]
        return r

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(_worker, path, i): path
            for i, path in enumerate(unprocessed)
        }

        for future in as_completed(futures):
            r = future.result()
            with status_lock:
                completed_count += 1

            if r.success:
                succeeded += 1
                total_in += r.input_tokens
                total_out += r.output_tokens
                with _print_lock:
                    print(f"  ✓ {r.label}  ({r.elapsed:.0f}s, {r.input_tokens:,}+{r.output_tokens:,} tokens)")
            else:
                failed += 1
                with _print_lock:
                    print(f"  ✗ {r.label}  — {r.error}")

    stop_status.set()
    status_thread.join()

    batch_elapsed = time.time() - batch_start
    print(f"\n{'═' * 60}")
    print(f"  Batch complete in {batch_elapsed:.0f}s")
    print(f"  {succeeded} succeeded, {failed} failed, {already_done} skipped")
    if total_in or total_out:
        print(f"  Total tokens: {total_in:,} in / {total_out:,} out")
    print(f"  Output: {output_dir}/")
    print(f"{'═' * 60}")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Format raw Whisper transcriptions into readable Markdown and Word documents using Claude.",
    )
    parser.add_argument(
        "transcript",
        help="Path to a transcription JSON file, or a directory to batch-process all unprocessed files",
    )
    parser.add_argument(
        "--context",
        help="Path to a show context JSON file with show_name, hosts, etc.",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for .md and .docx files (defaults to 'output/')",
        default=None,
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers for batch mode (default: 4)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Load show context
    if args.context:
        context_path = Path(args.context)
        if not context_path.exists():
            print(f"Error: Context file not found: {context_path}", file=sys.stderr)
            sys.exit(1)
        context = ShowContext.from_json_file(context_path)
        print(f"Show context: {context.show_name}")
    else:
        context = ShowContext()
        print("No show context provided, using empty context")

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("output")

    source = Path(args.transcript)

    if source.is_file():
        _run_single(source, context, output_dir)
    elif source.is_dir():
        _run_batch(source, context, output_dir, args.workers)
    else:
        print(f"Error: {source} is not a file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
