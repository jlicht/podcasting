#!/usr/bin/env python3
"""CLI tool to format a raw Whisper transcription using Claude.

Usage:
    uv run python format_transcript.py transcriptions/<job_id>.json --context show_context.json
"""

import argparse
import os
import sys
from pathlib import Path

from formatter import (
    ShowContext,
    TranscriptData,
    format_transcript,
    save_formatted_output,
)


def main():
    parser = argparse.ArgumentParser(
        description="Format a raw Whisper transcription into readable Markdown and Word documents using Claude.",
    )
    parser.add_argument(
        "transcript",
        help="Path to the transcription JSON file (e.g. transcriptions/<job_id>.json)",
    )
    parser.add_argument(
        "--context",
        help="Path to a show context JSON file with show_name, hosts, etc.",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for .md and .docx files (defaults to same directory as input)",
        default=None,
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Load transcript
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    transcript = TranscriptData.from_json_file(transcript_path)
    print(f"Loaded transcript: {transcript.filename} ({len(transcript.text)} chars)")

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
    output_dir = Path(args.output_dir) if args.output_dir else transcript_path.parent

    # Format
    print(f"Formatting with Claude ({transcript.job_id})...")
    try:
        result = format_transcript(transcript, context)
    except Exception as e:
        print(f"Error: Formatting failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Tokens used: {result.input_tokens} in, {result.output_tokens} out")

    # Save
    md_path, docx_path = save_formatted_output(result, output_dir, transcript.job_id)
    print(f"Saved: {md_path}")
    print(f"Saved: {docx_path}")


if __name__ == "__main__":
    main()
