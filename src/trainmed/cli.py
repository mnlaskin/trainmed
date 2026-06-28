"""Command-line entry point for TrainMed transcript extraction.

Examples:
    python -m trainmed.cli "https://www.youtube.com/watch?v=VIDEO_ID"
    python -m trainmed.cli --from-file urls.txt --languages en,en-US
    python -m trainmed.cli --list-langs "https://youtu.be/VIDEO_ID"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import extract as extract_mod
from . import store
from .youtube import expand_targets


def _read_url_file(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # Allow comments and blank lines.
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trainmed",
        description="Extract YouTube transcripts into the TrainMed knowledge base.",
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="Video / playlist / channel URLs or bare video ids.",
    )
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="Read URLs (one per line; # comments allowed) from a file.",
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated preferred caption languages (default: en).",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        help="Output directory (default: data/transcripts/).",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip yt-dlp metadata fetch (faster, captions only).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Write only JSON, not the markdown rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract videos even if already saved.",
    )
    parser.add_argument(
        "--list-langs",
        action="store_true",
        help="List available caption tracks for each video and exit.",
    )
    args = parser.parse_args(argv)

    inputs = list(args.urls)
    if args.from_file:
        inputs.extend(_read_url_file(args.from_file))
    if not inputs:
        parser.error("provide at least one URL or --from-file")

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.list_langs:
        return _run_list_langs(inputs)

    print(f"Resolving {len(inputs)} input(s)...", file=sys.stderr)
    targets = expand_targets(inputs)
    if not targets:
        print("No videos resolved from the given inputs.", file=sys.stderr)
        return 1
    print(f"Found {len(targets)} video(s).\n", file=sys.stderr)

    ok = skipped = failed = 0
    for i, (video_id, url) in enumerate(targets, 1):
        prefix = f"[{i}/{len(targets)}] {video_id}"
        if not args.overwrite and store.already_saved(video_id, out_dir):
            print(f"{prefix}  · already saved, skipping")
            skipped += 1
            continue
        try:
            t = extract_mod.extract(
                video_id,
                languages=languages,
                with_metadata=not args.no_metadata,
            )
        except extract_mod.TranscriptUnavailable as exc:
            print(f"{prefix}  ✗ no transcript ({exc})")
            failed += 1
            continue
        except Exception as exc:  # keep the batch going
            print(f"{prefix}  ✗ error ({exc})")
            failed += 1
            continue

        store.save(t, out_dir, write_markdown=not args.no_markdown)
        words = len(t.text().split())
        title = (t.title or "")[:60]
        print(f"{prefix}  ✓ {words:>6,} words  {title}")
        ok += 1

    print(
        f"\nDone. {ok} saved, {skipped} skipped, {failed} failed.",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 2


def _run_list_langs(inputs: list[str]) -> int:
    from .youtube import parse_video_id

    for value in inputs:
        vid = parse_video_id(value)
        if not vid:
            print(f"{value}: not a single-video URL/id (skipping for --list-langs)")
            continue
        print(f"\n{vid}:")
        try:
            tracks = extract_mod.list_languages(vid)
        except Exception as exc:
            print(f"  error: {exc}")
            continue
        if not tracks:
            print("  (no caption tracks)")
        for tr in tracks:
            kind = "auto" if tr["is_generated"] else "manual"
            xl = " translatable" if tr["is_translatable"] else ""
            print(f"  {tr['language_code']:<8} {tr['language']}  [{kind}]{xl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
