#!/usr/bin/env python3
"""Download pretrained AIR checkpoints from Google Drive."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1ALBA2Mw9UchimlPNgA-m_FjT4-U364Cb?usp=sharing"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained AIR checkpoints from Google Drive."
    )
    parser.add_argument(
        "--url",
        default=GOOGLE_DRIVE_FOLDER_URL,
        help="Google Drive folder URL containing checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where checkpoint files will be saved.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce download progress output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gdown = importlib.import_module("gdown")
    except ModuleNotFoundError:
        print(
            "Missing dependency: gdown. Install it with `pip install -r requirements.txt` "
            "or `pip install gdown`.",
            file=sys.stderr,
        )
        return 1

    downloaded_files = gdown.download_folder(
        url=args.url,
        output=str(output_dir),
        quiet=args.quiet,
        use_cookies=False,
    )

    if not downloaded_files:
        print("No checkpoint files were downloaded.", file=sys.stderr)
        return 1

    print(f"Downloaded {len(downloaded_files)} file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
