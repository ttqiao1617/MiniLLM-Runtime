#!/usr/bin/env python3
"""Download the pinned Qwen2.5-0.5B-Instruct snapshot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import snapshot_download

from qwen_common import DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--allow-unpinned-source",
        action="store_true",
        help="Allow a model/revision other than the audited pinned Qwen commit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pinned = args.model_id == MODEL_ID and args.revision == MODEL_REVISION
    commit_pinned = re.fullmatch(r"[0-9a-f]{40}", args.revision) is not None
    if (not pinned or not commit_pinned) and not args.allow_unpinned_source:
        raise ValueError(
            "refusing an unaudited model source; use the default model/revision or "
            "pass --allow-unpinned-source after reviewing the repository"
        )
    path = Path(snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "LICENSE",
            "README.md",
        ],
    ))
    required = ["config.json", "generation_config.json", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete snapshot, missing: {missing}")
    lock = {
        "model_id": args.model_id,
        "revision": args.revision,
        "path": str(path.resolve()),
    }
    (path / "minillm_snapshot.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
