#!/usr/bin/env python3
"""Create deterministic real-chat and length-controlled Qwen v0.9 workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwen_common import DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION, build_chat, load_tokenizer


REAL_CASES = (
    (
        "kv_cache_en",
        "You are a helpful assistant.",
        "Briefly explain what a KV cache does in LLM inference.",
    ),
    (
        "gqa_zh",
        "你是一名简洁的 AI 基础设施助手。",
        "用两句话解释 GQA 为什么可以减少 KV Cache 显存。",
    ),
    (
        "scheduler_en",
        "You are a systems engineer.",
        "What is the difference between request scheduling and GPU continuous batching?",
    ),
    (
        "short_math",
        "Answer directly.",
        "What is 17 multiplied by 19?",
    ),
    (
        "eos_instruction",
        "Follow the requested output format exactly.",
        "Reply with only the word OK.",
    ),
)

CONTROLLED_LENGTHS = (8, 32, 128, 256)
CONTROLLED_SHORT_TEXT = "KV cache benchmark for TensorRT runtime correctness."
CONTROLLED_TEXT = (
    "TensorRT executes a compiled neural network engine while a language-model runtime "
    "owns requests, token positions, KV cache state, scheduling, memory capacity, and "
    "correctness checks. This deterministic paragraph is repeated only to create stable "
    "length-controlled performance inputs. "
)


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_ids(path: Path, values: list[int], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(value) for value in values) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/qwen_v09"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "workloads.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --force")

    cases: list[dict[str, object]] = []
    for case_id, system, user in REAL_CASES:
        rendered, encoded = build_chat(tokenizer, system, user)
        values = [int(value) for value in encoded["input_ids"][0].tolist()]
        path = args.output_dir / f"real_{case_id}_ids.txt"
        write_ids(path, values, args.force)
        cases.append(
            {
                "case_id": case_id,
                "kind": "real_chat",
                "system": system,
                "user": user,
                "rendered_chat": rendered,
                "prompt_length": len(values),
                "requested_new_tokens": 16,
                "input_ids_path": str(path),
                "input_ids_sha256": sha256_text(path),
            }
        )

    seed = tokenizer.encode(CONTROLLED_TEXT * 16, add_special_tokens=False)
    short_seed = tokenizer.encode(CONTROLLED_SHORT_TEXT, add_special_tokens=False)
    if len(short_seed) < 8:
        raise RuntimeError("controlled short text did not produce eight tokens")
    if len(seed) < max(CONTROLLED_LENGTHS):
        raise RuntimeError("controlled text did not produce enough tokens")
    for length in CONTROLLED_LENGTHS:
        source = short_seed if length == 8 else seed
        values = [int(value) for value in source[:length]]
        path = args.output_dir / f"perf_p{length}_ids.txt"
        write_ids(path, values, args.force)
        cases.append(
            {
                "case_id": f"perf_p{length}",
                "kind": "length_controlled_synthetic",
                "prompt_length": length,
                "requested_new_tokens": 32,
                "input_ids_path": str(path),
                "input_ids_sha256": sha256_text(path),
            }
        )

    payload = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(args.model.resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "controlled_lengths": list(CONTROLLED_LENGTHS),
        "cases": cases,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cases)} workloads: {manifest_path}")


if __name__ == "__main__":
    main()
