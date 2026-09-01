#!/usr/bin/env python3
"""Generate cache/no-cache Hugging Face goldens for a workload manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_common import DEFAULT_MODEL_DIR, eos_token_ids, load_model, load_tokenizer
from qwen_hf_golden import greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--manifest", type=Path, default=Path("inputs/qwen_v09/workloads.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("inputs/qwen_v09/goldens")
    )
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--kind",
        action="append",
        choices=("real_chat", "length_controlled_synthetic"),
        help="Restrict case kinds; repeat to select both.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dtype == "float16" and args.device == "cpu":
        raise ValueError("float16 baseline requires --device cuda")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if not args.kind or case["kind"] in args.kind]
    if not cases:
        raise ValueError("no workload cases selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        output = args.output_dir / f"{case['case_id']}.json"
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {output}; pass --force")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.dtype, args.device)
    stop_ids = set(eos_token_ids(args.model))
    summary: list[dict[str, object]] = []
    for case in cases:
        ids_path = Path(case["input_ids_path"])
        values = [int(value) for value in ids_path.read_text(encoding="utf-8").split()]
        input_ids = torch.tensor([values], dtype=torch.int64, device=args.device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(values), dtype=torch.int64, device=args.device).unsqueeze(0)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        requested = int(case["requested_new_tokens"])
        generated, cached_ids, step_top5 = greedy_decode(
            model, model_inputs, requested, stop_ids, use_cache=True
        )
        _, recompute_ids, _ = greedy_decode(
            model, model_inputs, requested, stop_ids, use_cache=False
        )
        if cached_ids != recompute_ids:
            raise RuntimeError(
                f"HF cache/no-cache mismatch for {case['case_id']}: "
                f"cache={cached_ids}, no-cache={recompute_ids}"
            )
        payload = {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "model_id": manifest["model_id"],
            "revision": manifest["model_revision"],
            "model_path": str(args.model.resolve()),
            "dtype": args.dtype,
            "device": args.device,
            "input_ids_path": str(ids_path),
            "input_ids": values,
            "attention_mask": [1] * len(values),
            "position_ids": list(range(len(values))),
            "prompt_length": len(values),
            "requested_new_tokens": requested,
            "actual_new_tokens": len(cached_ids),
            "generated_ids": cached_ids,
            "no_cache_generated_ids": recompute_ids,
            "cache_vs_no_cache_match": True,
            "stopped_on_eos": bool(cached_ids and cached_ids[-1] in stop_ids),
            "eos_token_ids": sorted(stop_ids),
            "generated_text": tokenizer.decode(cached_ids, skip_special_tokens=False),
            "full_text": tokenizer.decode(generated[0].tolist(), skip_special_tokens=False),
            "step_top5": step_top5,
        }
        output = args.output_dir / f"{case['case_id']}.json"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append(
            {
                "case_id": case["case_id"],
                "prompt_length": len(values),
                "requested_new_tokens": requested,
                "actual_new_tokens": len(cached_ids),
                "stopped_on_eos": payload["stopped_on_eos"],
                "golden_path": str(output),
            }
        )
        print(
            f"{case['case_id']}: prompt={len(values)} generated={len(cached_ids)} "
            f"eos={payload['stopped_on_eos']}"
        )

    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {summary_path}; pass --force")
    summary_path.write_text(
        json.dumps({"schema_version": 1, "cases": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
