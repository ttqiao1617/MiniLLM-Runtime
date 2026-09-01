#!/usr/bin/env python3
"""Verify padding-aware Qwen batched decode against independent B=1 ORT runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from qwen_common import DEFAULT_ARTIFACT_ROOT


CONTROLLED_CASES = (
    Path("inputs/qwen_v09/perf_p8_ids.txt"),
    Path("inputs/qwen_v09/perf_p32_ids.txt"),
    Path("inputs/qwen_v09/perf_p128_ids.txt"),
    Path("inputs/qwen_v09/perf_p256_ids.txt"),
)


def parse_args() -> argparse.Namespace:
    root = DEFAULT_ARTIFACT_ROOT / "fp16_deployment"
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", type=Path, default=root / "qwen2_5_0_5b_prefill.onnx")
    parser.add_argument("--decode", type=Path, default=root / "qwen2_5_0_5b_decode.onnx")
    parser.add_argument(
        "--decode-batch", type=Path, default=root / "qwen2_5_0_5b_decode_batch.onnx"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-logits-abs", type=float, default=0.25)
    parser.add_argument("--max-logits-mean", type=float, default=0.006)
    parser.add_argument("--min-logits-cosine", type=float, default=0.999)
    parser.add_argument("--max-kv-abs", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qwen_batch_onnx_verify/verification.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def session(path: Path) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def output_map(
    runtime: ort.InferenceSession, values: list[np.ndarray]
) -> dict[str, np.ndarray]:
    return {item.name: value for item, value in zip(runtime.get_outputs(), values)}


def load_ids(path: Path) -> np.ndarray:
    values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not values:
        raise ValueError(f"empty token input: {path}")
    return np.asarray([values], dtype=np.int64)


def past_from_present(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.replace("present.", "past_key_values.", 1): value
        for name, value in outputs.items()
        if name.startswith("present.")
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64, copy=False).reshape(-1)
    b = right.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else float(a.size == 0)


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")

    prefill = session(args.prefill)
    seeds: list[dict[str, object]] = []
    for path in CONTROLLED_CASES:
        ids = load_ids(path)
        length = ids.shape[1]
        values = prefill.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids),
                "position_ids": np.arange(length, dtype=np.int64)[None, :],
            },
        )
        outputs = output_map(prefill, values)
        seeds.append(
            {
                "path": str(path),
                "length": length,
                "token": int(np.argmax(outputs["logits"][0, 0])),
                "cache": past_from_present(outputs),
            }
        )
    del prefill

    decode = session(args.decode)
    decode_batch = session(args.decode_batch)
    past_names = [item.name for item in decode_batch.get_inputs() if item.name.startswith("past_key_values.")]
    results: list[dict[str, object]] = []
    global_worst_logits_abs = 0.0
    global_worst_logits_mean = 0.0
    global_min_logits_cosine = 1.0
    global_worst_kv_abs = 0.0

    for batch_size in (1, 2, 4, 8):
        rows = [seeds[index % len(seeds)] for index in range(batch_size)]
        candidate_cache = [
            {name: value.copy() for name, value in row["cache"].items()}  # type: ignore[union-attr]
            for row in rows
        ]
        reference_cache = [
            {name: value.copy() for name, value in row["cache"].items()}  # type: ignore[union-attr]
            for row in rows
        ]
        tokens = [int(row["token"]) for row in rows]
        row_result: dict[str, object] = {
            "batch_size": batch_size,
            "initial_lengths": [int(row["length"]) for row in rows],
            "steps": [],
        }
        for step in range(args.steps):
            lengths = [next(iter(cache.values())).shape[2] for cache in candidate_cache]
            max_past = max(lengths)
            batch_inputs: dict[str, np.ndarray] = {
                "input_ids": np.asarray(tokens, dtype=np.int64).reshape(batch_size, 1),
                "position_ids": np.asarray(lengths, dtype=np.int64).reshape(batch_size, 1),
                "attention_mask": np.zeros((batch_size, max_past + 1), dtype=np.int64),
            }
            for row_index, length in enumerate(lengths):
                batch_inputs["attention_mask"][row_index, :length] = 1
                batch_inputs["attention_mask"][row_index, max_past] = 1
            for name in past_names:
                example = candidate_cache[0][name]
                packed = np.zeros(
                    (batch_size, example.shape[1], max_past, example.shape[3]),
                    dtype=example.dtype,
                )
                for row_index, length in enumerate(lengths):
                    packed[row_index, :, :length, :] = candidate_cache[row_index][name][0]
                batch_inputs[name] = packed
            batch_outputs = output_map(
                decode_batch, decode_batch.run(None, batch_inputs)
            )

            next_candidate_cache: list[dict[str, np.ndarray]] = []
            next_reference_cache: list[dict[str, np.ndarray]] = []
            next_tokens: list[int] = []
            step_logits_abs = 0.0
            step_logits_mean = 0.0
            step_logits_cosine = 1.0
            step_kv_abs = 0.0
            for row_index, length in enumerate(lengths):
                single_inputs: dict[str, np.ndarray] = {
                    "input_ids": np.asarray([[tokens[row_index]]], dtype=np.int64),
                    "position_ids": np.asarray([[length]], dtype=np.int64),
                    "attention_mask": np.ones((1, length + 1), dtype=np.int64),
                    **reference_cache[row_index],
                }
                reference_outputs = output_map(decode, decode.run(None, single_inputs))
                reference_logits = reference_outputs["logits"][0, 0].astype(np.float32)
                candidate_logits = batch_outputs["logits"][row_index, 0].astype(np.float32)
                difference = np.abs(reference_logits - candidate_logits)
                step_logits_abs = max(step_logits_abs, float(difference.max()))
                step_logits_mean = max(step_logits_mean, float(difference.mean()))
                step_logits_cosine = min(
                    step_logits_cosine, cosine(reference_logits, candidate_logits)
                )
                reference_top = int(np.argmax(reference_logits))
                candidate_top = int(np.argmax(candidate_logits))
                reference_top5 = set(np.argpartition(reference_logits, -5)[-5:].tolist())
                candidate_top5 = set(np.argpartition(candidate_logits, -5)[-5:].tolist())
                if reference_top != candidate_top or reference_top5 != candidate_top5:
                    raise RuntimeError(
                        f"batch={batch_size} step={step} row={row_index} top-k mismatch"
                    )
                compact: dict[str, np.ndarray] = {}
                for name in past_names:
                    present = name.replace("past_key_values.", "present.", 1)
                    reference_value = reference_outputs[present]
                    candidate_value = batch_outputs[present]
                    new_difference = np.abs(
                        reference_value[0, :, length, :].astype(np.float32)
                        - candidate_value[row_index, :, max_past, :].astype(np.float32)
                    )
                    step_kv_abs = max(step_kv_abs, float(new_difference.max()))
                    compact[name] = np.concatenate(
                        (
                            candidate_value[row_index : row_index + 1, :, :length, :],
                            candidate_value[
                                row_index : row_index + 1,
                                :,
                                max_past : max_past + 1,
                                :,
                            ],
                        ),
                        axis=2,
                    )
                next_candidate_cache.append(compact)
                next_reference_cache.append(past_from_present(reference_outputs))
                next_tokens.append(candidate_top)

            if step_logits_abs > args.max_logits_abs:
                raise RuntimeError(f"logits max abs gate failed: {step_logits_abs}")
            if step_logits_mean > args.max_logits_mean:
                raise RuntimeError(f"logits mean abs gate failed: {step_logits_mean}")
            if step_logits_cosine < args.min_logits_cosine:
                raise RuntimeError(f"logits cosine gate failed: {step_logits_cosine}")
            if step_kv_abs > args.max_kv_abs:
                raise RuntimeError(f"new-KV max abs gate failed: {step_kv_abs}")
            global_worst_logits_abs = max(global_worst_logits_abs, step_logits_abs)
            global_worst_logits_mean = max(global_worst_logits_mean, step_logits_mean)
            global_min_logits_cosine = min(global_min_logits_cosine, step_logits_cosine)
            global_worst_kv_abs = max(global_worst_kv_abs, step_kv_abs)
            row_result["steps"].append(  # type: ignore[union-attr]
                {
                    "step": step,
                    "past_lengths": lengths,
                    "max_past_length": max_past,
                    "padding_tokens": batch_size * max_past - sum(lengths),
                    "next_tokens": next_tokens,
                    "max_logits_abs": step_logits_abs,
                    "max_logits_mean": step_logits_mean,
                    "min_logits_cosine": step_logits_cosine,
                    "max_new_kv_abs": step_kv_abs,
                }
            )
            candidate_cache = next_candidate_cache
            reference_cache = next_reference_cache
            tokens = next_tokens
        results.append(row_result)
        print(f"B={batch_size}: {args.steps} padded decode steps passed")

    payload = {
        "schema_version": 1,
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "prefill": str(args.prefill.resolve()),
        "decode_reference": str(args.decode.resolve()),
        "decode_batch": str(args.decode_batch.resolve()),
        "batch_sizes": [1, 2, 4, 8],
        "worst": {
            "max_logits_abs": global_worst_logits_abs,
            "max_logits_mean": global_worst_logits_mean,
            "min_logits_cosine": global_min_logits_cosine,
            "max_new_kv_abs": global_worst_kv_abs,
        },
        "gates": {
            "max_logits_abs": args.max_logits_abs,
            "max_logits_mean": args.max_logits_mean,
            "min_logits_cosine": args.min_logits_cosine,
            "max_kv_abs": args.max_kv_abs,
            "top1_exact": True,
            "top5_set_exact": True,
        },
        "results": results,
        "status": "pass",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"verification: {args.output}")


if __name__ == "__main__":
    main()
