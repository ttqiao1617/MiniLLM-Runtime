#!/usr/bin/env python3
"""Verify the FP16 Qwen ONNX contracts against Hugging Face.

The verifier deliberately uses two independent execution paths:

* a local Hugging Face model in FP16 (CUDA by default), and
* ONNX Runtime's CPUExecutionProvider for the exported FP16 graphs.

It checks the logits-only, prefill, and with-past graphs, every exported KV
tensor, and an end-to-end cached greedy prefix against the checked-in HF
golden for each selected workload.  A report is always written after the
numeric checks; any contract, tolerance, top-k, or token mismatch makes the
process exit non-zero.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from qwen_common import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    MODEL_REVISION,
    load_config,
    load_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "inputs/qwen_v09/workloads.json"
DEFAULT_GOLDEN_DIR = REPO_ROOT / "inputs/qwen_v09/goldens"
DEFAULT_ONNX_DIR = DEFAULT_ARTIFACT_ROOT / "fp16_deployment"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results/qwen_onnx_verify/verification.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "results/qwen_onnx_verify/verification.csv"


@dataclass
class CaseReference:
    case_id: str
    kind: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    golden_ids: list[int]
    golden_path: Path
    hf_logits_only: np.ndarray
    hf_prefill_logits: np.ndarray
    hf_prefill_cache: dict[str, np.ndarray]
    hf_decode_logits: np.ndarray
    hf_decode_cache: dict[str, np.ndarray]
    decode_input_id: int
    ort_prefill_logits: np.ndarray | None = None
    ort_prefill_cache: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def prompt_length(self) -> int:
        return int(self.input_ids.shape[1])

    def common_inputs(self) -> dict[str, np.ndarray]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen2.5 FP16 logits/prefill/decode ONNX graphs with a "
            "local HF FP16 reference and verify cached greedy tokens."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="CASE_ID",
        help="Verify one case ID; repeat to select multiple cases (default: all).",
    )
    parser.add_argument(
        "--kind",
        action="append",
        choices=("real_chat", "length_controlled_synthetic"),
        help="Restrict workload kind; repeat to select both.",
    )
    parser.add_argument(
        "--greedy-tokens",
        type=int,
        default=8,
        help="Maximum HF-golden tokens checked per case (EOS-shortened goldens stay shorter).",
    )
    parser.add_argument(
        "--hf-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for the FP16 Hugging Face reference.",
    )
    parser.add_argument("--ort-intra-op-threads", type=int, default=0)
    parser.add_argument("--ort-inter-op-threads", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing reports."
    )

    tolerance = parser.add_argument_group("correctness gates")
    tolerance.add_argument("--logits-max-abs", type=float, default=0.25)
    tolerance.add_argument("--logits-mean-abs", type=float, default=0.03)
    tolerance.add_argument("--logits-min-cosine", type=float, default=0.999)
    tolerance.add_argument("--kv-max-abs", type=float, default=0.25)
    tolerance.add_argument("--kv-mean-abs", type=float, default=0.006)
    tolerance.add_argument("--kv-min-cosine", type=float, default=0.999)
    return parser.parse_args()


def resolve_input_path(raw: str | Path, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        (Path.cwd() / path).resolve(),
        (REPO_ROOT / path).resolve(),
        (manifest_path.parent / path).resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_token_ids(path: Path) -> list[int]:
    values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if not values:
        raise ValueError(f"empty token-ID workload: {path}")
    return values


def load_selected_cases(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise RuntimeError(
            f"workload model mismatch: {manifest.get('model_id')!r} != {MODEL_ID!r}"
        )
    if manifest.get("model_revision") != MODEL_REVISION:
        raise RuntimeError(
            "workload revision mismatch: "
            f"{manifest.get('model_revision')!r} != {MODEL_REVISION!r}"
        )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RuntimeError("workload manifest has no cases")

    by_id: dict[str, dict[str, Any]] = {}
    for raw_case in raw_cases:
        case_id = str(raw_case.get("case_id", ""))
        if not case_id or case_id in by_id:
            raise RuntimeError(f"invalid or duplicate workload case_id: {case_id!r}")
        by_id[case_id] = raw_case

    requested = set(args.case_ids or by_id)
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"unknown --case values: {', '.join(missing)}")
    selected = [case for case in raw_cases if case["case_id"] in requested]
    if args.kind:
        allowed_kinds = set(args.kind)
        selected = [case for case in selected if case.get("kind") in allowed_kinds]
    if not selected:
        raise ValueError("case filters selected no workloads")

    for case in selected:
        ids_path = resolve_input_path(case["input_ids_path"], manifest_path)
        if not ids_path.is_file():
            raise FileNotFoundError(
                f"missing workload IDs for {case['case_id']}: {ids_path}"
            )
        values = read_token_ids(ids_path)
        expected_length = int(case["prompt_length"])
        if len(values) != expected_length:
            raise RuntimeError(
                f"workload length mismatch for {case['case_id']}: "
                f"file={len(values)}, manifest={expected_length}"
            )
        expected_sha = case.get("input_ids_sha256")
        if expected_sha and sha256_file(ids_path) != expected_sha:
            raise RuntimeError(
                f"workload SHA-256 mismatch for {case['case_id']}: {ids_path}"
            )
        case["_ids_path"] = ids_path
        case["_input_ids"] = values
    return manifest, selected


def cache_names(prefix: str, num_layers: int) -> list[str]:
    return [
        f"{prefix}{layer}.{kind}"
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # Keep the reference dtype visible in the report.  Metric functions cast
    # to wider types before subtraction/accumulation.
    return tensor.detach().cpu().numpy()


def flatten_hf_cache(cache: Any, num_layers: int) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}
    layers = list(cache)
    if len(layers) != num_layers:
        raise RuntimeError(
            f"HF returned {len(layers)} cache layers; expected {num_layers}"
        )
    for layer, pair in enumerate(layers):
        if len(pair) != 2:
            raise RuntimeError(f"HF cache layer {layer} is not a key/value pair")
        flattened[f"present.{layer}.key"] = tensor_to_numpy(pair[0])
        flattened[f"present.{layer}.value"] = tensor_to_numpy(pair[1])
    return flattened


def load_golden(
    case: dict[str, Any], args: argparse.Namespace
) -> tuple[Path, dict[str, Any]]:
    golden_path = (args.golden_dir / f"{case['case_id']}.json").resolve()
    if not golden_path.is_file():
        raise FileNotFoundError(
            f"missing HF golden for {case['case_id']}: {golden_path}"
        )
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if golden.get("case_id") != case["case_id"]:
        raise RuntimeError(f"golden case ID mismatch: {golden_path}")
    if golden.get("revision") != MODEL_REVISION:
        raise RuntimeError(f"golden model revision mismatch: {golden_path}")
    if list(golden.get("input_ids", [])) != case["_input_ids"]:
        raise RuntimeError(f"golden input IDs differ from workload: {golden_path}")
    generated = [int(value) for value in golden.get("generated_ids", [])]
    if not generated:
        raise RuntimeError(f"golden has no generated tokens: {golden_path}")
    no_cache = [int(value) for value in golden.get("no_cache_generated_ids", [])]
    if no_cache and generated != no_cache:
        raise RuntimeError(f"HF cache/no-cache golden mismatch: {golden_path}")
    return golden_path, golden


def build_hf_references(
    model: Any,
    cases: Iterable[dict[str, Any]],
    num_layers: int,
    args: argparse.Namespace,
) -> list[CaseReference]:
    references: list[CaseReference] = []
    device = torch.device(args.hf_device)
    with torch.inference_mode():
        for case in cases:
            golden_path, golden = load_golden(case, args)
            ids = torch.tensor([case["_input_ids"]], dtype=torch.int64, device=device)
            mask = torch.ones_like(ids)
            positions = torch.arange(
                ids.shape[1], dtype=torch.int64, device=device
            ).unsqueeze(0)

            logits_only_output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            )
            prefill_output = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
                return_dict=True,
            )
            hf_logits_only = tensor_to_numpy(logits_only_output.logits[:, -1:, :])
            hf_prefill_logits = tensor_to_numpy(prefill_output.logits[:, -1:, :])
            hf_prefill_cache = flatten_hf_cache(
                prefill_output.past_key_values, num_layers
            )
            decode_input_id = int(np.argmax(hf_prefill_logits.reshape(-1)))

            decode_ids = torch.tensor(
                [[decode_input_id]], dtype=torch.int64, device=device
            )
            decode_mask = torch.ones(
                (1, ids.shape[1] + 1), dtype=torch.int64, device=device
            )
            decode_positions = torch.tensor(
                [[ids.shape[1]]], dtype=torch.int64, device=device
            )
            decode_output = model(
                input_ids=decode_ids,
                attention_mask=decode_mask,
                position_ids=decode_positions,
                past_key_values=prefill_output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            hf_decode_logits = tensor_to_numpy(decode_output.logits[:, -1:, :])
            hf_decode_cache = flatten_hf_cache(
                decode_output.past_key_values, num_layers
            )

            references.append(
                CaseReference(
                    case_id=case["case_id"],
                    kind=case["kind"],
                    input_ids=ids.cpu().numpy(),
                    attention_mask=mask.cpu().numpy(),
                    position_ids=positions.cpu().numpy(),
                    golden_ids=[int(value) for value in golden["generated_ids"]],
                    golden_path=golden_path,
                    hf_logits_only=hf_logits_only,
                    hf_prefill_logits=hf_prefill_logits,
                    hf_prefill_cache=hf_prefill_cache,
                    hf_decode_logits=hf_decode_logits,
                    hf_decode_cache=hf_decode_cache,
                    decode_input_id=decode_input_id,
                )
            )
            del logits_only_output, prefill_output, decode_output
    return references


def topk_ids(values: np.ndarray, k: int = 5) -> list[int]:
    flat = np.asarray(values).reshape(-1)
    count = min(k, flat.size)
    candidates = np.argpartition(-flat, count - 1)[:count]
    ordered = sorted(
        (int(index) for index in candidates), key=lambda index: (-flat[index], index)
    )
    return ordered


def cosine_similarity(expected: np.ndarray, actual: np.ndarray) -> float:
    left = np.asarray(expected, dtype=np.float64).reshape(-1)
    right = np.asarray(actual, dtype=np.float64).reshape(-1)
    left_norm = float(np.dot(left, left))
    right_norm = float(np.dot(right, right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / math.sqrt(left_norm * right_norm))


def numeric_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected_shape = list(expected.shape)
    actual_shape = list(actual.shape)
    result: dict[str, Any] = {
        "expected_shape": expected_shape,
        "actual_shape": actual_shape,
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "shape_match": expected_shape == actual_shape,
    }
    if expected_shape != actual_shape:
        result.update({"max_abs": None, "mean_abs": None, "cosine": None})
        return result
    left = np.asarray(expected, dtype=np.float32)
    right = np.asarray(actual, dtype=np.float32)
    absolute = np.abs(left - right)
    result.update(
        {
            "max_abs": float(np.max(absolute)) if absolute.size else 0.0,
            "mean_abs": (
                float(np.mean(absolute, dtype=np.float64)) if absolute.size else 0.0
            ),
            "cosine": cosine_similarity(left, right),
        }
    )
    return result


def compare_logits(
    expected: np.ndarray,
    actual: np.ndarray,
    graph: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    result = numeric_metrics(expected, actual)
    failures: list[str] = []
    if not result["shape_match"]:
        failures.append(
            f"{graph} logits shape {result['actual_shape']} != {result['expected_shape']}"
        )
        result["passed"] = False
        return result, failures

    expected_top5 = topk_ids(expected)
    actual_top5 = topk_ids(actual)
    expected_flat = expected.reshape(-1)
    actual_flat = actual.reshape(-1)
    expected_margin = float(
        expected_flat[expected_top5[0]] - expected_flat[expected_top5[1]]
    )
    actual_margin = float(actual_flat[actual_top5[0]] - actual_flat[actual_top5[1]])
    result.update(
        {
            "reference_top1": expected_top5[0],
            "actual_top1": actual_top5[0],
            "top1_match": expected_top5[0] == actual_top5[0],
            "reference_top5": expected_top5,
            "actual_top5": actual_top5,
            "top5_exact_match": expected_top5 == actual_top5,
            "top5_set_match": set(expected_top5) == set(actual_top5),
            "top5_overlap": len(set(expected_top5) & set(actual_top5)),
            "reference_top1_margin": expected_margin,
            "actual_top1_margin": actual_margin,
        }
    )
    if result["max_abs"] > args.logits_max_abs:
        failures.append(
            f"{graph} logits max_abs={result['max_abs']:.8g} > {args.logits_max_abs}"
        )
    if result["mean_abs"] > args.logits_mean_abs:
        failures.append(
            f"{graph} logits mean_abs={result['mean_abs']:.8g} > {args.logits_mean_abs}"
        )
    if result["cosine"] < args.logits_min_cosine:
        failures.append(
            f"{graph} logits cosine={result['cosine']:.10g} < {args.logits_min_cosine}"
        )
    if not result["top1_match"]:
        failures.append(
            f"{graph} top1 mismatch: actual={actual_top5[0]}, reference={expected_top5[0]}"
        )
    if not result["top5_set_match"]:
        failures.append(
            f"{graph} top5 set mismatch: actual={actual_top5}, reference={expected_top5}"
        )
    result["passed"] = not failures
    return result, failures


def compare_cache(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    stage: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str]]:
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    failures: list[str] = []
    if missing:
        failures.append(f"{stage} missing KV outputs: {missing}")
    if unexpected:
        failures.append(f"{stage} unexpected KV outputs: {unexpected}")

    tensors: list[dict[str, Any]] = []
    total_elements = 0
    total_abs = 0.0
    dot = 0.0
    expected_squared = 0.0
    actual_squared = 0.0
    max_abs = 0.0
    all_shapes_match = not missing and not unexpected
    for name in sorted(expected_names & actual_names):
        metric = numeric_metrics(expected[name], actual[name])
        metric["name"] = name
        tensor_failures: list[str] = []
        if not metric["shape_match"]:
            all_shapes_match = False
            tensor_failures.append(
                f"shape {metric['actual_shape']} != {metric['expected_shape']}"
            )
        else:
            left = np.asarray(expected[name], dtype=np.float64).reshape(-1)
            right = np.asarray(actual[name], dtype=np.float64).reshape(-1)
            absolute = np.abs(left - right)
            total_elements += int(absolute.size)
            total_abs += float(np.sum(absolute, dtype=np.float64))
            max_abs = max(max_abs, float(metric["max_abs"]))
            dot += float(np.dot(left, right))
            expected_squared += float(np.dot(left, left))
            actual_squared += float(np.dot(right, right))
        metric["passed"] = not tensor_failures
        metric["failures"] = tensor_failures
        tensors.append(metric)

    if expected_squared == 0.0 and actual_squared == 0.0:
        aggregate_cosine = 1.0
    elif expected_squared == 0.0 or actual_squared == 0.0:
        aggregate_cosine = 0.0
    else:
        aggregate_cosine = dot / math.sqrt(expected_squared * actual_squared)
    aggregate_mean = total_abs / total_elements if total_elements else 0.0
    if not all_shapes_match:
        failures.append(f"{stage} has one or more KV shape mismatches")
    if max_abs > args.kv_max_abs:
        failures.append(f"{stage} KV max_abs={max_abs:.8g} > {args.kv_max_abs}")
    if aggregate_mean > args.kv_mean_abs:
        failures.append(
            f"{stage} KV mean_abs={aggregate_mean:.8g} > {args.kv_mean_abs}"
        )
    if aggregate_cosine < args.kv_min_cosine:
        failures.append(
            f"{stage} KV cosine={aggregate_cosine:.10g} < {args.kv_min_cosine}"
        )
    result = {
        "expected_tensor_count": len(expected),
        "actual_tensor_count": len(actual),
        "compared_tensor_count": len(tensors),
        "missing": missing,
        "unexpected": unexpected,
        "all_shapes_match": all_shapes_match,
        "total_elements": total_elements,
        "max_abs": max_abs,
        "mean_abs": aggregate_mean,
        "cosine": aggregate_cosine,
        "tensors": tensors,
        "passed": not failures,
    }
    return result, failures


def create_ort_session(path: Path, args: argparse.Namespace):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required; use the project .venv or install an ORT CPU build"
        ) from exc
    options = ort.SessionOptions()
    if args.ort_intra_op_threads > 0:
        options.intra_op_num_threads = args.ort_intra_op_threads
    if args.ort_inter_op_threads > 0:
        options.inter_op_num_threads = args.ort_inter_op_threads
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(
            f"ORT did not select CPUExecutionProvider: {session.get_providers()}"
        )
    return session, ort.__version__


def validate_session_contract(session: Any, mode: str, num_layers: int) -> None:
    common = {"input_ids", "attention_mask", "position_ids"}
    expected_inputs = set(common)
    expected_outputs = {"logits"}
    if mode == "prefill":
        expected_outputs.update(cache_names("present.", num_layers))
    elif mode == "decode":
        expected_inputs.update(cache_names("past_key_values.", num_layers))
        expected_outputs.update(cache_names("present.", num_layers))
    actual_inputs = {item.name for item in session.get_inputs()}
    actual_outputs = {item.name for item in session.get_outputs()}
    if actual_inputs != expected_inputs:
        raise RuntimeError(
            f"{mode} ONNX input contract mismatch: "
            f"missing={sorted(expected_inputs - actual_inputs)}, "
            f"unexpected={sorted(actual_inputs - expected_inputs)}"
        )
    if actual_outputs != expected_outputs:
        raise RuntimeError(
            f"{mode} ONNX output contract mismatch: "
            f"missing={sorted(expected_outputs - actual_outputs)}, "
            f"unexpected={sorted(actual_outputs - expected_outputs)}"
        )
    for item in session.get_inputs():
        expected_type = "tensor(int64)" if item.name in common else "tensor(float16)"
        if item.type != expected_type:
            raise RuntimeError(
                f"{mode} input {item.name} has {item.type}; expected {expected_type}"
            )
    for item in session.get_outputs():
        if item.type != "tensor(float16)":
            raise RuntimeError(
                f"{mode} output {item.name} has {item.type}; expected tensor(float16)"
            )


def output_map(session: Any, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        descriptor.name: value
        for descriptor, value in zip(session.get_outputs(), outputs)
    }


def add_failures(
    case_result: dict[str, Any], stage: str, failures: Iterable[str]
) -> None:
    for failure in failures:
        case_result["failures"].append(f"{stage}: {failure}")


def validate_cache_lengths(
    cache: dict[str, np.ndarray], expected_length: int, label: str
) -> tuple[list[int], list[str]]:
    lengths: list[int] = []
    failures: list[str] = []
    for name, value in sorted(cache.items()):
        if value.ndim != 4:
            failures.append(f"{label} {name} rank={value.ndim}; expected rank 4")
            continue
        length = int(value.shape[2])
        lengths.append(length)
        if length != expected_length:
            failures.append(
                f"{label} {name} sequence={length}; expected {expected_length}"
            )
    return sorted(set(lengths)), failures


def verify_logits_graph(
    references: list[CaseReference],
    case_results: dict[str, dict[str, Any]],
    path: Path,
    num_layers: int,
    args: argparse.Namespace,
) -> str:
    session, version = create_ort_session(path, args)
    validate_session_contract(session, "logits", num_layers)
    for reference in references:
        outputs = output_map(session, session.run(None, reference.common_inputs()))
        metrics, failures = compare_logits(
            reference.hf_logits_only, outputs["logits"], "logits", args
        )
        case_results[reference.case_id]["graphs"]["logits"] = metrics
        add_failures(case_results[reference.case_id], "logits", failures)
    del session
    gc.collect()
    return version


def verify_prefill_graph(
    references: list[CaseReference],
    case_results: dict[str, dict[str, Any]],
    path: Path,
    num_layers: int,
    args: argparse.Namespace,
) -> str:
    session, version = create_ort_session(path, args)
    validate_session_contract(session, "prefill", num_layers)
    for reference in references:
        outputs = output_map(session, session.run(None, reference.common_inputs()))
        metrics, failures = compare_logits(
            reference.hf_prefill_logits, outputs["logits"], "prefill", args
        )
        result = case_results[reference.case_id]
        result["graphs"]["prefill"] = metrics
        add_failures(result, "prefill", failures)
        actual_cache = {
            name: value
            for name, value in outputs.items()
            if name.startswith("present.")
        }
        kv_metrics, kv_failures = compare_cache(
            reference.hf_prefill_cache, actual_cache, "prefill", args
        )
        result["kv"]["prefill"] = kv_metrics
        add_failures(result, "prefill KV", kv_failures)
        lengths, length_failures = validate_cache_lengths(
            actual_cache, reference.prompt_length, "prefill output"
        )
        result["kv"]["prefill"]["sequence_lengths"] = lengths
        add_failures(result, "prefill KV", length_failures)
        reference.ort_prefill_logits = outputs["logits"]
        reference.ort_prefill_cache = actual_cache
    del session
    gc.collect()
    return version


def make_decode_inputs(
    token_id: int,
    total_length: int,
    cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {
        "input_ids": np.asarray([[token_id]], dtype=np.int64),
        "attention_mask": np.ones((1, total_length), dtype=np.int64),
        "position_ids": np.asarray([[total_length - 1]], dtype=np.int64),
    }
    for present_name, value in cache.items():
        inputs[present_name.replace("present.", "past_key_values.", 1)] = value
    return inputs


def verify_decode_and_greedy(
    references: list[CaseReference],
    case_results: dict[str, dict[str, Any]],
    path: Path,
    num_layers: int,
    args: argparse.Namespace,
) -> str:
    session, version = create_ort_session(path, args)
    validate_session_contract(session, "decode", num_layers)
    for reference in references:
        result = case_results[reference.case_id]
        first_inputs = make_decode_inputs(
            reference.decode_input_id,
            reference.prompt_length + 1,
            reference.ort_prefill_cache,
        )
        outputs = output_map(session, session.run(None, first_inputs))
        metrics, failures = compare_logits(
            reference.hf_decode_logits, outputs["logits"], "decode", args
        )
        result["graphs"]["decode"] = metrics
        add_failures(result, "decode", failures)
        actual_cache = {
            name: value
            for name, value in outputs.items()
            if name.startswith("present.")
        }
        kv_metrics, kv_failures = compare_cache(
            reference.hf_decode_cache, actual_cache, "decode", args
        )
        result["kv"]["decode"] = kv_metrics
        add_failures(result, "decode KV", kv_failures)
        lengths, length_failures = validate_cache_lengths(
            actual_cache, reference.prompt_length + 1, "decode output"
        )
        result["kv"]["decode"]["sequence_lengths"] = lengths
        add_failures(result, "decode KV", length_failures)

        target_count = min(args.greedy_tokens, len(reference.golden_ids))
        expected_ids = reference.golden_ids[:target_count]
        if reference.ort_prefill_logits is None:
            raise RuntimeError(
                "internal error: prefill logits unavailable for greedy decode"
            )
        generated = [int(np.argmax(reference.ort_prefill_logits.reshape(-1)))]
        greedy_cache = reference.ort_prefill_cache
        steps: list[dict[str, Any]] = [
            {
                "step": 0,
                "source": "prefill",
                "token_id": generated[0],
                "cache_sequence_length": reference.prompt_length,
            }
        ]
        while len(generated) < target_count:
            expected_past = reference.prompt_length + len(generated) - 1
            past_lengths, past_failures = validate_cache_lengths(
                greedy_cache, expected_past, f"greedy step {len(generated)} input"
            )
            add_failures(result, "greedy KV", past_failures)
            total_length = reference.prompt_length + len(generated)
            decode_inputs = make_decode_inputs(
                generated[-1], total_length, greedy_cache
            )
            step_outputs = output_map(session, session.run(None, decode_inputs))
            greedy_cache = {
                name: value
                for name, value in step_outputs.items()
                if name.startswith("present.")
            }
            output_lengths, output_failures = validate_cache_lengths(
                greedy_cache, total_length, f"greedy step {len(generated)} output"
            )
            add_failures(result, "greedy KV", output_failures)
            next_id = int(np.argmax(step_outputs["logits"].reshape(-1)))
            generated.append(next_id)
            steps.append(
                {
                    "step": len(generated) - 1,
                    "source": "decode",
                    "token_id": next_id,
                    "attention_mask_length": total_length,
                    "position_id": total_length - 1,
                    "input_cache_sequence_lengths": past_lengths,
                    "output_cache_sequence_lengths": output_lengths,
                }
            )

        generated = generated[:target_count]
        exact_match = generated == expected_ids
        first_mismatch: int | None = None
        if not exact_match:
            for index, (actual_id, expected_id) in enumerate(
                zip(generated, expected_ids)
            ):
                if actual_id != expected_id:
                    first_mismatch = index
                    break
            if first_mismatch is None and len(generated) != len(expected_ids):
                first_mismatch = min(len(generated), len(expected_ids))
            result["failures"].append(
                "greedy: token mismatch at step "
                f"{first_mismatch}: actual={generated}, HF golden={expected_ids}"
            )
        result["greedy"] = {
            "requested_tokens": args.greedy_tokens,
            "checked_tokens": target_count,
            "golden_available_tokens": len(reference.golden_ids),
            "golden_shortened_by_eos": len(reference.golden_ids) < args.greedy_tokens,
            "expected_ids": expected_ids,
            "actual_ids": generated,
            "exact_match": exact_match,
            "first_mismatch_step": first_mismatch,
            "steps": steps,
            "passed": exact_match
            and not any("greedy KV" in item for item in result["failures"]),
        }
    del session
    gc.collect()
    return version


def csv_row(case: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "prompt_length": case["prompt_length"],
        "passed": case["passed"],
        "failure_count": len(case["failures"]),
        "failures": " | ".join(case["failures"]),
    }
    for graph in ("logits", "prefill", "decode"):
        metrics = case["graphs"].get(graph, {})
        for key in (
            "max_abs",
            "mean_abs",
            "cosine",
            "reference_top1",
            "actual_top1",
            "top1_match",
            "top5_exact_match",
            "top5_set_match",
            "top5_overlap",
            "reference_top1_margin",
        ):
            row[f"{graph}_{key}"] = metrics.get(key)
        row[f"{graph}_reference_top5"] = json.dumps(metrics.get("reference_top5"))
        row[f"{graph}_actual_top5"] = json.dumps(metrics.get("actual_top5"))
    for stage in ("prefill", "decode"):
        metrics = case["kv"].get(stage, {})
        for key in (
            "expected_tensor_count",
            "actual_tensor_count",
            "all_shapes_match",
            "total_elements",
            "max_abs",
            "mean_abs",
            "cosine",
            "passed",
        ):
            row[f"{stage}_kv_{key}"] = metrics.get(key)
        row[f"{stage}_kv_sequence_lengths"] = json.dumps(
            metrics.get("sequence_lengths")
        )
    greedy = case.get("greedy", {})
    row.update(
        {
            "greedy_checked_tokens": greedy.get("checked_tokens"),
            "greedy_expected_ids": json.dumps(greedy.get("expected_ids")),
            "greedy_actual_ids": json.dumps(greedy.get("actual_ids")),
            "greedy_exact_match": greedy.get("exact_match"),
            "greedy_first_mismatch_step": greedy.get("first_mismatch_step"),
        }
    )
    return row


def write_reports(
    report: dict[str, Any], args: argparse.Namespace, case_results: list[dict[str, Any]]
) -> None:
    for output in (args.output_json, args.output_csv):
        output = output.resolve()
        if output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {output}; pass --force")
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = [csv_row(case) for case in case_results]
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.greedy_tokens <= 0:
        raise ValueError("--greedy-tokens must be positive")
    for name in (
        "logits_max_abs",
        "logits_mean_abs",
        "kv_max_abs",
        "kv_mean_abs",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("logits_min_cosine", "kv_min_cosine"):
        if not -1.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [-1, 1]")
    if args.hf_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--hf-device cuda requested, but torch.cuda.is_available() is false"
        )
    for output in (args.output_json, args.output_csv):
        if output.resolve().exists() and not args.force:
            raise FileExistsError(
                f"refusing to overwrite {output.resolve()}; pass --force"
            )

    manifest, selected = load_selected_cases(args)
    config = load_config(args.model)
    num_layers = int(config.num_hidden_layers)
    onnx_paths = {
        mode: (args.onnx_dir / f"qwen2_5_0_5b_{mode}.onnx").resolve()
        for mode in ("logits", "prefill", "decode")
    }
    for mode, path in onnx_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {mode} ONNX graph: {path}")

    print(
        f"building HF FP16 references on {args.hf_device} for "
        f"{len(selected)} case(s): {', '.join(case['case_id'] for case in selected)}"
    )
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = load_model(args.model, "float16", args.hf_device)
    references = build_hf_references(model, selected, num_layers, args)

    case_results_by_id: dict[str, dict[str, Any]] = {
        reference.case_id: {
            "case_id": reference.case_id,
            "kind": reference.kind,
            "prompt_length": reference.prompt_length,
            "golden_path": str(reference.golden_path),
            "graphs": {},
            "kv": {},
            "greedy": {},
            "failures": [],
        }
        for reference in references
    }
    print("running logits graph with ONNX Runtime CPUExecutionProvider")
    ort_version = verify_logits_graph(
        references, case_results_by_id, onnx_paths["logits"], num_layers, args
    )
    print("running prefill graph and checking all KV outputs")
    prefill_ort_version = verify_prefill_graph(
        references, case_results_by_id, onnx_paths["prefill"], num_layers, args
    )
    print("running decode graph and cached greedy correctness gate")
    decode_ort_version = verify_decode_and_greedy(
        references, case_results_by_id, onnx_paths["decode"], num_layers, args
    )
    if len({ort_version, prefill_ort_version, decode_ort_version}) != 1:
        raise RuntimeError("ORT version changed while sessions were being created")

    case_results = [case_results_by_id[case["case_id"]] for case in selected]
    for result in case_results:
        result["passed"] = not result["failures"]
    failures = [
        f"{case['case_id']}: {failure}"
        for case in case_results
        for failure in case["failures"]
    ]
    report = {
        "schema_version": 1,
        "passed": not failures,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(args.model.resolve()),
            "hf_dtype": "float16",
            "hf_device": args.hf_device,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnxruntime": ort_version,
            "ort_provider": "CPUExecutionProvider",
        },
        "inputs": {
            "workload_manifest": str(args.manifest.resolve()),
            "workload_schema_version": manifest.get("schema_version"),
            "onnx": {
                mode: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for mode, path in onnx_paths.items()
            },
        },
        "tolerances": {
            "logits": {
                "max_abs": args.logits_max_abs,
                "mean_abs": args.logits_mean_abs,
                "min_cosine": args.logits_min_cosine,
                "require_top1_exact": True,
                "require_top5_set_exact": True,
                "record_ordered_top5_exact": True,
            },
            "kv": {
                "max_abs": args.kv_max_abs,
                "mean_abs": args.kv_mean_abs,
                "min_cosine": args.kv_min_cosine,
                "require_all_shapes_exact": True,
            },
            "cached_greedy": {"require_hf_golden_token_ids_exact": True},
        },
        "case_count": len(case_results),
        "failure_count": len(failures),
        "failures": failures,
        "cases": case_results,
    }
    write_reports(report, args, case_results)
    print(f"JSON report: {args.output_json.resolve()}")
    print(f"CSV report:  {args.output_csv.resolve()}")
    if failures:
        print(f"FAILED: {len(failures)} correctness gate(s)", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"PASS: {len(case_results)} case(s) satisfied every correctness gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
