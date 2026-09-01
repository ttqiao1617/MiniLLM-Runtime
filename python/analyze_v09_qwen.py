#!/usr/bin/env python3
"""Prepare, validate and aggregate immutable MiniLLM-RT v0.9 Qwen runs.

An official run is a self-contained evidence bundle under
``results/runs/v09/<run-id>``.  This tool never writes to ``results/raw`` and
refuses to replace a generated manifest or summary.  Create a new run ID when
an experiment or its configuration changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$")
PERFORMANCE_NAME_RE = re.compile(
    r"^v09_(full|kv)_([A-Za-z0-9][A-Za-z0-9._-]*)_p(\d+)_o(\d+)_r(\d+)\.csv$"
)
SCHEDULER_NAME_RE = re.compile(
    r"^v09_scheduler_a(\d+)_b(\d+)_r(\d+)_requests\.csv$"
)
VARIANT_NAMES = {"full": "full_recompute", "kv": "kv_cache"}
PROFILE_FIELDS = (
    "step_total_ms",
    "backend_total_ms",
    "input_prepare_ms",
    "backend_shape_ms",
    "backend_buffer_bind_ms",
    "h2d_ms",
    "enqueue_ms",
    "d2h_ms",
    "sync_ms",
    "logits_postprocess_ms",
    "sampler_ms",
    "h2d_bytes",
    "d2h_bytes",
    "bound_output_bytes",
    "output_tensors",
)
REQUIRED_STEP_COLUMNS = {
    "model",
    "provider",
    "mode",
    "prompt_len",
    "output_len",
    "step",
    "latency_ms",
    "next_token",
    "kv_init_ms",
    "managed_kv_bytes",
    "cuda_event_timing",
    *PROFILE_FIELDS,
}
EXPECTED_VARIANTS = ("full_recompute", "kv_cache")
EXPECTED_MODES = {"full": "naive", "kv": "kv_cache"}
MIN_WARMUP_RUNS = 5
MIN_MEASURED_REPEATS = 10
PROFILE_AGGREGATE_FIELDS = (
    "step_total_ms",
    "backend_total_ms",
    "input_prepare_ms",
    "backend_shape_ms",
    "backend_buffer_bind_ms",
    "h2d_ms",
    "enqueue_ms",
    "d2h_ms",
    "sync_ms",
    "logits_postprocess_ms",
    "sampler_ms",
    "h2d_bytes",
    "d2h_bytes",
    "bound_output_bytes",
    "output_tensors",
)
RUN_METRICS = (
    "ttft_ms",
    "kv_init_ms",
    "request_total_latency_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p95_tpot_ms",
    "decode_tokens_per_s",
    "end_to_end_tokens_per_s",
    "backend_ttft_ms",
    "backend_request_latency_ms",
    "backend_mean_tpot_ms",
    "backend_p50_tpot_ms",
    "backend_p95_tpot_ms",
    "backend_decode_tokens_per_s",
    "backend_tokens_per_s",
    "prefill_backend_total_ms",
    "prefill_enqueue_ms",
    "prefill_h2d_ms",
    "prefill_d2h_ms",
    "prefill_sampler_ms",
    "prefill_h2d_bytes",
    "prefill_d2h_bytes",
    "prefill_bound_output_bytes",
    "decode_avg_backend_total_ms",
    "decode_avg_enqueue_ms",
    "decode_avg_h2d_ms",
    "decode_avg_d2h_ms",
    "decode_avg_sampler_ms",
    "decode_avg_h2d_bytes",
    "decode_avg_d2h_bytes",
    "decode_avg_bound_output_bytes",
    "peak_managed_kv_bytes",
)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def positive_int_csv(raw: str) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return values


def positive_int_sequence_csv(raw: str) -> list[int]:
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def case_id_csv(raw: str) -> list[str]:
    values = raw.split(",")
    if not values or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) for value in values
    ):
        raise argparse.ArgumentTypeError("expected comma-separated workload case IDs")
    return values


def finite_float(raw: Any, *, field: str, source: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {field!r} in {source}: {raw!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field!r} in {source}: {raw!r}")
    return value


def nonnegative_float(raw: Any, *, field: str, source: Path) -> float:
    value = finite_float(raw, field=field, source=source)
    if value < 0.0:
        raise RuntimeError(f"negative {field!r} in {source}: {raw!r}")
    return value


def exact_int(raw: Any, *, field: str, source: Path) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid integer {field!r} in {source}: {raw!r}") from exc
    if str(raw).strip() != str(value):
        raise RuntimeError(f"non-canonical integer {field!r} in {source}: {raw!r}")
    return value


def optional_float(row: dict[str, str], field: str, source: Path) -> float:
    raw = row.get(field, "")
    if raw == "" and field == "bound_output_bytes":
        raw = row.get("output_bytes", "")
    return finite_float(raw, field=field, source=source) if raw != "" else 0.0


def mean_field(rows: Sequence[dict[str, str]], field: str, source: Path) -> float:
    values = [optional_float(row, field, source) for row in rows]
    return statistics.mean(values) if values else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path | None, hash_content: bool = True) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        if hash_content:
            record["sha256"] = sha256_file(resolved)
    return record


def write_text_immutable(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable run artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_json_immutable(path: Path, payload: Any) -> None:
    write_text_immutable(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_csv_immutable(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable run artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_model_spec(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    result: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid ModelSpec line {number} in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def parse_input_spec(raw: str) -> dict[str, Any]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "input spec must be CASE_ID:PROMPT_LEN:OUTPUT_LEN:IDS_PATH"
        )
    case_id, prompt_raw, output_raw, ids_raw = parts
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
        raise argparse.ArgumentTypeError(f"invalid case ID: {case_id!r}")
    try:
        prompt_len = int(prompt_raw)
        output_len = int(output_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid input lengths in {raw!r}") from exc
    if prompt_len <= 0 or output_len <= 0:
        raise argparse.ArgumentTypeError("prompt/output lengths must be positive")
    ids_path = Path(ids_raw).expanduser().resolve()
    if not ids_path.is_file():
        raise argparse.ArgumentTypeError(f"input IDs file does not exist: {ids_path}")
    raw_tokens = ids_path.read_text(encoding="utf-8").split()
    try:
        token_ids = [int(value) for value in raw_tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"input IDs file contains a non-integer: {ids_path}") from exc
    if any(value < 0 for value in token_ids):
        raise argparse.ArgumentTypeError(f"input IDs file contains a negative token: {ids_path}")
    actual_tokens = len(token_ids)
    if actual_tokens != prompt_len:
        raise argparse.ArgumentTypeError(
            f"{ids_path} contains {actual_tokens} IDs, expected {prompt_len}"
        )
    return {
        "case_id": case_id,
        "prompt_len": prompt_len,
        "output_len": output_len,
        "input_ids": artifact_record(ids_path),
        "observed_prompt_tokens": actual_tokens,
    }


def config_workloads(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        warmup = int(config["warmup_runs"])
        repeats = int(config["repeats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark config lacks valid warmup_runs/repeats") from exc
    if warmup < MIN_WARMUP_RUNS:
        raise RuntimeError(
            f"official v0.9 requires warmup_runs >= {MIN_WARMUP_RUNS}, found {warmup}"
        )
    if repeats < MIN_MEASURED_REPEATS:
        raise RuntimeError(
            f"official v0.9 requires repeats >= {MIN_MEASURED_REPEATS}, found {repeats}"
        )
    if config.get("variants") != list(EXPECTED_VARIANTS):
        raise RuntimeError(
            f"benchmark variants must be exactly {list(EXPECTED_VARIANTS)}, "
            f"found {config.get('variants')!r}"
        )
    if config.get("alternating_ab_order") is not True:
        raise RuntimeError("official v0.9 config must declare alternating_ab_order=true")
    workloads = config.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise RuntimeError("benchmark config must contain at least one workload")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, workload in enumerate(workloads):
        if not isinstance(workload, dict):
            raise RuntimeError(f"invalid workload at index {index}: expected an object")
        case_id = str(workload.get("case_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id):
            raise RuntimeError(f"invalid workload case_id at index {index}: {case_id!r}")
        if case_id in seen:
            raise RuntimeError(f"duplicate workload case_id: {case_id}")
        seen.add(case_id)
        try:
            prompt_len = int(workload["prompt_len"])
            output_len = int(workload["output_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid lengths for workload {case_id}") from exc
        if prompt_len <= 0 or output_len <= 0:
            raise RuntimeError(f"non-positive lengths for workload {case_id}")
        input_record = workload.get("input_ids")
        if not isinstance(input_record, dict) or input_record.get("exists") is not True:
            raise RuntimeError(f"workload {case_id} lacks an existing input_ids artifact")
        ids_path = Path(str(input_record.get("path", "")))
        if not ids_path.is_file():
            raise RuntimeError(f"workload input IDs no longer exist for {case_id}: {ids_path}")
        tokens = ids_path.read_text(encoding="utf-8").split()
        if len(tokens) != prompt_len:
            raise RuntimeError(
                f"workload input length changed for {case_id}: {len(tokens)} vs {prompt_len}"
            )
        if input_record.get("sha256") and sha256_file(ids_path) != input_record["sha256"]:
            raise RuntimeError(f"workload input hash changed for {case_id}: {ids_path}")
        normalized.append(
            {
                **workload,
                "case_id": case_id,
                "prompt_len": prompt_len,
                "output_len": output_len,
            }
        )
    return normalized


def expected_performance_files(config: dict[str, Any]) -> dict[str, tuple[str, str, int, int, int]]:
    workloads = config_workloads(config)
    repeats = int(config["repeats"])
    expected: dict[str, tuple[str, str, int, int, int]] = {}
    for workload in workloads:
        case_id = str(workload["case_id"])
        prompt_len = int(workload["prompt_len"])
        output_len = int(workload["output_len"])
        for repeat in range(1, repeats + 1):
            for variant in ("full", "kv"):
                name = (
                    f"v09_{variant}_{case_id}_p{prompt_len}_o{output_len}_r{repeat}.csv"
                )
                expected[name] = (variant, case_id, prompt_len, output_len, repeat)
    return expected


def config_scheduler(config: dict[str, Any]) -> dict[str, Any]:
    scheduler = config.get("scheduler")
    if not isinstance(scheduler, dict):
        raise RuntimeError("benchmark config lacks scheduler protocol")
    try:
        active = [int(value) for value in scheduler["max_active_requests"]]
        block_sizes = [int(value) for value in scheduler["block_sizes"]]
        repeats = int(scheduler["repeats"])
        request_count = int(scheduler["request_count"])
        request_case_ids = [str(value) for value in scheduler["request_case_ids"]]
        output_limits = [int(value) for value in scheduler["output_limits"]]
        pool_tokens = int(scheduler["pool_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("benchmark config contains an invalid scheduler protocol") from exc
    if not active or not block_sizes or any(value <= 0 for value in active + block_sizes):
        raise RuntimeError("scheduler active/block-size values must be positive")
    if len(active) != len(set(active)) or len(block_sizes) != len(set(block_sizes)):
        raise RuntimeError("scheduler active/block-size values must be unique")
    if repeats < MIN_MEASURED_REPEATS or request_count <= 0:
        raise RuntimeError("scheduler requires at least 10 repeats and a positive request count")
    workload_ids = {str(workload["case_id"]) for workload in config_workloads(config)}
    if len(request_case_ids) != request_count or any(
        case_id not in workload_ids for case_id in request_case_ids
    ):
        raise RuntimeError("scheduler request_case_ids do not match benchmark workloads/count")
    if len(output_limits) != request_count or any(value <= 0 for value in output_limits):
        raise RuntimeError("scheduler output_limits do not match request count")
    if pool_tokens <= 0 or any(pool_tokens % block_size for block_size in block_sizes):
        raise RuntimeError("scheduler pool_tokens must be positive and divisible by block sizes")
    return {
        "max_active_requests": active,
        "block_sizes": block_sizes,
        "repeats": repeats,
        "request_count": request_count,
        "request_case_ids": request_case_ids,
        "output_limits": output_limits,
        "pool_tokens": pool_tokens,
    }


def selected_model_metadata(model_dir: Path | None) -> list[dict[str, Any]]:
    if model_dir is None:
        return []
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model.safetensors",
    )
    return [
        artifact_record(model_dir / name)
        for name in names
        if (model_dir / name).is_file()
    ]


def validate_run_id(run_id: str, run_dir: Path) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise RuntimeError(
            "run ID must be 3-80 characters containing only letters, digits, '.', '_' or '-'"
        )
    if run_dir.name != run_id:
        raise RuntimeError(f"run directory basename {run_dir.name!r} does not match run ID {run_id!r}")


def command_prepare(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    validate_run_id(args.run_id, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup < MIN_WARMUP_RUNS or args.repeats < MIN_MEASURED_REPEATS:
        raise RuntimeError(
            f"official v0.9 requires warmup >= {MIN_WARMUP_RUNS} and "
            f"repeats >= {MIN_MEASURED_REPEATS}"
        )
    workloads = [parse_input_spec(raw) for raw in args.input_spec]
    if not workloads:
        raise RuntimeError("official v0.9 requires at least one --input-spec")
    case_ids = [item["case_id"] for item in workloads]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("input case IDs must be unique within one run")
    hash_content = not args.skip_artifact_hashes
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    model_spec = args.model_spec.expanduser().resolve() if args.model_spec else None
    engines = {
        "full_recompute": artifact_record(args.baseline_engine, hash_content),
        "prefill": artifact_record(args.prefill_engine, hash_content),
        "with_past_decode": artifact_record(args.decode_engine, hash_content),
    }
    required_artifacts = {
        "model_spec": artifact_record(model_spec, hash_content),
        "runtime_binary": artifact_record(args.runtime_binary, hash_content),
        **engines,
    }
    missing_artifacts = [
        name for name, record in required_artifacts.items()
        if record is None or record.get("exists") is not True
    ]
    if missing_artifacts:
        raise RuntimeError(
            "official v0.9 artifacts do not exist: " + ", ".join(missing_artifacts)
        )
    model_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "precision": args.precision,
        "model_dir": str(model_dir) if model_dir else None,
        "selected_model_metadata": selected_model_metadata(model_dir),
        "model_spec_artifact": artifact_record(model_spec, hash_content),
        "model_spec": parse_model_spec(model_spec),
        "engines": engines,
    }
    benchmark_config = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "minillm-rt-v09-qwen-ab-v1",
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "alternating_ab_order": True,
        "variants": ["full_recompute", "kv_cache"],
        "workloads": workloads,
        "runtime_binary": artifact_record(args.runtime_binary, hash_content),
        "model_spec_path": str(model_spec) if model_spec else None,
        "raw_performance_directory": "raw/performance",
        "raw_scheduler_directory": "raw/scheduler",
        "golden_paths": [str(path.expanduser().resolve()) for path in args.golden],
        "golden_artifacts": [
            artifact_record(path.expanduser().resolve(), hash_content) for path in args.golden
        ],
        "operator_notes": args.notes,
        "runner_command": args.runner_command,
        "scheduler": {
            "max_active_requests": args.scheduler_active,
            "block_sizes": args.scheduler_block_sizes,
            "repeats": args.scheduler_repeats,
            "request_count": args.scheduler_request_count,
            "request_case_ids": args.scheduler_case_ids,
            "output_limits": args.scheduler_output_limits,
            "pool_tokens": args.scheduler_pool_tokens,
            "max_tokens_per_step_policy": "equals_max_active_requests",
            "configuration_order": "repeat-major_rotated_alternating",
            "execution": "serial_round_robin_enqueue",
        },
    }
    config_workloads(benchmark_config)
    config_scheduler(benchmark_config)
    golden_paths = [path.expanduser().resolve() for path in args.golden]
    if not golden_paths:
        raise RuntimeError("official v0.9 requires at least one --golden with an exact case_id")
    validate_goldens(load_goldens(golden_paths), workloads)
    write_json_immutable(run_dir / "model_manifest.json", model_manifest)
    write_json_immutable(run_dir / "benchmark_config.json", benchmark_config)
    print(f"prepared immutable manifests in {run_dir}")


def read_step_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise RuntimeError(f"duplicate columns in performance CSV: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty performance CSV: {path}")
    missing = REQUIRED_STEP_COLUMNS.difference(fieldnames)
    if missing:
        raise RuntimeError(f"missing columns in {path}: {sorted(missing)}")
    return rows


def token_sequence(rows: Sequence[dict[str, str]], source: Path) -> list[int]:
    result: list[int] = []
    for row in rows:
        try:
            result.append(int(row["next_token"]))
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"invalid next_token in {source}: {row.get('next_token')!r}") from exc
    return result


def summarize_step_csv(
    path: Path,
    variant: str,
    case_id: str,
    prompt: int,
    output: int,
    repeat: int,
    eos_token_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_step_csv(path)
    metadata_prompt = {
        exact_int(row["prompt_len"], field="prompt_len", source=path) for row in rows
    }
    if metadata_prompt != {prompt}:
        raise RuntimeError(f"prompt metadata mismatch in {path}: {metadata_prompt} vs {prompt}")
    metadata_output = {
        exact_int(row["output_len"], field="output_len", source=path) for row in rows
    }
    if metadata_output != {output}:
        raise RuntimeError(f"output metadata mismatch in {path}: {metadata_output} vs {output}")
    providers = {row["provider"] for row in rows}
    if providers != {"tensorrt"}:
        raise RuntimeError(f"provider must be tensorrt in {path}, found {sorted(providers)}")
    modes = {row["mode"] for row in rows}
    expected_mode = EXPECTED_MODES[variant]
    if modes != {expected_mode}:
        raise RuntimeError(
            f"mode mismatch in {path}: expected {expected_mode!r}, found {sorted(modes)}"
        )
    models = {row["model"] for row in rows}
    if len(models) != 1 or not next(iter(models)).strip():
        raise RuntimeError(f"model metadata must be one non-empty value in {path}")

    steps = [exact_int(row["step"], field="step", source=path) for row in rows]
    if steps != list(range(len(rows))):
        raise RuntimeError(f"non-contiguous step sequence in {path}: {steps}")

    backend_latencies = [
        nonnegative_float(row["latency_ms"], field="latency_ms", source=path)
        for row in rows
    ]
    profile_values: dict[str, list[float]] = {
        field: [nonnegative_float(row[field], field=field, source=path) for row in rows]
        for field in PROFILE_FIELDS
    }
    step_totals = profile_values["step_total_ms"]
    for index, (backend_ms, total_ms) in enumerate(zip(backend_latencies, step_totals)):
        if total_ms + 1e-9 < backend_ms:
            raise RuntimeError(
                f"step_total_ms is smaller than latency_ms at step {index} in {path}"
            )
    kv_init_values = [
        nonnegative_float(row["kv_init_ms"], field="kv_init_ms", source=path)
        for row in rows
    ]
    if any(value != 0.0 for value in kv_init_values[1:]):
        raise RuntimeError(f"kv_init_ms must only be populated on step 0 in {path}")
    for row in rows:
        if exact_int(
            row["cuda_event_timing"], field="cuda_event_timing", source=path
        ) != 1:
            raise RuntimeError(f"cuda_event_timing must be 1 for every TensorRT row in {path}")

    sequence = token_sequence(rows, path)
    if any(token < 0 for token in sequence):
        raise RuntimeError(f"negative next_token in {path}")
    actual_tokens = len(sequence)
    if actual_tokens > output:
        raise RuntimeError(
            f"generated {actual_tokens} tokens in {path}, exceeding requested output {output}"
        )
    eos_positions = [index for index, token in enumerate(sequence) if token in eos_token_ids]
    if eos_positions and eos_positions[0] != actual_tokens - 1:
        raise RuntimeError(f"generation continued after EOS at step {eos_positions[0]} in {path}")
    if actual_tokens < output and (not sequence or sequence[-1] not in eos_token_ids):
        raise RuntimeError(
            f"short generation in {path}: {actual_tokens}/{output} tokens without terminal EOS"
        )

    backend_tpot = backend_latencies[1:]
    token_ready_tpot = step_totals[1:]
    backend_total = sum(backend_latencies)
    token_ready_total = sum(step_totals)
    kv_init = kv_init_values[0]
    request_total = token_ready_total + kv_init
    result: dict[str, Any] = {
        "source_csv": str(path),
        "case_id": case_id,
        "prompt_len": prompt,
        "requested_output_len": output,
        "actual_output_len": actual_tokens,
        "repeat": repeat,
        "variant": VARIANT_NAMES[variant],
        "model": rows[0]["model"],
        "provider": rows[0]["provider"],
        "mode": rows[0]["mode"],
        # Primary user-visible timing includes logits postprocess and sampling.
        "ttft_ms": step_totals[0],
        "kv_init_ms": kv_init,
        "request_total_latency_ms": request_total,
        "mean_tpot_ms": statistics.mean(token_ready_tpot) if token_ready_tpot else 0.0,
        "p50_tpot_ms": percentile(token_ready_tpot, 50),
        "p95_tpot_ms": percentile(token_ready_tpot, 95),
        "decode_tokens_per_s": (
            len(token_ready_tpot) / (sum(token_ready_tpot) / 1000.0)
            if sum(token_ready_tpot) > 0
            else 0.0
        ),
        "end_to_end_tokens_per_s": (
            actual_tokens / (request_total / 1000.0) if request_total > 0 else 0.0
        ),
        # Backend-only timing remains available under explicit names.
        "backend_ttft_ms": backend_latencies[0],
        "backend_request_latency_ms": backend_total,
        "backend_mean_tpot_ms": statistics.mean(backend_tpot) if backend_tpot else 0.0,
        "backend_p50_tpot_ms": percentile(backend_tpot, 50),
        "backend_p95_tpot_ms": percentile(backend_tpot, 95),
        "backend_decode_tokens_per_s": (
            len(backend_tpot) / (sum(backend_tpot) / 1000.0)
            if sum(backend_tpot) > 0
            else 0.0
        ),
        "backend_tokens_per_s": (
            actual_tokens / (backend_total / 1000.0) if backend_total > 0 else 0.0
        ),
        "token_sequence": " ".join(str(token) for token in sequence),
    }
    decode_rows = rows[1:]
    for field in PROFILE_AGGREGATE_FIELDS:
        values = profile_values[field]
        result[f"prefill_{field}"] = values[0]
        result[f"decode_avg_{field}"] = statistics.mean(values[1:]) if decode_rows else 0.0
        # Retained for consumers of the pre-v0.9 draft schema; do not use for
        # prefill/decode comparisons because it mixes the two phases.
        result[f"avg_{field}"] = statistics.mean(values)
    # Optional memory columns are retained as peaks only when the runtime emits them.
    for field in ("managed_kv_bytes", "kv_allocated_bytes", "active_kv_bytes", "gpu_memory_used_bytes"):
        values = [
            nonnegative_float(row[field], field=field, source=path)
            for row in rows
            if row.get(field, "") != ""
        ]
        result[f"peak_{field}"] = max(values) if values else None

    step_records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        detail: dict[str, Any] = {
            "source_csv": str(path),
            "case_id": case_id,
            "prompt_len": prompt,
            "requested_output_len": output,
            "repeat": repeat,
            "variant": VARIANT_NAMES[variant],
            "phase": "prefill" if index == 0 else "decode",
            "step": index,
            "history_len": prompt + index,
            "next_token": sequence[index],
            "backend_latency_ms": backend_latencies[index],
        }
        for field in PROFILE_FIELDS:
            detail[field] = profile_values[field][index]
        step_records.append(detail)
    return result, step_records


def first_mismatch(reference: Sequence[int], candidate: Sequence[int]) -> int:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    return min(len(reference), len(candidate)) if len(reference) != len(candidate) else -1


def correctness_row(
    *,
    case_id: str,
    prompt_len: int,
    requested_output_len: int,
    repeat: int | str,
    reference_name: str,
    candidate_name: str,
    reference: Sequence[int],
    candidate: Sequence[int],
    source: str,
) -> dict[str, Any]:
    mismatch = first_mismatch(reference, candidate)
    return {
        "case_id": case_id,
        "prompt_len": prompt_len,
        "requested_output_len": requested_output_len,
        "repeat": repeat,
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "exact_match": 1 if mismatch < 0 else 0,
        "first_mismatch_step": mismatch,
        "reference_sequence": " ".join(str(token) for token in reference),
        "candidate_sequence": " ".join(str(token) for token in candidate),
        "source": source,
    }


def load_goldens(paths: Iterable[Path]) -> list[dict[str, Any]]:
    goldens = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict) or "generated_ids" not in payload:
            raise RuntimeError(f"golden JSON lacks generated_ids: {path}")
        goldens.append(
            {
                "path": path,
                "case_id": payload.get("case_id", path.stem),
                "prompt_len": int(payload["prompt_length"]),
                "requested_output_len": int(payload["requested_new_tokens"]),
                "generated_ids": [int(value) for value in payload["generated_ids"]],
                "input_ids": [int(value) for value in payload.get("input_ids", [])],
            }
        )
    return goldens


def validate_goldens(
    goldens: Sequence[dict[str, Any]], workloads: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for golden in goldens:
        case_id = str(golden["case_id"])
        if case_id in by_case:
            raise RuntimeError(f"duplicate HF golden case_id: {case_id}")
        by_case[case_id] = golden
    expected = {str(workload["case_id"]): workload for workload in workloads}
    missing = sorted(set(expected).difference(by_case))
    extra = sorted(set(by_case).difference(expected))
    if missing or extra:
        raise RuntimeError(
            f"HF golden/workload case mismatch: missing={missing}, unexpected={extra}"
        )
    for case_id, workload in expected.items():
        golden = by_case[case_id]
        prompt = int(workload["prompt_len"])
        output = int(workload["output_len"])
        if int(golden["prompt_len"]) != prompt or int(golden["requested_output_len"]) != output:
            raise RuntimeError(
                f"HF golden lengths mismatch for {case_id}: "
                f"golden={golden['prompt_len']}/{golden['requested_output_len']} "
                f"workload={prompt}/{output}"
            )
        generated = list(golden["generated_ids"])
        if not generated or len(generated) > output:
            raise RuntimeError(f"HF golden output length is invalid for {case_id}")
        ids_path = Path(str(workload["input_ids"]["path"]))
        workload_ids = [int(value) for value in ids_path.read_text(encoding="utf-8").split()]
        if golden["input_ids"] and list(golden["input_ids"]) != workload_ids:
            raise RuntimeError(f"HF golden input IDs do not match workload {case_id}")
    return by_case


def aggregate_performance(
    run_dir: Path,
    config: dict[str, Any],
    model_manifest: dict[str, Any],
    golden_paths: Sequence[Path],
    allow_incomplete_pairs: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_dir = run_dir / "raw" / "performance"
    expected = expected_performance_files(config)
    discovered: dict[str, Path] = {}
    malformed: list[str] = []
    for path in sorted(raw_dir.glob("*.csv")):
        if PERFORMANCE_NAME_RE.fullmatch(path.name):
            discovered[path.name] = path
        elif path.name.startswith("v09_"):
            malformed.append(path.name)
    if malformed:
        raise RuntimeError(f"malformed v0.9 performance filenames: {malformed}")
    missing = sorted(set(expected).difference(discovered))
    unexpected = sorted(set(discovered).difference(expected))
    if unexpected or (missing and not allow_incomplete_pairs):
        raise RuntimeError(
            f"performance matrix mismatch: missing={missing[:12]}"
            f"{'...' if len(missing) > 12 else ''}, unexpected={unexpected}"
        )

    workloads = config_workloads(config)
    goldens = load_goldens(golden_paths)
    golden_by_case = validate_goldens(goldens, workloads)
    spec = model_manifest.get("model_spec", {})
    try:
        eos_token_ids = {
            int(value.strip()) for value in str(spec["eos_token_ids"]).split(",")
        }
    except (KeyError, ValueError) as exc:
        raise RuntimeError("model manifest lacks valid eos_token_ids") from exc
    expected_engines = {
        "full": Path(model_manifest["engines"]["full_recompute"]["path"]).resolve(),
        "kv": Path(model_manifest["engines"]["prefill"]["path"]).resolve(),
    }
    element_sizes = {"float16": 2, "bfloat16": 2, "float32": 4}
    try:
        kv_element_size = element_sizes[str(spec["kv_dtype"])]
        layers = int(spec["num_hidden_layers"])
        kv_heads = int(spec["num_kv_heads"])
        head_dim = int(spec["head_dim"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("model manifest lacks valid KV byte dimensions") from exc

    records: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    for name, metadata in sorted(expected.items()):
        if name not in discovered:
            continue
        variant, case_id, prompt, output, repeat = metadata
        record, steps = summarize_step_csv(
            discovered[name], variant, case_id, prompt, output, repeat, eos_token_ids
        )
        actual_engine = Path(str(record["model"])).resolve()
        if actual_engine != expected_engines[variant]:
            raise RuntimeError(
                f"engine metadata mismatch in {discovered[name]}: "
                f"{actual_engine} vs {expected_engines[variant]}"
            )
        expected_managed_kv = (
            0
            if variant == "full"
            else 2 * layers * 2 * kv_heads * (prompt + output) * head_dim * kv_element_size
        )
        if int(float(record["peak_managed_kv_bytes"])) != expected_managed_kv:
            raise RuntimeError(
                f"managed KV byte accounting mismatch in {discovered[name]}: "
                f"{record['peak_managed_kv_bytes']} vs {expected_managed_kv}"
            )
        records.append(record)
        all_steps.extend(steps)
    if not records:
        raise RuntimeError(f"no v0.9 performance CSVs found in {raw_dir}")

    pairs: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["case_id"]),
            int(record["prompt_len"]),
            int(record["requested_output_len"]),
            int(record["repeat"]),
        )
        pairs[key][str(record["variant"])] = record

    correctness: list[dict[str, Any]] = []
    for key, pair in sorted(pairs.items()):
        if set(pair) != {"full_recompute", "kv_cache"}:
            if allow_incomplete_pairs:
                continue
            raise RuntimeError(f"missing full/KV pair for {key}: found {sorted(pair)}")
        full = [int(value) for value in str(pair["full_recompute"]["token_sequence"]).split()]
        kv = [int(value) for value in str(pair["kv_cache"]["token_sequence"]).split()]
        correctness.append(
            correctness_row(
                case_id=key[0],
                prompt_len=key[1],
                requested_output_len=key[2],
                repeat=key[3],
                reference_name="trt_full_recompute",
                candidate_name="trt_kv_cache",
                reference=full,
                candidate=kv,
                source="paired_performance_csv",
            )
        )

    for record in records:
        golden = golden_by_case[str(record["case_id"])]
        candidate = [int(value) for value in str(record["token_sequence"]).split()]
        correctness.append(
            correctness_row(
                case_id=str(record["case_id"]),
                prompt_len=int(record["prompt_len"]),
                requested_output_len=int(record["requested_output_len"]),
                repeat=int(record["repeat"]),
                reference_name="hf_greedy",
                candidate_name=f"trt_{record['variant']}",
                reference=golden["generated_ids"],
                candidate=candidate,
                source=str(golden["path"]),
            )
        )

    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["case_id"]),
                int(record["prompt_len"]),
                int(record["requested_output_len"]),
                str(record["variant"]),
            )
        ].append(record)
    aggregate: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        actual_lengths = {int(row["actual_output_len"]) for row in rows}
        out: dict[str, Any] = {
            "case_id": key[0],
            "prompt_len": key[1],
            "requested_output_len": key[2],
            "variant": key[3],
            "repeats": len(rows),
            "min_actual_output_len": min(actual_lengths),
            "max_actual_output_len": max(actual_lengths),
        }
        for metric in RUN_METRICS:
            values = [float(row[metric]) for row in rows]
            out[f"mean_{metric}"] = f"{statistics.mean(values):.6f}"
            out[f"median_{metric}"] = f"{statistics.median(values):.6f}"
            out[f"stdev_{metric}"] = f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"
            out[f"p95_{metric}"] = f"{percentile(values, 95):.6f}"
        aggregate.append(out)

    aggregate_lookup = {
        (str(row["case_id"]), int(row["prompt_len"]), int(row["requested_output_len"]), str(row["variant"])): row
        for row in aggregate
    }
    cases = sorted({key[:3] for key in aggregate_lookup})
    for case in cases:
        full_key = (*case, "full_recompute")
        kv_key = (*case, "kv_cache")
        if full_key not in aggregate_lookup or kv_key not in aggregate_lookup:
            if allow_incomplete_pairs:
                continue
            raise RuntimeError(f"missing aggregate full/KV variant for {case}")
        full_summary = aggregate_lookup[full_key]
        kv_summary = aggregate_lookup[kv_key]
        repeat_pairs = [pair for key, pair in pairs.items() if key[:3] == case and set(pair) == {"full_recompute", "kv_cache"}]
        for metric in RUN_METRICS:
            full_summary[f"change_{metric}_pct"] = "0.000000"
            full_summary[f"paired_mean_change_{metric}_pct"] = "0.000000"
            full_summary[f"paired_median_change_{metric}_pct"] = "0.000000"
            baseline_mean = float(full_summary[f"mean_{metric}"])
            candidate_mean = float(kv_summary[f"mean_{metric}"])
            kv_summary[f"change_{metric}_pct"] = (
                f"{(candidate_mean / baseline_mean - 1.0) * 100.0:.6f}"
                if baseline_mean
                else ("0.000000" if candidate_mean == 0.0 else "")
            )
            changes = []
            for pair in repeat_pairs:
                baseline = float(pair["full_recompute"][metric])
                candidate = float(pair["kv_cache"][metric])
                if baseline:
                    changes.append((candidate / baseline - 1.0) * 100.0)
                elif candidate == 0.0:
                    changes.append(0.0)
            kv_summary[f"paired_mean_change_{metric}_pct"] = (
                f"{statistics.mean(changes):.6f}" if changes else ""
            )
            kv_summary[f"paired_median_change_{metric}_pct"] = (
                f"{statistics.median(changes):.6f}" if changes else ""
            )

    decode_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in all_steps:
        if row["phase"] != "decode":
            continue
        key = (
            row["case_id"],
            row["prompt_len"],
            row["requested_output_len"],
            row["variant"],
            row["step"],
            row["history_len"],
        )
        decode_grouped[key].append(row)
    decode_summary: list[dict[str, Any]] = []
    trend_metrics = (
        "step_total_ms",
        "backend_latency_ms",
        "backend_total_ms",
        "enqueue_ms",
        "h2d_ms",
        "d2h_ms",
        "sampler_ms",
        "h2d_bytes",
        "d2h_bytes",
        "bound_output_bytes",
    )
    for key, rows in sorted(decode_grouped.items()):
        out: dict[str, Any] = {
            "case_id": key[0],
            "prompt_len": key[1],
            "requested_output_len": key[2],
            "variant": key[3],
            "step": key[4],
            "history_len": key[5],
            "repeats": len(rows),
        }
        for metric in trend_metrics:
            values = [float(row[metric]) for row in rows]
            out[f"mean_{metric}"] = f"{statistics.mean(values):.6f}"
            out[f"median_{metric}"] = f"{statistics.median(values):.6f}"
            out[f"p95_{metric}"] = f"{percentile(values, 95):.6f}"
        decode_summary.append(out)
    if not decode_summary:
        raise RuntimeError("no decode steps were available for trend aggregation")
    return records, aggregate, correctness, decode_summary


def aggregate_scheduler(
    run_dir: Path,
    performance_records: Sequence[dict[str, Any]],
    config: dict[str, Any],
    model_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_dir = run_dir / "raw" / "scheduler"
    runs: list[dict[str, Any]] = []
    reference: dict[tuple[str, int, int], str] = {}
    scheduler_config = config_scheduler(config)
    try:
        eos_token_ids = {
            int(value.strip())
            for value in str(model_manifest["model_spec"]["eos_token_ids"]).split(",")
        }
        kv_dtype = str(model_manifest["model_spec"]["kv_dtype"])
        kv_element_size = {"float16": 2, "bfloat16": 2, "float32": 4}[kv_dtype]
        kv_layers = int(model_manifest["model_spec"]["num_hidden_layers"])
        kv_heads = int(model_manifest["model_spec"]["num_kv_heads"])
        kv_head_dim = int(model_manifest["model_spec"]["head_dim"])
        engine_max_sequence = int(
            model_manifest["model_spec"]["engine_max_sequence_length"]
        )
    except (KeyError, ValueError) as exc:
        raise RuntimeError("model manifest lacks scheduler EOS token IDs") from exc
    workload_by_case = {
        str(workload["case_id"]): workload for workload in config_workloads(config)
    }
    scheduled_case_ids = set(scheduler_config["request_case_ids"])
    single_request_reference: dict[str, list[int]] = {}
    for record in performance_records:
        case_id = str(record["case_id"])
        if record["variant"] != "full_recompute" or case_id not in scheduled_case_ids:
            continue
        tokens = [int(value) for value in str(record["token_sequence"]).split()]
        if case_id in single_request_reference and single_request_reference[case_id] != tokens:
            raise RuntimeError(f"full-recompute reference is non-deterministic for {case_id}")
        single_request_reference[case_id] = tokens
    required_request_columns = {
        "request_id", "prompt_len", "output_len", "finish_reason", "queue_wait_ms",
        "estimated_ttft_ms", "mean_compute_tpot_ms", "p95_compute_tpot_ms", "end_to_end_ms",
        "kv_allocated_bytes", "block_count", "block_size", "reserved_tokens",
        "committed_tokens", "block_rounding_waste_tokens", "block_table",
        "token_sequence",
    }
    required_timeline_columns = {
        "event_index", "request_id", "phase", "active_requests", "active_kv_bytes",
        "free_kv_blocks", "allocated_kv_blocks", "live_kv_tokens",
        "reserved_kv_tokens", "block_rounding_waste_tokens", "start_ms", "end_ms",
    }
    for path in sorted(raw_dir.glob("*_requests.csv")):
        match = SCHEDULER_NAME_RE.fullmatch(path.name)
        if not match:
            continue
        active, block_size, repeat = map(int, match.groups())
        with path.open(newline="", encoding="utf-8-sig") as stream:
            request_reader = csv.DictReader(stream)
            request_fields = set(request_reader.fieldnames or [])
            requests = list(request_reader)
        if not requests:
            raise RuntimeError(f"empty scheduler request CSV: {path}")
        if len(requests) != scheduler_config["request_count"]:
            raise RuntimeError(
                f"scheduler request count mismatch in {path}: "
                f"{len(requests)} vs {scheduler_config['request_count']}"
            )
        if missing := required_request_columns.difference(request_fields):
            raise RuntimeError(f"scheduler request CSV lacks {sorted(missing)}: {path}")
        timeline_path = path.with_name(path.name.replace("_requests.csv", "_timeline.csv"))
        if not timeline_path.is_file():
            raise RuntimeError(f"missing scheduler timeline: {timeline_path}")
        with timeline_path.open(newline="", encoding="utf-8-sig") as stream:
            timeline_reader = csv.DictReader(stream)
            timeline_fields = set(timeline_reader.fieldnames or [])
            timeline = list(timeline_reader)
        if not timeline:
            raise RuntimeError(f"empty scheduler timeline: {timeline_path}")
        if missing := required_timeline_columns.difference(timeline_fields):
            raise RuntimeError(f"scheduler timeline lacks {sorted(missing)}: {timeline_path}")
        expected_ids = [f"request_{index}" for index in range(len(requests))]
        if [row["request_id"] for row in requests] != expected_ids:
            raise RuntimeError(f"scheduler request IDs/order changed in {path}")
        visible_ttft: list[float] = []
        visible_tpot: list[float] = []
        compute_tpot: list[float] = []
        compute_p95_tpot: list[float] = []
        for index, row in enumerate(requests):
            expected_case = scheduler_config["request_case_ids"][index]
            expected_limit = scheduler_config["output_limits"][index]
            expected_prompt = int(workload_by_case[expected_case]["prompt_len"])
            prompt = exact_int(row["prompt_len"], field="prompt_len", source=path)
            output = exact_int(row["output_len"], field="output_len", source=path)
            if prompt != expected_prompt:
                raise RuntimeError(
                    f"scheduler prompt mismatch for request {index}: {prompt} vs {expected_prompt}"
                )
            if expected_case not in single_request_reference:
                raise RuntimeError(
                    f"scheduler case {expected_case} lacks a single-request reference"
                )
            sequence = [int(value) for value in row["token_sequence"].split()]
            if len(sequence) != output:
                raise RuntimeError(f"scheduler output_len/token sequence mismatch in {path}")
            if sequence != single_request_reference[expected_case][:output]:
                raise RuntimeError(
                    f"scheduler output differs from single-request reference for request {index}: {path}"
                )
            if row["finish_reason"] not in {"length", "eos"}:
                raise RuntimeError(f"invalid scheduler finish reason in {path}: {row['finish_reason']}")
            if row["finish_reason"] == "length" and output != expected_limit:
                raise RuntimeError(
                    f"scheduler request {index} finished early by length: {output}/{expected_limit}"
                )
            if row["finish_reason"] == "eos" and output > expected_limit:
                raise RuntimeError(f"scheduler EOS request exceeded its output limit in {path}")
            if row["finish_reason"] == "eos" and (
                not sequence or sequence[-1] not in eos_token_ids
            ):
                raise RuntimeError(f"scheduler EOS finish reason lacks terminal EOS token in {path}")
            if exact_int(row["block_size"], field="block_size", source=path) != block_size:
                raise RuntimeError(f"scheduler block_size metadata mismatch in {path}")
            committed = exact_int(
                row["committed_tokens"], field="committed_tokens", source=path
            )
            if committed != prompt + expected_limit:
                raise RuntimeError(f"scheduler committed-token reservation mismatch in {path}")
            block_count = exact_int(row["block_count"], field="block_count", source=path)
            expected_blocks = (committed + block_size - 1) // block_size
            reserved = exact_int(row["reserved_tokens"], field="reserved_tokens", source=path)
            waste = exact_int(
                row["block_rounding_waste_tokens"],
                field="block_rounding_waste_tokens",
                source=path,
            )
            block_table = row["block_table"].split()
            if block_count != expected_blocks or len(block_table) != expected_blocks:
                raise RuntimeError(f"scheduler block-table length mismatch in {path}")
            if reserved != block_count * block_size or waste != reserved - committed:
                raise RuntimeError(f"scheduler block accounting mismatch in {path}")
            physical_capacity = min(reserved, engine_max_sequence)
            expected_kv_bytes = (
                2
                * kv_layers
                * 2
                * kv_heads
                * physical_capacity
                * kv_head_dim
                * kv_element_size
            )
            actual_kv_bytes = exact_int(
                row["kv_allocated_bytes"], field="kv_allocated_bytes", source=path
            )
            if actual_kv_bytes != expected_kv_bytes:
                raise RuntimeError(
                    f"scheduler managed KV byte accounting mismatch in {path}: "
                    f"{actual_kv_bytes} vs {expected_kv_bytes}"
                )
            key = (row["request_id"], prompt, output)
            reference.setdefault(key, row["token_sequence"])
            if row["token_sequence"] != reference[key]:
                raise RuntimeError(f"scheduler token mismatch for request {index}: {path}")
            token_events = sorted(
                (
                    event
                    for event in timeline
                    if event["request_id"] == row["request_id"]
                    and event["phase"] in {"prefill", "decode"}
                ),
                key=lambda event: exact_int(
                    event["event_index"], field="event_index", source=timeline_path
                ),
            )
            if len(token_events) != output:
                raise RuntimeError(f"scheduler token event count mismatch in {timeline_path}")
            ready_times = [
                nonnegative_float(event["end_ms"], field="end_ms", source=timeline_path)
                for event in token_events
            ]
            visible_ttft.append(ready_times[0])
            visible_tpot.extend(
                later - earlier for earlier, later in zip(ready_times, ready_times[1:])
            )
            arrival_ms = nonnegative_float(
                row["arrival_ms"], field="arrival_ms", source=path
            )
            request_end_to_end_ms = nonnegative_float(
                row["end_to_end_ms"], field="end_to_end_ms", source=path
            )
            # Runtime CSVs use the C++ stream's default six significant
            # digits, so timestamps above 1,000 ms are quantized at roughly
            # 0.01 ms. Compare the same relative timestamp and account for
            # that text serialization precision without weakening the gate
            # at short durations.
            if not math.isclose(
                ready_times[-1] - arrival_ms,
                request_end_to_end_ms,
                rel_tol=1e-5,
                abs_tol=1e-3,
            ):
                raise RuntimeError(f"scheduler completion/timeline timestamp mismatch in {path}")
            compute_tpot.append(
                nonnegative_float(
                    row["mean_compute_tpot_ms"], field="mean_compute_tpot_ms", source=path
                )
            )
            compute_p95_tpot.append(
                nonnegative_float(
                    row["p95_compute_tpot_ms"], field="p95_compute_tpot_ms", source=path
                )
            )
        release_events = [row for row in timeline if row["phase"] == "release"]
        if len(release_events) != len(requests):
            raise RuntimeError(f"scheduler must emit one release event per request: {timeline_path}")
        last = timeline[-1]
        for field in ("active_requests", "active_kv_bytes", "allocated_kv_blocks",
                      "live_kv_tokens", "reserved_kv_tokens"):
            if exact_int(last[field], field=field, source=timeline_path) != 0:
                raise RuntimeError(f"scheduler did not release all resources ({field}): {timeline_path}")
        observed_peak_active = max(
            exact_int(row["active_requests"], field="active_requests", source=timeline_path)
            for row in timeline
        )
        pool_blocks = scheduler_config["pool_tokens"] // block_size
        expected_peak_active = min(active, scheduler_config["request_count"])
        first_reservations = 0
        for index in range(expected_peak_active):
            case_id = scheduler_config["request_case_ids"][index]
            prompt = int(workload_by_case[case_id]["prompt_len"])
            committed = prompt + scheduler_config["output_limits"][index]
            first_reservations += (committed + block_size - 1) // block_size
        if first_reservations > pool_blocks:
            raise RuntimeError(
                f"official scheduler pool cannot reach configured active={active} for block={block_size}"
            )
        if observed_peak_active != expected_peak_active:
            raise RuntimeError(
                f"scheduler peak active mismatch in {timeline_path}: "
                f"{observed_peak_active} vs {expected_peak_active}"
            )
        for row in timeline:
            free_blocks = exact_int(
                row["free_kv_blocks"], field="free_kv_blocks", source=timeline_path
            )
            allocated_blocks = exact_int(
                row["allocated_kv_blocks"],
                field="allocated_kv_blocks",
                source=timeline_path,
            )
            if free_blocks + allocated_blocks != pool_blocks:
                raise RuntimeError(f"scheduler block-pool conservation failed in {timeline_path}")
        makespan = max(float(row["end_to_end_ms"]) for row in requests)
        tokens = sum(int(row["output_len"]) for row in requests)
        runs.append(
            {
                "max_active_requests": active,
                "block_size": block_size,
                "repeat": repeat,
                "request_count": len(requests),
                "makespan_ms": makespan,
                "aggregate_tokens_per_s": tokens / (makespan / 1000.0),
                "mean_queue_wait_ms": statistics.mean(float(row["queue_wait_ms"]) for row in requests),
                "mean_visible_ttft_ms": statistics.mean(visible_ttft),
                "p95_visible_ttft_ms": percentile(visible_ttft, 95),
                "mean_visible_tpot_ms": statistics.mean(visible_tpot),
                "p95_visible_tpot_ms": percentile(visible_tpot, 95),
                "mean_compute_only_tpot_ms": statistics.mean(compute_tpot),
                "mean_compute_only_p95_tpot_ms": statistics.mean(compute_p95_tpot),
                "mean_end_to_end_ms": statistics.mean(
                    float(row["end_to_end_ms"]) for row in requests
                ),
                "p95_end_to_end_ms": percentile(
                    [float(row["end_to_end_ms"]) for row in requests], 95
                ),
                "peak_kv_bytes": max(int(row.get("active_kv_bytes", 0) or 0) for row in timeline),
                "peak_allocated_blocks": max(int(row.get("allocated_kv_blocks", 0) or 0) for row in timeline),
                "peak_rounding_waste_tokens": max(int(row.get("block_rounding_waste_tokens", 0) or 0) for row in timeline),
                "release_events": len(release_events),
            }
        )
    if not runs:
        raise RuntimeError(f"no scheduler evidence found in {raw_dir}")
    grouped_repeats: dict[tuple[int, int], set[int]] = defaultdict(set)
    for row in runs:
        grouped_repeats[(int(row["max_active_requests"]), int(row["block_size"]))].add(
            int(row["repeat"])
        )
    required_matrix = {
        (active, block)
        for active in scheduler_config["max_active_requests"]
        for block in scheduler_config["block_sizes"]
    }
    missing_matrix = sorted(required_matrix.difference(grouped_repeats))
    if missing_matrix:
        raise RuntimeError(f"scheduler matrix is incomplete: missing {missing_matrix}")
    expected_repeats = set(range(1, scheduler_config["repeats"] + 1))
    for configuration in required_matrix:
        if grouped_repeats[configuration] != expected_repeats:
            raise RuntimeError(
                f"scheduler repeats are incomplete for {configuration}: "
                f"{sorted(grouped_repeats[configuration])}"
            )
    metrics = [key for key in runs[0] if key not in {"max_active_requests", "block_size", "repeat"}]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(int(row["max_active_requests"]), int(row["block_size"]))].append(row)
    result = []
    for (active, block_size), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "max_active_requests": active,
            "block_size": block_size,
            "repeats": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            out[f"mean_{metric}"] = f"{statistics.mean(values):.6f}"
            out[f"median_{metric}"] = f"{statistics.median(values):.6f}"
            out[f"stdev_{metric}"] = f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"
            out[f"p95_{metric}"] = f"{percentile(values, 95):.6f}"
        result.append(out)
    return result


def union_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def assert_csv_matches(
    path: Path, expected_rows: Sequence[dict[str, Any]], label: str
) -> None:
    actual = read_csv_dicts(path)
    fields = union_fields(expected_rows)
    expected = [
        {
            field: "" if row.get(field) is None else str(row.get(field, ""))
            for field in fields
        }
        for row in expected_rows
    ]
    actual_projected = [{field: row.get(field, "") for field in fields} for row in actual]
    if set(actual[0] if actual else {}) != set(fields) or actual_projected != expected:
        raise RuntimeError(f"frozen {label} does not match regenerated raw evidence: {path}")


def aggregate_gpu_trace(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "raw" / "nvml" / "gpu_trace.csv"
    if not path.is_file():
        raise RuntimeError(f"missing NVML GPU trace: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty NVML GPU trace: {path}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["phase"])].append(row)
    result: list[dict[str, Any]] = []
    numeric_fields = (
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "memory.used",
        "memory.total",
    )
    for phase, samples in sorted(grouped.items()):
        out: dict[str, Any] = {"phase": phase, "samples": len(samples)}
        for field in numeric_fields:
            values = []
            for row in samples:
                raw = str(row.get(field, "")).strip()
                if not raw or "N/A" in raw:
                    continue
                values.append(finite_float(raw, field=field, source=path))
            out[f"min_{field}"] = f"{min(values):.6f}" if values else ""
            out[f"mean_{field}"] = f"{statistics.mean(values):.6f}" if values else ""
            out[f"max_{field}"] = f"{max(values):.6f}" if values else ""
        memory = [
            finite_float(row["memory.used"], field="memory.used", source=path)
            for row in samples
            if row.get("memory.used", "") and "N/A" not in row["memory.used"]
        ]
        if not memory:
            raise RuntimeError(f"NVML trace has no numeric memory.used samples for phase {phase}")
        out["memory_used_delta_mib"] = f"{max(memory) - min(memory):.6f}"
        result.append(out)
    return result


def aggregate_process_runs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "raw" / "process_runs.csv"
    if not path.is_file():
        raise RuntimeError(f"missing process wall-time evidence: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty process wall-time evidence: {path}")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["kind"], row["case_id"], row["variant"])
        grouped[key].append(
            nonnegative_float(row["process_wall_ms"], field="process_wall_ms", source=path)
        )
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "kind": key[0],
                "case_id": key[1],
                "variant": key[2],
                "runs": len(values),
                "mean_process_wall_ms": f"{statistics.mean(values):.6f}",
                "median_process_wall_ms": f"{statistics.median(values):.6f}",
                "p95_process_wall_ms": f"{percentile(values, 95):.6f}",
            }
        )
    return result


def command_aggregate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "benchmark_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing benchmark config: {config_path}")
    config = load_json(config_path)
    validate_run_id(str(config["run_id"]), run_dir)
    model_manifest = load_json(run_dir / "model_manifest.json")
    golden_paths = [path.expanduser().resolve() for path in args.golden]
    if not golden_paths:
        golden_paths = [Path(path) for path in config.get("golden_paths", [])]
    records, performance, correctness, decode_steps = aggregate_performance(
        run_dir, config, model_manifest, golden_paths, args.allow_incomplete_pairs
    )
    mismatches = [row for row in correctness if int(row["exact_match"]) != 1]
    if mismatches and not args.allow_token_mismatch:
        examples = ", ".join(
            f"{row['case_id']}/r{row['repeat']}:{row['reference']}->{row['candidate']}@{row['first_mismatch_step']}"
            for row in mismatches[:5]
        )
        raise RuntimeError(f"correctness gate failed for {len(mismatches)} comparisons: {examples}")

    # Validate every raw evidence family before creating any immutable summary.
    # This avoids stranding an expensive run with a half-written final bundle.
    scheduler = aggregate_scheduler(run_dir, records, config, model_manifest)
    memory = aggregate_gpu_trace(run_dir)
    process_summary = aggregate_process_runs(run_dir)

    raw_details = run_dir / "raw" / "performance_runs.csv"
    write_csv_immutable(raw_details, records, union_fields(records))
    write_csv_immutable(
        run_dir / "correctness_summary.csv", correctness, union_fields(correctness)
    )
    write_csv_immutable(
        run_dir / "performance_summary.csv", performance, union_fields(performance)
    )
    write_csv_immutable(
        run_dir / "decode_step_summary.csv", decode_steps, union_fields(decode_steps)
    )
    if scheduler:
        write_csv_immutable(
            run_dir / "scheduler_summary.csv", scheduler, union_fields(scheduler)
        )
    write_csv_immutable(run_dir / "memory_summary.csv", memory, union_fields(memory))
    write_csv_immutable(
        run_dir / "process_summary.csv", process_summary, union_fields(process_summary)
    )

    readme = [
        f"# MiniLLM-RT v0.9 benchmark run `{config['run_id']}`",
        "",
        "> This directory is immutable benchmark evidence. Do not regenerate files in place; use a new run ID.",
        "",
        f"- Protocol: `{config.get('protocol', 'unknown')}`",
        f"- Warmup runs: {config.get('warmup_runs')}",
        f"- Measured repeats: {config.get('repeats')}",
        f"- Operator notes: {config.get('operator_notes') or 'none'}",
        "- Scheduler execution: serial round-robin enqueue (not continuous batching)",
        f"- Performance run CSVs: {len(records)}",
        f"- Correctness comparisons: {len(correctness)} ({len(mismatches)} mismatch)",
        f"- Scheduler configurations: {len(scheduler)}",
        "",
        "Tracked evidence:",
        "",
        "- `environment.json`: hardware/software/GPU state at run start",
        "- `model_manifest.json`: model revision, ModelSpec and Engine hashes",
        "- `benchmark_config.json`: exact workload and A/B protocol",
        "- `correctness_summary.csv`: complete token-sequence gates",
        "- `performance_summary.csv`: repeated and paired aggregates",
        "- `decode_step_summary.csv`: per-history-length decode trends",
        "- `memory_summary.csv`: global NVML memory/clock/power samples by run phase",
        "- `process_summary.csv`: deserialize + warmup + measured-process wall time",
    ]
    if scheduler:
        readme.append("- `scheduler_summary.csv`: request/scheduler aggregates")
    readme.extend(
        [
            "",
            "Large per-step CSVs and logs remain under `raw/` and are intentionally not tracked by Git.",
            "",
        ]
    )
    write_text_immutable(run_dir / "README.md", "\n".join(readme))
    print(f"validated {len(records)} performance runs and {len(correctness)} correctness comparisons")
    print(f"performance summary: {run_dir / 'performance_summary.csv'}")
    if scheduler:
        print(f"scheduler summary: {run_dir / 'scheduler_summary.csv'}")


def command_verify(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    required = ["environment.json", "model_manifest.json", "benchmark_config.json"]
    if args.final:
        required.extend(
            [
                "artifact_manifest.json",
                "correctness_summary.csv",
                "performance_summary.csv",
                "decode_step_summary.csv",
                "scheduler_summary.csv",
                "memory_summary.csv",
                "process_summary.csv",
                "README.md",
            ]
        )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete run {run_dir}: missing {', '.join(missing)}")
    config = load_json(run_dir / "benchmark_config.json")
    validate_run_id(str(config["run_id"]), run_dir)
    environment = load_json(run_dir / "environment.json")
    model = load_json(run_dir / "model_manifest.json")
    for name, payload in (("environment", environment), ("model", model), ("config", config)):
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported {name} schema: {payload.get('schema_version')}")
    workloads = config_workloads(config)
    for label, record in {
        "runtime_binary": config.get("runtime_binary"),
        "model_spec": model.get("model_spec_artifact"),
        **dict(model.get("engines", {})),
    }.items():
        if not isinstance(record, dict) or record.get("exists") is not True:
            raise RuntimeError(f"missing artifact record in final manifest: {label}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"artifact changed or disappeared since prepare: {label}: {path}")
        if record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"artifact hash changed since prepare: {label}: {path}")
    for index, record in enumerate(config.get("golden_artifacts", [])):
        if not isinstance(record, dict) or record.get("exists") is not True:
            raise RuntimeError(f"invalid golden artifact record at index {index}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"HF golden changed or disappeared: {path}")
        if record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"HF golden hash changed: {path}")
    if args.final:
        for csv_name in (
            "correctness_summary.csv",
            "performance_summary.csv",
            "decode_step_summary.csv",
            "scheduler_summary.csv",
            "memory_summary.csv",
            "process_summary.csv",
        ):
            with (run_dir / csv_name).open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise RuntimeError(f"empty final summary: {run_dir / csv_name}")
        with (run_dir / "correctness_summary.csv").open(newline="", encoding="utf-8-sig") as stream:
            correctness = list(csv.DictReader(stream))
        mismatches = [row for row in correctness if row.get("exact_match") != "1"]
        if mismatches:
            raise RuntimeError(f"final correctness summary contains {len(mismatches)} mismatch rows")
        expected_correctness = len(workloads) * int(config["repeats"]) * 3
        if len(correctness) != expected_correctness:
            raise RuntimeError(
                f"correctness summary row count mismatch: {len(correctness)} vs {expected_correctness}"
            )
        with (run_dir / "performance_summary.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            performance_summary = list(csv.DictReader(stream))
        if len(performance_summary) != len(workloads) * 2:
            raise RuntimeError("performance summary does not cover every workload/variant")

        golden_paths = [Path(path) for path in config.get("golden_paths", [])]
        records, regenerated_performance, regenerated_correctness, regenerated_steps = (
            aggregate_performance(run_dir, config, model, golden_paths, False)
        )
        scheduler = aggregate_scheduler(run_dir, records, config, model)
        memory = aggregate_gpu_trace(run_dir)
        process_summary = aggregate_process_runs(run_dir)
        assert_csv_matches(
            run_dir / "performance_summary.csv", regenerated_performance, "performance summary"
        )
        assert_csv_matches(
            run_dir / "correctness_summary.csv", regenerated_correctness, "correctness summary"
        )
        assert_csv_matches(
            run_dir / "decode_step_summary.csv", regenerated_steps, "decode-step summary"
        )
        assert_csv_matches(run_dir / "scheduler_summary.csv", scheduler, "scheduler summary")
        assert_csv_matches(run_dir / "memory_summary.csv", memory, "memory summary")
        assert_csv_matches(run_dir / "process_summary.csv", process_summary, "process summary")
        scheduler_config = config_scheduler(config)
        expected_scheduler_rows = (
            len(scheduler_config["max_active_requests"])
            * len(scheduler_config["block_sizes"])
        )
        if len(scheduler) != expected_scheduler_rows:
            raise RuntimeError(
                f"scheduler summary must cover {expected_scheduler_rows} configurations, "
                f"found {len(scheduler)}"
            )

        artifact_manifest = load_json(run_dir / "artifact_manifest.json")
        if int(artifact_manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise RuntimeError("unsupported artifact manifest schema")
        artifacts = artifact_manifest.get("raw_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("artifact manifest is empty")
        for record in artifacts:
            path = run_dir / str(record.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
                raise RuntimeError(f"raw artifact missing or size changed: {path}")
            if sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"raw artifact hash changed: {path}")
        evidence = artifact_manifest.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise RuntimeError("artifact manifest lacks frozen top-level evidence hashes")
        for record in evidence:
            path = run_dir / str(record.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(record.get("size_bytes", -1)):
                raise RuntimeError(f"frozen evidence missing or size changed: {path}")
            if sha256_file(path) != record.get("sha256"):
                raise RuntimeError(f"frozen evidence hash changed: {path}")
    print(f"run bundle verified ({'final' if args.final else 'prepared'}): {run_dir}")


def add_common_run_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and aggregate immutable v0.9 Qwen benchmark evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write model/config manifests once")
    add_common_run_dir(prepare)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    prepare.add_argument("--model-revision", default="unknown")
    prepare.add_argument("--precision", default="fp16")
    prepare.add_argument("--model-dir", type=Path)
    prepare.add_argument("--model-spec", type=Path)
    prepare.add_argument("--baseline-engine", type=Path)
    prepare.add_argument("--prefill-engine", type=Path)
    prepare.add_argument("--decode-engine", type=Path)
    prepare.add_argument("--runtime-binary", type=Path, default=Path("build/minillm_rt"))
    prepare.add_argument(
        "--input-spec",
        action="append",
        default=[],
        metavar="CASE:PROMPT:OUTPUT:PATH",
    )
    prepare.add_argument("--golden", type=Path, action="append", default=[])
    prepare.add_argument("--warmup", type=int, default=5)
    prepare.add_argument("--repeats", type=int, default=10)
    prepare.add_argument("--scheduler-active", type=positive_int_csv, default=[1, 2, 4])
    prepare.add_argument("--scheduler-block-sizes", type=positive_int_csv, default=[8, 16, 32])
    prepare.add_argument("--scheduler-repeats", type=int, default=10)
    prepare.add_argument("--scheduler-request-count", type=int, default=9)
    prepare.add_argument(
        "--scheduler-case-ids",
        type=case_id_csv,
        default=["perf_p8", "perf_p32", "perf_p128", "perf_p256"] * 2
        + ["eos_instruction"],
    )
    prepare.add_argument(
        "--scheduler-output-limits",
        type=positive_int_sequence_csv,
        default=[8, 16, 32, 32, 16, 8, 32, 16, 16],
    )
    prepare.add_argument("--scheduler-pool-tokens", type=int, default=2048)
    prepare.add_argument("--notes", default="")
    prepare.add_argument("--runner-command", default="")
    prepare.add_argument(
        "--skip-artifact-hashes",
        action="store_true",
        help="Fast dry-run only; official runs should retain Engine hashes",
    )
    prepare.set_defaults(func=command_prepare)

    aggregate = subparsers.add_parser(
        "aggregate", help="validate raw A/B CSVs and write frozen summaries"
    )
    add_common_run_dir(aggregate)
    aggregate.add_argument("--golden", type=Path, action="append", default=[])
    aggregate.add_argument("--allow-incomplete-pairs", action="store_true")
    aggregate.add_argument("--allow-token-mismatch", action="store_true")
    aggregate.set_defaults(func=command_aggregate)

    verify = subparsers.add_parser("verify", help="verify a prepared or final run bundle")
    add_common_run_dir(verify)
    verify.add_argument("--final", action="store_true")
    verify.set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
