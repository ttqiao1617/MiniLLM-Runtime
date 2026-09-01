#!/usr/bin/env python3
"""Validate and aggregate immutable MiniLLM-RT v1.0 continuous-batching runs.

The benchmark configuration is the authoritative experiment matrix.  This
program deliberately validates the raw request, timeline, batch, process, log
and NVML evidence before writing any summary.  Generated files are immutable:
an experiment change requires a new run ID, never an in-place regeneration.
Final verification supports both full local external-artifact rehashing and a
published offline-evidence replay that explicitly skips those external files.
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
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
PROTOCOL = "minillm-rt-v10-continuous-batch-ab-v2"
VARIANTS = ("serial", "continuous")
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EXPECTED_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_MAX_BATCH_SIZE = 8
MIN_WARMUP_RUNS = 1
MIN_REPEATS = 10
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$")
CONFIG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,79}$")
EXPECTED_CONFIG_IDS = {
    "equal_n1", "equal_n2", "equal_n4", "equal_n8", "equal_n16",
    "mixed_a1", "mixed_a4", "mixed_a8", "staggered_a8", "correctness_long",
}

REQUEST_COLUMNS = {
    "request_id", "prompt_len", "output_len", "finish_reason",
    "queue_wait_ms", "estimated_ttft_ms", "mean_compute_tpot_ms",
    "p95_compute_tpot_ms", "end_to_end_ms", "kv_allocated_bytes",
    "block_count", "block_size", "reserved_tokens", "committed_tokens",
    "block_rounding_waste_tokens", "final_block_utilization",
    "block_alloc_ms", "block_release_ms", "block_table", "token_sequence",
    "arrival_iteration", "arrival_ms", "batch_slot",
}
TIMELINE_COLUMNS = {
    "event_index", "iteration", "request_id", "phase", "token_step",
    "token_id", "active_requests", "active_kv_bytes", "free_kv_blocks",
    "allocated_kv_blocks", "live_kv_tokens", "reserved_kv_tokens",
    "block_rounding_waste_tokens", "batch_id", "batch_slot", "batch_size",
    "max_past_length", "padding_tokens", "start_ms", "end_ms", "duration_ms",
}
BATCH_COLUMNS = {
    "batch_id", "iteration", "batch_size", "request_ids", "slot_ids",
    "past_lengths", "max_past_length", "padding_tokens", "padding_ratio",
    "input_prepare_ms", "shape_ms", "buffer_bind_ms", "kv_gather_ms", "h2d_ms",
    "enqueue_ms", "d2h_ms", "kv_scatter_ms", "sync_ms", "backend_total_ms",
    "batch_service_ms", "h2d_bytes", "d2h_bytes", "kv_gather_bytes",
    "kv_scatter_bytes", "kv_staging_zero_bytes", "batch_scratch_bytes",
    "batch_scratch_allocated_bytes", "buffer_allocation_count", "enqueue_before",
    "enqueue_after",
}
PROCESS_COLUMNS = {
    "sequence_index", "kind", "config_id", "variant", "repeat", "ab_order",
    "request_count", "max_active_requests", "total_requested_tokens",
    "process_wall_ms", "return_code", "started_at_utc", "completed_at_utc",
    "requests_csv", "timeline_csv", "batches_csv", "log", "command",
}
NVML_COLUMNS = {
    "monotonic_seconds", "phase", "timestamp", "name", "pstate",
    "temperature.gpu", "power.draw", "power.limit", "clocks.current.sm",
    "clocks.current.memory", "memory.used", "memory.total", "utilization.gpu",
    "utilization.memory",
}

RUN_METRICS = (
    "process_wall_ms",
    "makespan_ms",
    "aggregate_output_tokens_per_s",
    "decode_tokens_per_s",
    "mean_queue_wait_ms",
    "mean_visible_ttft_ms",
    "p95_visible_ttft_ms",
    "mean_visible_tpot_ms",
    "p95_visible_tpot_ms",
    "mean_request_compute_tpot_ms",
    "mean_request_p95_compute_tpot_ms",
    "mean_request_end_to_end_ms",
    "p50_request_end_to_end_ms",
    "p95_request_end_to_end_ms",
    "decode_enqueues",
    "decode_tokens_per_enqueue",
    "slot_reuse_events",
    "mean_batch_size",
    "p95_batch_size",
    "max_batch_size_observed",
    "mean_batch_capacity_utilization",
    "multi_request_batch_fraction",
    "padding_ratio",
    "total_kv_gather_ms",
    "total_kv_scatter_ms",
    "total_kv_gather_bytes",
    "total_kv_scatter_bytes",
    "total_kv_staging_zero_bytes",
    "mean_backend_total_ms",
    "mean_batch_service_ms",
    "peak_batch_scratch_bytes",
    "peak_batch_scratch_allocated_bytes",
    "peak_active_kv_bytes",
    "peak_allocated_kv_blocks",
)

BATCH_RUN_METRICS = (
    "batch_count", "decode_tokens", "mean_batch_size", "median_batch_size",
    "p95_batch_size", "max_batch_size", "mean_capacity_utilization",
    "multi_request_batch_fraction", "padding_tokens", "padding_ratio",
    "kv_gather_ms", "kv_scatter_ms", "kv_gather_bytes", "kv_scatter_bytes",
    "kv_staging_zero_bytes", "mean_backend_total_ms", "mean_batch_service_ms",
    "buffer_allocation_count",
    "peak_batch_scratch_bytes", "peak_batch_scratch_allocated_bytes",
    "decode_tokens_per_enqueue",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing JSON artifact: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path, required: set[str] | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV artifact: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise RuntimeError(f"duplicate CSV columns in {path}")
        if required and (missing := required.difference(fields)):
            raise RuntimeError(f"missing columns {sorted(missing)} in {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty CSV artifact: {path}")
    return rows


def finite_float(raw: Any, field: str, source: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid {field}={raw!r} in {source}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field}={raw!r} in {source}")
    return value


def nonnegative_float(raw: Any, field: str, source: Path) -> float:
    value = finite_float(raw, field, source)
    if value < 0.0:
        raise RuntimeError(f"negative {field}={raw!r} in {source}")
    return value


def exact_int(raw: Any, field: str, source: Path) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid integer {field}={raw!r} in {source}") from exc
    if str(raw).strip() != str(value):
        raise RuntimeError(f"non-canonical integer {field}={raw!r} in {source}")
    return value


def close_enough(actual: float, expected: float, *, absolute: float = 0.02,
                 relative: float = 1e-5) -> bool:
    return abs(actual - expected) <= max(absolute, abs(expected) * relative)


def split_ints(raw: str, field: str, source: Path) -> list[int]:
    if not raw.strip():
        return []
    try:
        return [int(value) for value in raw.split()]
    except ValueError as exc:
        raise RuntimeError(f"invalid integer list {field}={raw!r} in {source}") from exc


def write_text_immutable(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_csv_immutable(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty summary: {path}")
    fields = union_fields(rows)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json_immutable(path: Path, payload: Any) -> None:
    write_text_immutable(
        path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def union_fields(rows: Sequence[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        for field in row:
            if field not in result:
                result.append(field)
    return result


def csv_projection(rows: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    fields = union_fields(rows)
    return [
        {field: "" if row.get(field) is None else str(row.get(field, "")) for field in fields}
        for row in rows
    ]


def assert_csv_matches(path: Path, expected: Sequence[dict[str, Any]], label: str) -> None:
    actual = read_csv(path)
    fields = union_fields(expected)
    if set(actual[0]) != set(fields):
        raise RuntimeError(f"frozen {label} columns do not match regenerated evidence: {path}")
    if [{field: row.get(field, "") for field in fields} for row in actual] != csv_projection(expected):
        raise RuntimeError(f"frozen {label} does not match regenerated evidence: {path}")


def artifact_row(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def resolve_record_path(record: dict[str, Any]) -> Path:
    return Path(str(record.get("path", ""))).expanduser().resolve()


def validate_artifact_record(record: Any, label: str) -> Path:
    if not isinstance(record, dict) or not record.get("path"):
        raise RuntimeError(f"missing artifact record: {label}")
    path = resolve_record_path(record)
    if record.get("exists") is False or not path.is_file():
        raise RuntimeError(f"artifact does not exist: {label}: {path}")
    if "size_bytes" not in record or path.stat().st_size != int(record["size_bytes"]):
        raise RuntimeError(f"artifact size changed: {label}: {path}")
    expected_hash = str(record.get("sha256", ""))
    if not expected_hash or sha256_file(path) != expected_hash:
        raise RuntimeError(f"artifact hash changed or absent: {label}: {path}")
    return path


def validate_artifact_record_metadata(record: Any, label: str) -> Path:
    """Validate a frozen external-artifact identity without opening the file."""
    if not isinstance(record, dict) or not record.get("path"):
        raise RuntimeError(f"missing artifact record: {label}")
    path = resolve_record_path(record)
    try:
        size = int(record["size_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid artifact size metadata: {label}") from exc
    digest = str(record.get("sha256", ""))
    if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"invalid artifact identity metadata: {label}")
    return path


def resolve_published_path(path: Path, repo_root: Path | None) -> Path:
    """Resolve tracked inputs/configs after a repository is cloned elsewhere."""
    if repo_root is not None:
        parts = path.parts
        for anchor in ("inputs", "configs"):
            if anchor in parts:
                candidate = repo_root.joinpath(*parts[parts.index(anchor):])
                if candidate.is_file():
                    return candidate.resolve()
    return path.expanduser().resolve()


def validate_published_artifact_record(
    record: Any, label: str, repo_root: Path
) -> Path:
    original = validate_artifact_record_metadata(record, label)
    path = resolve_published_path(original, repo_root)
    if not path.is_file():
        raise RuntimeError(f"published artifact is unavailable: {label}: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise RuntimeError(f"published artifact size changed: {label}: {path}")
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"published artifact hash changed: {label}: {path}")
    return path


def parse_model_spec(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid ModelSpec line {number} in {path}")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def dtype_size(raw: str) -> int:
    normalized = raw.lower().replace("torch.", "")
    sizes = {
        "int64": 8, "int32": 4, "float32": 4, "fp32": 4,
        "float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2,
    }
    if normalized not in sizes:
        raise RuntimeError(f"unsupported ModelSpec dtype for byte audit: {raw!r}")
    return sizes[normalized]


def parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"invalid boolean value in ModelSpec: {raw!r}")


def validate_run_id(config: dict[str, Any], run_dir: Path) -> None:
    run_id = str(config.get("run_id", ""))
    if not RUN_ID_RE.fullmatch(run_id) or run_dir.name != run_id:
        raise RuntimeError(f"invalid or mismatched run ID {run_id!r} for {run_dir}")


def normalize_config(
    config: dict[str, Any], repo_root: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported benchmark schema: {config.get('schema_version')}")
    if config.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected benchmark protocol: {config.get('protocol')!r}")
    if config.get("variants") != list(VARIANTS):
        raise RuntimeError(f"variants must be exactly {list(VARIANTS)}")
    if config.get("alternating_ab_order") is not True:
        raise RuntimeError("benchmark must declare alternating_ab_order=true")
    try:
        warmup = int(config["warmup_runs"])
        warmup_generations = int(config["warmup_generations"])
        repeats = int(config["repeats"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid warmup_runs/repeats in benchmark config") from exc
    if warmup != MIN_WARMUP_RUNS or warmup_generations != 1 or repeats < MIN_REPEATS:
        raise RuntimeError(f"official v1.0 requires one representative warmup generation and repeats>={MIN_REPEATS}")
    if config.get("warmup_unit") != "representative_full_generation":
        raise RuntimeError("warmup_unit must be 'representative_full_generation'")
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise RuntimeError("benchmark config lacks execution contract")
    required_execution = {
        "provider": "tensorrt",
        "prefill": "serial",
        "serial_decode": "one_request_per_enqueue",
        "continuous_decode": "one_dynamic_batch_per_iteration",
        "comparison_engine_policy": "identical_prefill_and_decode_batch_engines",
    }
    for field, expected in required_execution.items():
        if execution.get(field) != expected:
            raise RuntimeError(f"execution.{field} must be {expected!r}")
    try:
        max_batch = int(execution["max_batch_size"])
        block_size = int(execution["block_size"])
        pool_tokens = int(execution["pool_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid execution batch/block/pool dimensions") from exc
    if max_batch != EXPECTED_MAX_BATCH_SIZE or block_size <= 0 \
            or pool_tokens <= 0 or pool_tokens % block_size:
        raise RuntimeError("invalid max batch size or KV block pool dimensions")

    raw_configs = config.get("configurations")
    if not isinstance(raw_configs, list) or not raw_configs:
        raise RuntimeError("benchmark config has no configurations")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    performance_count = 0
    correctness_count = 0
    for index, item in enumerate(raw_configs):
        if not isinstance(item, dict):
            raise RuntimeError(f"configuration {index} is not an object")
        config_id = str(item.get("config_id", ""))
        category = str(item.get("category", ""))
        if not CONFIG_ID_RE.fullmatch(config_id) or config_id in seen:
            raise RuntimeError(f"invalid or duplicate config_id: {config_id!r}")
        seen.add(config_id)
        try:
            request_count = int(item["request_count"])
            max_active = int(item["max_active_requests"])
            item_repeats = int(item["repeats"])
            case_ids = [str(value) for value in item["request_case_ids"]]
            input_paths = [str(value) for value in item["input_paths"]]
            prompts = [int(value) for value in item["prompt_lengths"]]
            limits = [int(value) for value in item["output_limits"]]
            arrivals = [int(value) for value in item["arrival_iterations"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid configuration fields for {config_id}") from exc
        arrays = (case_ids, input_paths, prompts, limits, arrivals)
        if request_count <= 0 or any(len(values) != request_count for values in arrays):
            raise RuntimeError(f"request arrays/count mismatch for {config_id}")
        if max_active <= 0 or max_active > min(request_count, max_batch):
            raise RuntimeError(f"invalid max_active_requests for {config_id}")
        if any(value <= 0 for value in prompts + limits) or any(value < 0 for value in arrivals):
            raise RuntimeError(f"invalid prompt/output/arrival values for {config_id}")
        if arrivals != sorted(arrivals):
            raise RuntimeError(f"arrival_iterations must be nondecreasing for {config_id}")
        is_correctness = category.lower().startswith("correctness")
        if is_correctness:
            correctness_count += 1
            if item_repeats != 1:
                raise RuntimeError(f"correctness configuration {config_id} must run exactly once")
        else:
            performance_count += 1
            if item_repeats != repeats or item_repeats < MIN_REPEATS:
                raise RuntimeError(f"performance configuration {config_id} lacks {repeats} repeats")
        for request_index, (path_raw, prompt) in enumerate(zip(input_paths, prompts)):
            path = resolve_published_path(
                Path(path_raw).expanduser().resolve(), repo_root
            )
            if not path.is_file():
                raise RuntimeError(f"missing input {request_index} for {config_id}: {path}")
            try:
                tokens = [int(value) for value in path.read_text(encoding="utf-8").split()]
            except ValueError as exc:
                raise RuntimeError(f"non-integer input IDs in {path}") from exc
            if len(tokens) != prompt or any(value < 0 for value in tokens):
                raise RuntimeError(f"input/prompt binding mismatch for {config_id} request {request_index}")
        normalized.append({
            **item,
            "config_id": config_id,
            "category": category,
            "request_count": request_count,
            "max_active_requests": max_active,
            "repeats": item_repeats,
            "request_case_ids": case_ids,
            "input_paths": input_paths,
            "prompt_lengths": prompts,
            "output_limits": limits,
            "arrival_iterations": arrivals,
        })
    if not performance_count or not correctness_count:
        raise RuntimeError("v1.0 matrix requires performance and long-correctness configurations")
    if seen != EXPECTED_CONFIG_IDS:
        raise RuntimeError(
            f"v1.0 configuration matrix mismatch: "
            f"missing={sorted(EXPECTED_CONFIG_IDS - seen)}, unexpected={sorted(seen - EXPECTED_CONFIG_IDS)}"
        )
    return {
        "warmup_runs": warmup,
        "warmup_generations": warmup_generations,
        "repeats": repeats,
        "max_batch_size": max_batch,
        "block_size": block_size,
        "pool_tokens": pool_tokens,
    }, normalized


def locate_artifact_record(payload: dict[str, Any], names: Sequence[str]) -> dict[str, Any] | None:
    """Find a named artifact record in common manifest layouts."""
    for name in names:
        value = payload.get(name)
        if isinstance(value, dict) and value.get("path"):
            return value
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for name in names:
            value = artifacts.get(name)
            if isinstance(value, dict) and value.get("path"):
                return value
    return None


def validate_manifests(
    run_dir: Path,
    offline_repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = load_json(run_dir / "benchmark_config.json")
    model = load_json(run_dir / "model_manifest.json")
    environment = load_json(run_dir / "environment.json")
    if not isinstance(config, dict) or not isinstance(model, dict) or not isinstance(environment, dict):
        raise RuntimeError("run manifests must contain JSON objects")
    validate_run_id(config, run_dir)
    if int(model.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported model manifest schema: {model.get('schema_version')}")
    if model.get("model_id") != EXPECTED_MODEL_ID \
            or model.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("model manifest does not match the pinned Qwen model/revision")
    if model.get("precision") != "fp16-strongly-typed":
        raise RuntimeError("model manifest does not declare the FP16 strongly-typed contract")
    if int(environment.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported environment schema: {environment.get('schema_version')}")
    protocol, configurations = normalize_config(config, offline_repo_root)

    runtime_record = locate_artifact_record(config, ("runtime_binary",))
    spec_record = locate_artifact_record(config, ("model_spec", "model_spec_artifact"))
    if spec_record is None:
        spec_record = locate_artifact_record(model, ("model_spec_artifact", "model_spec_file"))
    workload_record = locate_artifact_record(
        config, ("workloads_manifest", "workload_manifest", "workloads_artifact")
    )
    if offline_repo_root is None:
        validate_artifact_record(runtime_record, "runtime_binary")
        spec_path = validate_artifact_record(spec_record, "model_spec")
        validate_artifact_record(workload_record, "workloads_manifest")
    else:
        validate_artifact_record_metadata(runtime_record, "runtime_binary")
        spec_path = validate_published_artifact_record(
            spec_record, "model_spec", offline_repo_root
        )
        validate_published_artifact_record(
            workload_record, "workloads_manifest", offline_repo_root
        )

    engine_container = model.get("engines")
    if not isinstance(engine_container, dict):
        raise RuntimeError("model manifest lacks Engine artifacts")
    if not any("prefill" in str(name).lower() for name in engine_container):
        raise RuntimeError("model manifest lacks a prefill Engine")
    if not any("decode" in str(name).lower() for name in engine_container):
        raise RuntimeError("model manifest lacks a batch-decode Engine")
    engine_records: dict[str, dict[str, Any]] = {}
    for name, record in sorted(engine_container.items()):
        if offline_repo_root is None:
            validate_artifact_record(record, f"engine.{name}")
        else:
            validate_artifact_record_metadata(record, f"engine.{name}")
        engine_records[str(name)] = record
    if "prefill" not in engine_records or "decode_batch" not in engine_records:
        raise RuntimeError("model manifest must bind prefill and decode_batch Engines explicitly")

    manifest_records = model.get("engine_manifests")
    if not isinstance(manifest_records, list) or not manifest_records:
        raise RuntimeError("model manifest lacks hashed Engine build/inspection manifests")
    manifest_paths = [
        (
            validate_artifact_record(record, f"engine_manifest[{index}]")
            if offline_repo_root is None
            else validate_artifact_record_metadata(record, f"engine_manifest[{index}]")
        )
        for index, record in enumerate(manifest_records)
    ]
    decode_inspections = [
        path for path in manifest_paths
        if path.name == "qwen2_5_0_5b_decode_batch.engine.json"
    ]
    if len(decode_inspections) != 1:
        raise RuntimeError("exactly one decode_batch Engine inspection manifest is required")
    if offline_repo_root is None:
        decode_contract = load_json(decode_inspections[0])
        if not isinstance(decode_contract, dict) \
                or decode_contract.get("mode") != "decode_batch" \
                or int(decode_contract.get("max_batch_size", 0)) != EXPECTED_MAX_BATCH_SIZE \
                or int(decode_contract.get("max_sequence_length", 0)) != 512 \
                or decode_contract.get("precision_contract") != "strongly_typed_fp16_onnx":
            raise RuntimeError("decode_batch Engine inspection has the wrong semantic/profile contract")
        inspected_engine = decode_contract.get("engine")
        inspected_onnx = decode_contract.get("onnx")
        if not isinstance(inspected_engine, dict) or not isinstance(inspected_onnx, dict):
            raise RuntimeError("decode_batch Engine inspection lacks Engine/ONNX artifact records")
        bound_engine = engine_records["decode_batch"]
        if inspected_engine.get("sha256") != bound_engine.get("sha256") \
                or int(inspected_engine.get("size_bytes", -1)) != int(bound_engine.get("size_bytes", -2)):
            raise RuntimeError("decode_batch semantic inspection is not bound to the benchmark Engine")
        validate_artifact_record(inspected_onnx, "decode_batch.onnx")

    golden_records = config.get("golden_artifacts")
    if not isinstance(golden_records, list) or not golden_records:
        raise RuntimeError("benchmark config lacks golden_artifacts")
    for index, record in enumerate(golden_records):
        if offline_repo_root is None:
            validate_artifact_record(record, f"golden[{index}]")
        else:
            validate_published_artifact_record(
                record, f"golden[{index}]", offline_repo_root
            )

    if offline_repo_root is not None:
        registry = config.get("workload_registry")
        if not isinstance(registry, list) or not registry:
            raise RuntimeError("benchmark config lacks workload_registry")
        for index, workload in enumerate(registry):
            if not isinstance(workload, dict):
                raise RuntimeError(f"invalid workload_registry[{index}]")
            validate_published_artifact_record(
                workload.get("input_ids"),
                f"workload_registry[{index}].input_ids",
                offline_repo_root,
            )
            validate_published_artifact_record(
                workload.get("golden"),
                f"workload_registry[{index}].golden",
                offline_repo_root,
            )

    parsed_spec = parse_model_spec(spec_path)
    embedded_spec = model.get("model_spec")
    if isinstance(embedded_spec, dict) and embedded_spec and embedded_spec != parsed_spec:
        # JSON producers sometimes retain numeric/bool types.  Compare through strings.
        normalized_embedded = {str(key): str(value).lower() for key, value in embedded_spec.items()}
        normalized_parsed = {str(key): str(value).lower() for key, value in parsed_spec.items()}
        if normalized_embedded != normalized_parsed:
            raise RuntimeError("embedded ModelSpec differs from hashed ModelSpec file")

    required_spec = {
        "num_hidden_layers", "num_kv_heads", "head_dim", "vocab_size",
        "engine_max_sequence_length", "input_ids_dtype", "attention_mask_dtype",
        "position_ids_dtype", "logits_dtype", "kv_dtype", "requires_position_ids",
        "eos_token_ids", "kv_layout",
    }
    if missing := required_spec.difference(parsed_spec):
        raise RuntimeError(f"ModelSpec lacks fields needed for audit: {sorted(missing)}")
    if not parse_bool(parsed_spec["requires_position_ids"]):
        raise RuntimeError("mixed-length padded batching requires explicit per-row position_ids")
    if parsed_spec["kv_layout"].lower() != "bhsd":
        raise RuntimeError("v1.0 KV byte/layout audit requires the declared [B,H,S,D] layout")
    max_sequence = int(parsed_spec["engine_max_sequence_length"])
    for item in configurations:
        for prompt, limit in zip(item["prompt_lengths"], item["output_limits"]):
            if prompt + limit > max_sequence:
                raise RuntimeError(f"configuration {item['config_id']} exceeds Engine sequence profile")
    return config, model, environment, {**protocol, "model_spec": parsed_spec}, configurations


def load_goldens(
    config: dict[str, Any], repo_root: Path | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(config["golden_artifacts"]):
        recorded_path_text = str(record.get("path", ""))
        recorded_path = resolve_record_path(record)
        source_path = (
            validate_published_artifact_record(record, f"golden[{index}]", repo_root)
            if repo_root is not None else recorded_path
        )
        payload = load_json(source_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"golden must be an object: {source_path}")
        case_id = str(payload.get("case_id", ""))
        if not CONFIG_ID_RE.fullmatch(case_id) or case_id in result:
            raise RuntimeError(f"invalid or duplicate golden case_id {case_id!r}")
        try:
            prompt = int(payload["prompt_length"])
            requested = int(payload["requested_new_tokens"])
            generated = [int(value) for value in payload["generated_ids"]]
            input_ids = [int(value) for value in payload.get("input_ids", [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid golden payload: {source_path}") from exc
        if prompt <= 0 or requested <= 0 or not generated or len(generated) > requested:
            raise RuntimeError(f"invalid golden lengths: {source_path}")
        result[case_id] = {
            # Preserve the immutable path recorded in correctness_summary.csv;
            # source_path may be the portable path in a fresh clone.
            "path": recorded_path_text,
            "case_id": case_id,
            "prompt_length": prompt,
            "requested_new_tokens": requested,
            "generated_ids": generated,
            "input_ids": input_ids,
            "stopped_on_eos": bool(payload.get("stopped_on_eos", len(generated) < requested)),
        }
    if len(result) != len(config["golden_artifacts"]):
        raise RuntimeError(f"not every golden artifact was loaded ({len(result)}/{len(config['golden_artifacts'])})")
    return result


def expected_files(
    run_dir: Path, configurations: Sequence[dict[str, Any]]
) -> tuple[set[Path], set[Path], set[Path], dict[tuple[str, str, int], dict[str, Path]]]:
    csv_files: set[Path] = set()
    log_files: set[Path] = set()
    batch_files: set[Path] = set()
    matrix: dict[tuple[str, str, int], dict[str, Path]] = {}
    for item in configurations:
        config_id = item["config_id"]
        for repeat in range(1, int(item["repeats"]) + 1):
            for variant in VARIANTS:
                base = run_dir / "raw" / variant / f"{config_id}_r{repeat}"
                requests = base.with_name(base.name + "_requests.csv")
                timeline = base.with_name(base.name + "_timeline.csv")
                log = run_dir / "raw" / "logs" / f"{variant}_{config_id}_r{repeat}.log"
                paths = {"requests": requests, "timeline": timeline, "log": log}
                csv_files.update((requests, timeline))
                log_files.add(log)
                if variant == "continuous":
                    batches = base.with_name(base.name + "_batches.csv")
                    paths["batches"] = batches
                    csv_files.add(batches)
                    batch_files.add(batches)
                matrix[(variant, config_id, repeat)] = paths
    return csv_files, log_files, batch_files, matrix


def validate_raw_matrix(
    run_dir: Path, configurations: Sequence[dict[str, Any]]
) -> dict[tuple[str, str, int], dict[str, Path]]:
    expected_csv, expected_logs, _, matrix = expected_files(run_dir, configurations)
    discovered_csv: set[Path] = set()
    for variant in VARIANTS:
        directory = run_dir / "raw" / variant
        if directory.is_dir():
            discovered_csv.update(path.resolve() for path in directory.glob("*.csv"))
    expected_csv = {path.resolve() for path in expected_csv}
    missing_csv = sorted(str(path) for path in expected_csv.difference(discovered_csv))
    extra_csv = sorted(str(path) for path in discovered_csv.difference(expected_csv))
    if missing_csv or extra_csv:
        raise RuntimeError(
            f"raw CSV matrix mismatch: missing={missing_csv[:8]}, unexpected={extra_csv[:8]}"
        )
    log_dir = run_dir / "raw" / "logs"
    discovered_logs = {path.resolve() for path in log_dir.glob("*.log")} if log_dir.is_dir() else set()
    expected_logs = {path.resolve() for path in expected_logs}
    missing_logs = sorted(str(path) for path in expected_logs.difference(discovered_logs))
    extra_logs = sorted(str(path) for path in discovered_logs.difference(expected_logs))
    if missing_logs or extra_logs:
        raise RuntimeError(
            f"raw log matrix mismatch: missing={missing_logs[:8]}, unexpected={extra_logs[:8]}"
        )
    return matrix


def first_mismatch(reference: Sequence[int], candidate: Sequence[int]) -> int:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    return min(len(reference), len(candidate)) if len(reference) != len(candidate) else -1


def parse_runtime_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing runtime log: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s{2}([^:]+):\s*(.*?)\s*$", line)
        if match:
            fields[match.group(1).strip()] = match.group(2).strip()
    required = {
        "execution", "requests", "max active requests", "max batch size",
        "generated tokens", "decode tokens", "decode enqueues",
        "decode tokens/enqueue", "slot reuse events", "makespan ms",
        "aggregate tokens/s",
    }
    if missing := required.difference(fields):
        raise RuntimeError(f"runtime summary lacks {sorted(missing)} in {path}")
    result: dict[str, Any] = {"execution": fields["execution"]}
    for field in (
        "requests", "max active requests", "max batch size", "generated tokens",
        "decode tokens", "decode enqueues", "slot reuse events",
    ):
        result[field] = exact_int(fields[field], field, path)
    for field in ("decode tokens/enqueue", "makespan ms", "aggregate tokens/s"):
        result[field] = nonnegative_float(fields[field], field, path)
    return result


def expected_golden_for_request(
    golden: dict[str, Any], output_limit: int, eos_ids: set[int]
) -> tuple[list[int], str]:
    generated = list(golden["generated_ids"])
    if golden["stopped_on_eos"]:
        if not generated or generated[-1] not in eos_ids:
            raise RuntimeError(f"EOS golden does not end in a ModelSpec EOS: {golden['path']}")
        if any(token in eos_ids for token in generated[:-1]):
            raise RuntimeError(f"EOS golden contains a non-terminal EOS: {golden['path']}")
    expected = generated[:output_limit]
    finish = "eos" if expected and expected[-1] in eos_ids else "length"
    if finish == "length" and len(expected) != output_limit:
        raise RuntimeError(f"golden {golden['case_id']} is too short for output limit {output_limit}")
    return expected, finish


def resolved_csv_reference(raw: str, source: Path) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (source.parent.parent / path).resolve()
    else:
        path = path.resolve()
    return path


def validate_process_matrix(
    run_dir: Path,
    configurations: Sequence[dict[str, Any]],
    matrix: dict[tuple[str, str, int], dict[str, Path]],
) -> tuple[dict[tuple[str, str, int], dict[str, str]], list[dict[str, Any]]]:
    path = run_dir / "raw" / "process_runs.csv"
    rows = read_csv(path, PROCESS_COLUMNS)
    expected_keys = set(matrix)
    observed: dict[tuple[str, str, int], dict[str, str]] = {}
    sequence_indices: list[int] = []
    config_by_id = {str(item["config_id"]): item for item in configurations}
    for row in rows:
        variant = row["variant"]
        config_id = row["config_id"]
        repeat = exact_int(row["repeat"], "repeat", path)
        key = (variant, config_id, repeat)
        if key not in expected_keys or key in observed:
            raise RuntimeError(f"unexpected or duplicate process row {key} in {path}")
        item = config_by_id[config_id]
        expected_kind = "correctness" if item["category"].lower().startswith("correctness") else "performance"
        if row["kind"] != expected_kind:
            raise RuntimeError(f"process kind mismatch for {key}: {row['kind']!r}")
        expected_ints = {
            "request_count": item["request_count"],
            "max_active_requests": item["max_active_requests"],
            "total_requested_tokens": sum(item["output_limits"]),
            "return_code": 0,
        }
        for field, expected in expected_ints.items():
            if exact_int(row[field], field, path) != expected:
                raise RuntimeError(f"process {field} mismatch for {key}")
        if nonnegative_float(row["process_wall_ms"], "process_wall_ms", path) <= 0.0:
            raise RuntimeError(f"process wall time must be positive for {key}")
        if not row["started_at_utc"].strip() or not row["completed_at_utc"].strip():
            raise RuntimeError(f"process timestamps are absent for {key}")
        if not row["command"].strip() or not row["ab_order"].strip():
            raise RuntimeError(f"process command/A-B order is absent for {key}")
        references = {
            "requests_csv": matrix[key]["requests"],
            "timeline_csv": matrix[key]["timeline"],
            "log": matrix[key]["log"],
        }
        for field, expected_path in references.items():
            actual_path = resolved_csv_reference(row[field], path)
            if actual_path != expected_path.resolve():
                raise RuntimeError(f"process {field} binding mismatch for {key}: {actual_path}")
        batches_path = resolved_csv_reference(row["batches_csv"], path)
        expected_batch = matrix[key].get("batches")
        if (batches_path is None) != (expected_batch is None) or (
            batches_path is not None and batches_path != expected_batch.resolve()
        ):
            raise RuntimeError(f"process batches_csv binding mismatch for {key}")
        sequence_indices.append(exact_int(row["sequence_index"], "sequence_index", path))
        observed[key] = row
    if set(observed) != expected_keys:
        missing = sorted(expected_keys.difference(observed))
        raise RuntimeError(f"process matrix is incomplete: {missing[:10]}")
    if len(sequence_indices) != len(set(sequence_indices)):
        raise RuntimeError("process sequence_index values are not unique")
    ordered_indices = sorted(sequence_indices)
    if ordered_indices not in (
        list(range(len(rows))), list(range(1, len(rows) + 1))
    ):
        raise RuntimeError("process sequence_index values must be contiguous")
    performance_items = [
        item for item in configurations
        if not item["category"].lower().startswith("correctness")
    ]
    correctness_items = [
        item for item in configurations
        if item["category"].lower().startswith("correctness")
    ]
    expected_sequence: list[tuple[str, str, int]] = []
    for item in correctness_items:
        expected_sequence.extend(
            (variant, item["config_id"], 1) for variant in VARIANTS
        )
    for repeat in range(1, int(performance_items[0]["repeats"]) + 1):
        offset = (repeat - 1) % len(performance_items)
        ordered_items = performance_items[offset:] + performance_items[:offset]
        if repeat % 2 == 0:
            ordered_items.reverse()
        variant_order = VARIANTS if repeat % 2 == 1 else tuple(reversed(VARIANTS))
        for item in ordered_items:
            expected_sequence.extend(
                (variant, item["config_id"], repeat) for variant in variant_order
            )
    actual_sequence = sorted(
        observed, key=lambda key: int(observed[key]["sequence_index"])
    )
    if actual_sequence != expected_sequence:
        mismatch = next(
            (
                index for index, pair in enumerate(zip(actual_sequence, expected_sequence))
                if pair[0] != pair[1]
            ),
            min(len(actual_sequence), len(expected_sequence)),
        )
        raise RuntimeError(f"A/B process schedule differs from the protocol at index {mismatch}")
    for item in configurations:
        for repeat in range(1, int(item["repeats"]) + 1):
            serial = observed[("serial", item["config_id"], repeat)]
            continuous = observed[("continuous", item["config_id"], repeat)]
            serial_index = int(serial["sequence_index"])
            continuous_index = int(continuous["sequence_index"])
            if serial["ab_order"] != continuous["ab_order"]:
                raise RuntimeError(f"A/B order labels disagree for {item['config_id']}/r{repeat}")
            serial_first = serial_index < continuous_index
            if serial_first != (repeat % 2 == 1):
                raise RuntimeError(f"A/B execution was not alternated for {item['config_id']}/r{repeat}")

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for key, row in observed.items():
        item = config_by_id[key[1]]
        grouped[(row["kind"], key[1], key[0])].append(float(row["process_wall_ms"]))
    summary: list[dict[str, Any]] = []
    for (kind, config_id, variant), values in sorted(grouped.items()):
        summary.append({
            "kind": kind,
            "config_id": config_id,
            "variant": variant,
            "runs": len(values),
            "mean_process_wall_ms": f"{statistics.mean(values):.6f}",
            "median_process_wall_ms": f"{statistics.median(values):.6f}",
            "stdev_process_wall_ms": f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}",
            "p95_process_wall_ms": f"{percentile(values, 95):.6f}",
        })
    return observed, summary


def validate_request_and_timeline(
    *,
    variant: str,
    item: dict[str, Any],
    repeat: int,
    paths: dict[str, Path],
    protocol: dict[str, Any],
    goldens: dict[str, dict[str, Any]],
    process_row: dict[str, str],
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    request_path = paths["requests"]
    timeline_path = paths["timeline"]
    requests = read_csv(request_path, REQUEST_COLUMNS)
    timeline = read_csv(timeline_path, TIMELINE_COLUMNS)
    if len(requests) != item["request_count"]:
        raise RuntimeError(f"request count mismatch in {request_path}")
    expected_ids = [f"request_{index}" for index in range(item["request_count"])]
    if [row["request_id"] for row in requests] != expected_ids:
        raise RuntimeError(f"request IDs/order changed in {request_path}")

    spec = protocol["model_spec"]
    eos_ids = {int(value.strip()) for value in spec["eos_token_ids"].split(",")}
    if not eos_ids:
        raise RuntimeError("ModelSpec has no EOS token IDs")
    layers = int(spec["num_hidden_layers"])
    kv_heads = int(spec["num_kv_heads"])
    head_dim = int(spec["head_dim"])
    kv_size = dtype_size(spec["kv_dtype"])
    max_sequence = int(spec["engine_max_sequence_length"])
    block_size = int(protocol["block_size"])
    pool_blocks = int(protocol["pool_tokens"]) // block_size

    event_indices = [exact_int(row["event_index"], "event_index", timeline_path) for row in timeline]
    if event_indices != list(range(len(timeline))):
        raise RuntimeError(f"timeline event_index is not contiguous in {timeline_path}")
    iterations = [exact_int(row["iteration"], "iteration", timeline_path) for row in timeline]
    if iterations != sorted(iterations):
        raise RuntimeError(f"timeline iterations are not monotonic in {timeline_path}")
    last_start = -1.0
    for row in timeline:
        if row["request_id"] not in set(expected_ids):
            raise RuntimeError(f"unknown request_id in {timeline_path}: {row['request_id']}")
        if row["phase"] not in {"prefill", "decode", "release"}:
            raise RuntimeError(f"invalid timeline phase in {timeline_path}: {row['phase']}")
        start = nonnegative_float(row["start_ms"], "start_ms", timeline_path)
        end = nonnegative_float(row["end_ms"], "end_ms", timeline_path)
        duration = nonnegative_float(row["duration_ms"], "duration_ms", timeline_path)
        if end + 1e-9 < start or not close_enough(duration, end - start, absolute=0.01):
            raise RuntimeError(f"timeline duration mismatch in {timeline_path}")
        if start + 0.02 < last_start:
            raise RuntimeError(f"timeline timestamps move backwards in {timeline_path}")
        last_start = max(last_start, start)
        for field in (
            "active_requests", "active_kv_bytes", "free_kv_blocks",
            "allocated_kv_blocks", "live_kv_tokens", "reserved_kv_tokens",
            "block_rounding_waste_tokens", "batch_slot", "batch_size",
            "max_past_length", "padding_tokens",
        ):
            value = exact_int(row[field], field, timeline_path)
            if field != "batch_slot" and value < 0:
                raise RuntimeError(f"negative {field} in {timeline_path}")
        free_blocks = int(row["free_kv_blocks"])
        allocated_blocks = int(row["allocated_kv_blocks"])
        if free_blocks + allocated_blocks != pool_blocks:
            raise RuntimeError(f"KV block-pool conservation failed in {timeline_path}")
        if int(row["active_requests"]) > item["max_active_requests"]:
            raise RuntimeError(f"active request limit exceeded in {timeline_path}")

    event_by_request: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in timeline:
        event_by_request[row["request_id"]].append(row)
    correctness: list[dict[str, Any]] = []
    visible_ttfts: list[float] = []
    visible_tpots: list[float] = []
    compute_tpots: list[float] = []
    compute_p95s: list[float] = []
    end_to_ends: list[float] = []
    queue_waits: list[float] = []
    expected_outputs: dict[str, list[int]] = {}
    request_metadata: dict[str, dict[str, Any]] = {}
    slot_intervals: list[tuple[int, int, int, str, set[int]]] = []

    for index, row in enumerate(requests):
        request_id = row["request_id"]
        case_id = item["request_case_ids"][index]
        if case_id not in goldens:
            raise RuntimeError(f"configuration references missing golden case {case_id}")
        golden = goldens[case_id]
        prompt = item["prompt_lengths"][index]
        limit = item["output_limits"][index]
        arrival_iteration = item["arrival_iterations"][index]
        input_path = resolve_published_path(
            Path(item["input_paths"][index]).expanduser().resolve(), repo_root
        )
        input_ids = [int(value) for value in input_path.read_text(encoding="utf-8").split()]
        if golden["prompt_length"] != prompt:
            raise RuntimeError(f"golden prompt mismatch for {case_id} in {request_path}")
        if golden["input_ids"] and golden["input_ids"] != input_ids:
            raise RuntimeError(f"golden/input IDs mismatch for {case_id} in {request_path}")
        expected, expected_finish = expected_golden_for_request(golden, limit, eos_ids)
        candidate = split_ints(row["token_sequence"], "token_sequence", request_path)
        output = exact_int(row["output_len"], "output_len", request_path)
        if exact_int(row["prompt_len"], "prompt_len", request_path) != prompt:
            raise RuntimeError(f"request prompt mismatch for {request_id} in {request_path}")
        if output != len(candidate) or output != len(expected):
            raise RuntimeError(f"request output length mismatch for {request_id} in {request_path}")
        mismatch = first_mismatch(expected, candidate)
        correctness.append({
            "config_id": item["config_id"],
            "category": item["category"],
            "variant": variant,
            "repeat": repeat,
            "request_id": request_id,
            "case_id": case_id,
            "prompt_len": prompt,
            "output_limit": limit,
            "reference_tokens": len(expected),
            "candidate_tokens": len(candidate),
            "finish_reason": row["finish_reason"],
            "expected_finish_reason": expected_finish,
            "exact_match": 1 if mismatch < 0 else 0,
            "first_mismatch_step": mismatch,
            "reference_sequence": " ".join(str(token) for token in expected),
            "candidate_sequence": " ".join(str(token) for token in candidate),
            "golden_path": str(golden["path"]),
        })
        if mismatch >= 0:
            raise RuntimeError(
                f"HF token gate failed for {variant}/{item['config_id']}/r{repeat}/{request_id} at {mismatch}"
            )
        if row["finish_reason"] != expected_finish:
            raise RuntimeError(f"finish reason mismatch for {request_id} in {request_path}")
        if expected_finish == "eos" and (not candidate or candidate[-1] not in eos_ids):
            raise RuntimeError(f"EOS finish lacks terminal EOS for {request_id} in {request_path}")
        if any(token in eos_ids for token in candidate[:-1]):
            raise RuntimeError(f"generation continued after EOS for {request_id} in {request_path}")
        if exact_int(row["arrival_iteration"], "arrival_iteration", request_path) != arrival_iteration:
            raise RuntimeError(f"arrival iteration mismatch for {request_id} in {request_path}")
        arrival_ms = nonnegative_float(row["arrival_ms"], "arrival_ms", request_path)
        queue_wait = nonnegative_float(row["queue_wait_ms"], "queue_wait_ms", request_path)
        queue_waits.append(queue_wait)
        for field in (
            "estimated_ttft_ms", "mean_compute_tpot_ms", "p95_compute_tpot_ms",
            "end_to_end_ms", "block_alloc_ms", "block_release_ms",
        ):
            nonnegative_float(row[field], field, request_path)
        compute_tpots.append(float(row["mean_compute_tpot_ms"]))
        compute_p95s.append(float(row["p95_compute_tpot_ms"]))
        end_to_ends.append(float(row["end_to_end_ms"]))
        slot = exact_int(row["batch_slot"], "batch_slot", request_path)
        if slot < 0 or slot >= item["max_active_requests"]:
            raise RuntimeError(f"batch slot outside configured pool for {request_id} in {request_path}")
        if exact_int(row["block_size"], "block_size", request_path) != block_size:
            raise RuntimeError(f"block size mismatch for {request_id} in {request_path}")
        committed = exact_int(row["committed_tokens"], "committed_tokens", request_path)
        if committed != prompt + limit:
            raise RuntimeError(f"committed-token mismatch for {request_id} in {request_path}")
        block_count = exact_int(row["block_count"], "block_count", request_path)
        blocks = split_ints(row["block_table"], "block_table", request_path)
        expected_blocks = (committed + block_size - 1) // block_size
        reserved = exact_int(row["reserved_tokens"], "reserved_tokens", request_path)
        waste = exact_int(row["block_rounding_waste_tokens"], "block_rounding_waste_tokens", request_path)
        if block_count != expected_blocks or len(blocks) != expected_blocks or len(blocks) != len(set(blocks)):
            raise RuntimeError(f"block table/count mismatch for {request_id} in {request_path}")
        if any(block < 0 or block >= pool_blocks for block in blocks):
            raise RuntimeError(f"block ID outside KV pool for {request_id} in {request_path}")
        if reserved != block_count * block_size or waste != reserved - committed:
            raise RuntimeError(f"block reservation accounting mismatch for {request_id} in {request_path}")
        capacity = min(reserved, max_sequence)
        expected_kv_bytes = 2 * layers * 2 * kv_heads * capacity * head_dim * kv_size
        if exact_int(row["kv_allocated_bytes"], "kv_allocated_bytes", request_path) != expected_kv_bytes:
            raise RuntimeError(f"managed KV byte formula failed for {request_id} in {request_path}")
        utilization = finite_float(row["final_block_utilization"], "final_block_utilization", request_path)
        if not close_enough(utilization, (prompt + output) / reserved, absolute=1e-5):
            raise RuntimeError(f"final block utilization mismatch for {request_id} in {request_path}")

        events = event_by_request.get(request_id, [])
        token_events = [event for event in events if event["phase"] in {"prefill", "decode"}]
        releases = [event for event in events if event["phase"] == "release"]
        if len(token_events) != output or len(releases) != 1 or not token_events:
            raise RuntimeError(f"token/release event count mismatch for {request_id} in {timeline_path}")
        if [int(event["token_step"]) for event in token_events] != list(range(output)):
            raise RuntimeError(f"token steps are not contiguous for {request_id} in {timeline_path}")
        if [int(event["token_id"]) for event in token_events] != candidate:
            raise RuntimeError(f"timeline token sequence mismatch for {request_id} in {timeline_path}")
        if token_events[0]["phase"] != "prefill" or any(
            event["phase"] != "decode" for event in token_events[1:]
        ):
            raise RuntimeError(f"prefill/decode event ordering mismatch for {request_id}")
        if any(int(event["batch_slot"]) != slot for event in events):
            raise RuntimeError(f"request-to-slot mapping changed for {request_id} in {timeline_path}")
        ready = [float(event["end_ms"]) for event in token_events]
        visible_ttft = ready[0] - arrival_ms
        if visible_ttft < -0.02:
            raise RuntimeError(f"negative visible TTFT for {request_id} in {timeline_path}")
        visible_ttfts.append(max(0.0, visible_ttft))
        visible_tpots.extend(later - earlier for earlier, later in zip(ready, ready[1:]))
        if any(value < -0.02 for value in visible_tpots):
            raise RuntimeError(f"negative visible TPOT in {timeline_path}")
        if not close_enough(float(token_events[0]["start_ms"]) - arrival_ms, queue_wait, absolute=0.05):
            raise RuntimeError(f"queue-wait/timeline mismatch for {request_id} in {request_path}")
        if not close_enough(ready[-1] - arrival_ms, float(row["end_to_end_ms"]), absolute=0.05):
            raise RuntimeError(f"completion/timeline mismatch for {request_id} in {request_path}")
        release = releases[0]
        if float(release["start_ms"]) + 0.02 < ready[-1]:
            raise RuntimeError(f"request released before its last token for {request_id}")
        prefill_index = int(token_events[0]["event_index"])
        release_index = int(release["event_index"])
        slot_intervals.append((prefill_index, release_index, slot, request_id, set(blocks)))
        expected_outputs[request_id] = expected
        request_metadata[request_id] = {
            "prompt": prompt, "output": output, "slot": slot,
            "kv_bytes": expected_kv_bytes,
            "block_count": block_count,
            "reserved": reserved,
            "committed": committed,
            "rounding_waste": waste,
            "token_events": token_events, "release": release,
        }

    # Replay request lifetime transitions and recompute every scheduler resource
    # snapshot.  Batched decode updates all member live-token counts before any
    # per-request timeline row is emitted, so apply each batch atomically.
    live: dict[str, dict[str, int]] = {}
    applied_batches: set[int] = set()
    decode_rows_by_batch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for event in timeline:
        if event["phase"] == "decode" and int(event["batch_id"]) >= 0:
            decode_rows_by_batch[int(event["batch_id"])].append(event)
    for event in timeline:
        request_id = event["request_id"]
        metadata = request_metadata[request_id]
        phase = event["phase"]
        if phase == "prefill":
            if request_id in live:
                raise RuntimeError(f"duplicate live request {request_id} in {timeline_path}")
            live[request_id] = {
                "live_tokens": metadata["prompt"] + 1,
                "kv_bytes": metadata["kv_bytes"],
                "block_count": metadata["block_count"],
                "reserved": metadata["reserved"],
                "committed": metadata["committed"],
                "rounding_waste": metadata["rounding_waste"],
            }
        elif phase == "decode":
            batch_id = int(event["batch_id"])
            updates = [event] if batch_id < 0 else decode_rows_by_batch[batch_id]
            if batch_id < 0 or batch_id not in applied_batches:
                for update in updates:
                    member = update["request_id"]
                    if member not in live:
                        raise RuntimeError(f"decode of non-live request {member} in {timeline_path}")
                    member_metadata = request_metadata[member]
                    live[member]["live_tokens"] = (
                        member_metadata["prompt"] + int(update["token_step"]) + 1
                    )
                if batch_id >= 0:
                    applied_batches.add(batch_id)
        else:
            if request_id not in live:
                raise RuntimeError(f"release of non-live request {request_id} in {timeline_path}")
            del live[request_id]

        expected_resources = {
            "active_requests": len(live),
            "active_kv_bytes": sum(value["kv_bytes"] for value in live.values()),
            "allocated_kv_blocks": sum(value["block_count"] for value in live.values()),
            "live_kv_tokens": sum(value["live_tokens"] for value in live.values()),
            "reserved_kv_tokens": sum(value["reserved"] for value in live.values()),
            "block_rounding_waste_tokens": sum(
                value["rounding_waste"] for value in live.values()
            ),
        }
        expected_resources["free_kv_blocks"] = (
            pool_blocks - expected_resources["allocated_kv_blocks"]
        )
        for field, expected_value in expected_resources.items():
            if int(event[field]) != expected_value:
                raise RuntimeError(
                    f"scheduler resource snapshot mismatch ({field}) in {timeline_path}: "
                    f"{event[field]} vs {expected_value}"
                )
    if live:
        raise RuntimeError(f"scheduler lifetime replay ended with live requests in {timeline_path}")

    for left_index, left in enumerate(slot_intervals):
        for right in slot_intervals[left_index + 1:]:
            overlap = left[0] < right[1] and right[0] < left[1]
            if overlap and left[2] == right[2]:
                raise RuntimeError(f"active requests share batch slot {left[2]} in {timeline_path}")
            if overlap and left[4].intersection(right[4]):
                raise RuntimeError(f"active requests share KV blocks in {timeline_path}")

    release_events = [row for row in timeline if row["phase"] == "release"]
    if len(release_events) != item["request_count"]:
        raise RuntimeError(f"one release per request is required in {timeline_path}")
    last = timeline[-1]
    for field in (
        "active_requests", "active_kv_bytes", "allocated_kv_blocks",
        "live_kv_tokens", "reserved_kv_tokens", "block_rounding_waste_tokens",
    ):
        if exact_int(last[field], field, timeline_path) != 0:
            raise RuntimeError(f"final scheduler resources are nonzero ({field}) in {timeline_path}")
    if int(last["free_kv_blocks"]) != pool_blocks:
        raise RuntimeError(f"final KV block pool is not fully free in {timeline_path}")

    slot_counts: dict[int, int] = defaultdict(int)
    for _, _, slot, _, _ in slot_intervals:
        slot_counts[slot] += 1
    derived_slot_reuse = sum(max(0, count - 1) for count in slot_counts.values())
    if item["request_count"] > item["max_active_requests"] and derived_slot_reuse <= 0:
        raise RuntimeError(f"required slot reuse was not observed in {timeline_path}")

    log = parse_runtime_log(paths["log"])
    expected_execution = "continuous_batch" if variant == "continuous" else "serial_round_robin"
    if log["execution"] != expected_execution:
        raise RuntimeError(f"execution mode mismatch in {paths['log']}")
    if log["requests"] != item["request_count"] or log["max active requests"] != item["max_active_requests"]:
        raise RuntimeError(f"runtime request/active summary mismatch in {paths['log']}")
    if log["max batch size"] != (protocol["max_batch_size"] if variant == "continuous" else 1):
        raise RuntimeError(f"runtime max batch summary mismatch in {paths['log']}")
    generated_tokens = sum(len(sequence) for sequence in expected_outputs.values())
    decode_events = [row for row in timeline if row["phase"] == "decode"]
    decode_rounds: dict[int, list[str]] = defaultdict(list)
    for event in decode_events:
        event_iteration = exact_int(event["iteration"], "iteration", timeline_path)
        request_id = event["request_id"]
        if request_id in decode_rounds[event_iteration]:
            raise RuntimeError(
                f"request {request_id} decoded more than once in scheduler round "
                f"{event_iteration}: {timeline_path}"
            )
        decode_rounds[event_iteration].append(request_id)
    decode_round_membership = tuple(
        (event_iteration, tuple(sorted(request_ids)))
        for event_iteration, request_ids in sorted(decode_rounds.items())
    )
    if log["generated tokens"] != generated_tokens or log["decode tokens"] != len(decode_events):
        raise RuntimeError(f"runtime token totals mismatch in {paths['log']}")
    if log["slot reuse events"] != derived_slot_reuse:
        raise RuntimeError(f"slot-reuse log/evidence mismatch in {paths['log']}")
    expected_ratio = len(decode_events) / log["decode enqueues"] if log["decode enqueues"] else 0.0
    if not close_enough(log["decode tokens/enqueue"], expected_ratio, absolute=1e-4):
        raise RuntimeError(f"tokens/enqueue summary mismatch in {paths['log']}")

    batch_metrics = validate_batch_evidence(
        variant=variant,
        item=item,
        repeat=repeat,
        paths=paths,
        protocol=protocol,
        timeline=timeline,
        request_metadata=request_metadata,
        decode_events=decode_events,
        log=log,
    )
    if not close_enough(
        log["aggregate tokens/s"], generated_tokens / (log["makespan ms"] / 1000.0),
        absolute=0.02,
    ):
        raise RuntimeError(f"runtime throughput formula mismatch in {paths['log']}")
    if not close_enough(
        float(process_row["process_wall_ms"]), float(process_row["process_wall_ms"]), absolute=0.0
    ):
        raise AssertionError("unreachable non-finite process wall time")

    run_record = {
        "config_id": item["config_id"],
        "category": item["category"],
        "variant": variant,
        "repeat": repeat,
        "request_count": item["request_count"],
        "max_active_requests": item["max_active_requests"],
        "requested_output_tokens": sum(item["output_limits"]),
        "actual_output_tokens": generated_tokens,
        "decode_tokens": len(decode_events),
        "process_wall_ms": float(process_row["process_wall_ms"]),
        "makespan_ms": log["makespan ms"],
        "aggregate_output_tokens_per_s": generated_tokens / (log["makespan ms"] / 1000.0),
        "decode_tokens_per_s": len(decode_events) / (log["makespan ms"] / 1000.0),
        "mean_queue_wait_ms": statistics.mean(queue_waits),
        "mean_visible_ttft_ms": statistics.mean(visible_ttfts),
        "p95_visible_ttft_ms": percentile(visible_ttfts, 95),
        "mean_visible_tpot_ms": statistics.mean(visible_tpots) if visible_tpots else 0.0,
        "p95_visible_tpot_ms": percentile(visible_tpots, 95),
        "mean_request_compute_tpot_ms": statistics.mean(compute_tpots),
        "mean_request_p95_compute_tpot_ms": statistics.mean(compute_p95s),
        "mean_request_end_to_end_ms": statistics.mean(end_to_ends),
        "p50_request_end_to_end_ms": percentile(end_to_ends, 50),
        "p95_request_end_to_end_ms": percentile(end_to_ends, 95),
        "decode_enqueues": log["decode enqueues"],
        "decode_tokens_per_enqueue": expected_ratio,
        "slot_reuse_events": derived_slot_reuse,
        "peak_active_kv_bytes": max(int(row["active_kv_bytes"]) for row in timeline),
        "peak_allocated_kv_blocks": max(int(row["allocated_kv_blocks"]) for row in timeline),
        **batch_metrics,
    }
    auxiliary = {
        "batch_run": batch_metrics.get("batch_run"),
        "observed_multi_batch": bool(batch_metrics.get("max_batch_size_observed", 0) > 1),
        "observed_slot_reuse": derived_slot_reuse > 0,
        "observed_membership_churn": bool(batch_metrics.get("membership_churn", False)),
        "decode_round_membership": decode_round_membership,
    }
    run_record.pop("batch_run", None)
    run_record.pop("membership_churn", None)
    return run_record, correctness, auxiliary


def validate_batch_evidence(
    *,
    variant: str,
    item: dict[str, Any],
    repeat: int,
    paths: dict[str, Path],
    protocol: dict[str, Any],
    timeline: Sequence[dict[str, str]],
    request_metadata: dict[str, dict[str, Any]],
    decode_events: Sequence[dict[str, str]],
    log: dict[str, Any],
) -> dict[str, Any]:
    zero_metrics: dict[str, Any] = {
        "mean_batch_size": 1.0 if decode_events else 0.0,
        "p95_batch_size": 1.0 if decode_events else 0.0,
        "max_batch_size_observed": 1 if decode_events else 0,
        "mean_batch_capacity_utilization": 1.0 if decode_events else 0.0,
        "multi_request_batch_fraction": 0.0,
        "padding_ratio": 0.0,
        "total_kv_gather_ms": 0.0,
        "total_kv_scatter_ms": 0.0,
        "total_kv_gather_bytes": 0,
        "total_kv_scatter_bytes": 0,
        "total_kv_staging_zero_bytes": 0,
        "mean_backend_total_ms": 0.0,
        "mean_batch_service_ms": 0.0,
        "peak_batch_scratch_bytes": 0,
        "peak_batch_scratch_allocated_bytes": 0,
        "batch_run": None,
        "membership_churn": False,
    }
    if variant == "serial":
        if "batches" in paths or any(path.name.endswith("_batches.csv") for path in paths.values()):
            raise RuntimeError(f"serial run unexpectedly declares batch evidence: {paths}")
        for event in decode_events:
            if int(event["batch_id"]) != -1 or int(event["batch_size"]) != 1:
                raise RuntimeError(f"serial decode contains batch metadata in {paths['timeline']}")
            if int(event["padding_tokens"]) != 0:
                raise RuntimeError(f"serial decode contains padding in {paths['timeline']}")
            request = request_metadata[event["request_id"]]
            expected_past = request["prompt"] + int(event["token_step"]) - 1
            if int(event["max_past_length"]) != expected_past:
                raise RuntimeError(f"serial past-length evidence mismatch in {paths['timeline']}")
        if log["decode enqueues"] != len(decode_events):
            raise RuntimeError(
                f"serial decode-token/enqueue invariant failed in {paths['log']}: "
                f"{len(decode_events)} tokens vs {log['decode enqueues']} enqueues"
            )
        return zero_metrics

    batch_path = paths.get("batches")
    if batch_path is None:
        raise RuntimeError("continuous run lacks a batch CSV path")
    batches = read_csv(batch_path, BATCH_COLUMNS)
    batch_ids = [exact_int(row["batch_id"], "batch_id", batch_path) for row in batches]
    if batch_ids != list(range(len(batches))):
        raise RuntimeError(f"batch_id is not contiguous in {batch_path}")
    spec = protocol["model_spec"]
    layers = int(spec["num_hidden_layers"])
    kv_heads = int(spec["num_kv_heads"])
    head_dim = int(spec["head_dim"])
    vocab = int(spec["vocab_size"])
    kv_element = dtype_size(spec["kv_dtype"])
    input_element = dtype_size(spec["input_ids_dtype"])
    mask_element = dtype_size(spec["attention_mask_dtype"])
    position_element = dtype_size(spec["position_ids_dtype"])
    logits_element = dtype_size(spec["logits_dtype"])
    requires_positions = parse_bool(spec["requires_position_ids"])
    tensor_count = layers * 2
    max_batch_capacity = int(protocol["max_batch_size"])
    max_sequence = int(spec["engine_max_sequence_length"])
    workspace_batch_capacity = min(
        item["request_count"], item["max_active_requests"], max_batch_capacity
    )
    expected_allocated_scratch = workspace_batch_capacity * (
        input_element
        + max_sequence * mask_element
        + (position_element if requires_positions else 0)
        + vocab * logits_element
        + tensor_count * (
            kv_heads * (max_sequence - 1) * head_dim * kv_element
            + kv_heads * max_sequence * head_dim * kv_element
        )
    )

    timeline_by_batch: dict[int, list[dict[str, str]]] = defaultdict(list)
    for event in decode_events:
        batch_id = exact_int(event["batch_id"], "batch_id", paths["timeline"])
        if batch_id < 0:
            raise RuntimeError(f"continuous decode lacks batch_id in {paths['timeline']}")
        timeline_by_batch[batch_id].append(event)
    if set(timeline_by_batch) != set(batch_ids):
        raise RuntimeError(f"batch/timeline ID sets differ in {batch_path}")

    batch_sizes: list[int] = []
    paddings: list[int] = []
    padded_capacity_tokens: list[int] = []
    gather_ms_values: list[float] = []
    scatter_ms_values: list[float] = []
    gather_bytes_values: list[int] = []
    scatter_bytes_values: list[int] = []
    scratch_values: list[int] = []
    allocated_values: list[int] = []
    staging_zero_values: list[int] = []
    backend_total_values: list[float] = []
    batch_service_values: list[float] = []
    memberships: list[tuple[str, ...]] = []
    previous_enqueue_after: int | None = None

    def kv_tensor_bytes(sequence: int) -> int:
        if sequence <= 0:
            raise RuntimeError(f"invalid KV sequence length {sequence} in {batch_path}")
        return kv_heads * sequence * head_dim * kv_element

    for row in batches:
        batch_id = int(row["batch_id"])
        batch_size = exact_int(row["batch_size"], "batch_size", batch_path)
        iteration = exact_int(row["iteration"], "iteration", batch_path)
        request_ids = row["request_ids"].split()
        slots = split_ints(row["slot_ids"], "slot_ids", batch_path)
        past_lengths = split_ints(row["past_lengths"], "past_lengths", batch_path)
        if not (1 <= batch_size <= min(item["max_active_requests"], max_batch_capacity)):
            raise RuntimeError(f"batch size outside configured profile in {batch_path}")
        if any(len(values) != batch_size for values in (request_ids, slots, past_lengths)):
            raise RuntimeError(f"batch vector lengths differ from batch_size in {batch_path}")
        if len(request_ids) != len(set(request_ids)) or len(slots) != len(set(slots)):
            raise RuntimeError(f"duplicate request/slot within batch {batch_id} in {batch_path}")
        if any(request_id not in request_metadata for request_id in request_ids):
            raise RuntimeError(f"unknown request membership in batch {batch_id} of {batch_path}")
        events = sorted(timeline_by_batch[batch_id], key=lambda event: int(event["event_index"]))
        if [event["request_id"] for event in events] != request_ids:
            raise RuntimeError(f"batch membership/order differs from timeline for batch {batch_id}")
        if any(int(event["iteration"]) != iteration for event in events):
            raise RuntimeError(f"batch iteration differs from timeline for batch {batch_id}")
        expected_slots = [request_metadata[request_id]["slot"] for request_id in request_ids]
        if slots != expected_slots or [int(event["batch_slot"]) for event in events] != slots:
            raise RuntimeError(f"batch slot mapping differs from timeline for batch {batch_id}")
        expected_past = [
            request_metadata[event["request_id"]]["prompt"] + int(event["token_step"]) - 1
            for event in events
        ]
        if past_lengths != expected_past:
            raise RuntimeError(f"batch past lengths differ from token state for batch {batch_id}")
        max_past = exact_int(row["max_past_length"], "max_past_length", batch_path)
        padding = exact_int(row["padding_tokens"], "padding_tokens", batch_path)
        if max_past != max(past_lengths) or padding != sum(max_past - value for value in past_lengths):
            raise RuntimeError(f"max-past/padding formula failed for batch {batch_id}")
        denominator = batch_size * max_past
        padding_ratio = nonnegative_float(row["padding_ratio"], "padding_ratio", batch_path)
        if not close_enough(padding_ratio, padding / denominator if denominator else 0.0, absolute=1e-5):
            raise RuntimeError(f"padding ratio formula failed for batch {batch_id}")
        for event in events:
            if int(event["batch_size"]) != batch_size or int(event["max_past_length"]) != max_past:
                raise RuntimeError(f"batch shape metadata differs from timeline for batch {batch_id}")
            if int(event["padding_tokens"]) != padding:
                raise RuntimeError(f"batch padding differs from timeline for batch {batch_id}")

        h2d_bytes = exact_int(row["h2d_bytes"], "h2d_bytes", batch_path)
        d2h_bytes = exact_int(row["d2h_bytes"], "d2h_bytes", batch_path)
        gather_bytes = exact_int(row["kv_gather_bytes"], "kv_gather_bytes", batch_path)
        scatter_bytes = exact_int(row["kv_scatter_bytes"], "kv_scatter_bytes", batch_path)
        staging_zero_bytes = exact_int(
            row["kv_staging_zero_bytes"], "kv_staging_zero_bytes", batch_path
        )
        expected_gather = tensor_count * sum(kv_tensor_bytes(value) for value in past_lengths)
        expected_scatter = tensor_count * sum(kv_tensor_bytes(value + 1) for value in past_lengths)
        expected_h2d = (
            batch_size * input_element
            + batch_size * (max_past + 1) * mask_element
            + (batch_size * position_element if requires_positions else 0)
        )
        expected_d2h = batch_size * vocab * logits_element
        expected_staging_zero = tensor_count * batch_size * kv_tensor_bytes(max_past)
        if gather_bytes != expected_gather or scatter_bytes != expected_scatter:
            raise RuntimeError(
                f"KV gather/scatter byte formula failed for batch {batch_id}: "
                f"{gather_bytes}/{scatter_bytes} vs {expected_gather}/{expected_scatter}"
            )
        if h2d_bytes != expected_h2d or d2h_bytes != expected_d2h:
            raise RuntimeError(
                f"batch H2D/D2H byte formula failed for batch {batch_id}: "
                f"{h2d_bytes}/{d2h_bytes} vs {expected_h2d}/{expected_d2h}"
            )
        if staging_zero_bytes != expected_staging_zero:
            raise RuntimeError(
                f"KV staging memset byte formula failed for batch {batch_id}: "
                f"{staging_zero_bytes} vs {expected_staging_zero}"
            )
        scratch = exact_int(row["batch_scratch_bytes"], "batch_scratch_bytes", batch_path)
        allocated = exact_int(
            row["batch_scratch_allocated_bytes"], "batch_scratch_allocated_bytes", batch_path
        )
        expected_scratch = (
            batch_size * input_element
            + batch_size * (max_past + 1) * mask_element
            + (batch_size * position_element if requires_positions else 0)
            + tensor_count * batch_size * (kv_tensor_bytes(max_past) + kv_tensor_bytes(max_past + 1))
            + batch_size * vocab * logits_element
        )
        if scratch != expected_scratch:
            raise RuntimeError(
                f"batch scratch byte formula failed for batch {batch_id}: {scratch} vs {expected_scratch}"
            )
        if allocated != expected_allocated_scratch or allocated < scratch:
            raise RuntimeError(
                f"persistent scratch allocation mismatch for batch {batch_id}: "
                f"{allocated} vs {expected_allocated_scratch}"
            )
        if exact_int(
            row["buffer_allocation_count"], "buffer_allocation_count", batch_path
        ) != 0:
            raise RuntimeError(f"measured decode allocated a staging buffer in batch {batch_id}")
        enqueue_before = exact_int(row["enqueue_before"], "enqueue_before", batch_path)
        enqueue_after = exact_int(row["enqueue_after"], "enqueue_after", batch_path)
        if enqueue_before < 0 or enqueue_after < 0 or enqueue_after - enqueue_before != 1:
            raise RuntimeError(f"batch {batch_id} did not execute exactly one enqueue")
        if previous_enqueue_after is not None and enqueue_before != previous_enqueue_after:
            raise RuntimeError(f"decode enqueue counter is discontinuous before batch {batch_id}")
        previous_enqueue_after = enqueue_after
        for field in (
            "input_prepare_ms", "shape_ms", "buffer_bind_ms", "kv_gather_ms",
            "h2d_ms", "enqueue_ms", "d2h_ms", "kv_scatter_ms", "sync_ms",
            "backend_total_ms", "batch_service_ms",
        ):
            nonnegative_float(row[field], field, batch_path)
        if float(row["batch_service_ms"]) + 0.05 < float(row["backend_total_ms"]):
            raise RuntimeError(f"batch service time is smaller than backend time for batch {batch_id}")

        batch_sizes.append(batch_size)
        paddings.append(padding)
        padded_capacity_tokens.append(denominator)
        gather_ms_values.append(float(row["kv_gather_ms"]))
        scatter_ms_values.append(float(row["kv_scatter_ms"]))
        gather_bytes_values.append(gather_bytes)
        scatter_bytes_values.append(scatter_bytes)
        scratch_values.append(scratch)
        allocated_values.append(allocated)
        staging_zero_values.append(staging_zero_bytes)
        backend_total_values.append(float(row["backend_total_ms"]))
        batch_service_values.append(float(row["batch_service_ms"]))
        memberships.append(tuple(request_ids))

    if sum(batch_sizes) != len(decode_events):
        raise RuntimeError(
            f"sum(batch_size) != decode token events in {batch_path}: "
            f"{sum(batch_sizes)} vs {len(decode_events)}"
        )
    if log["decode enqueues"] != len(batches):
        raise RuntimeError(f"continuous batch count/enqueue count mismatch in {paths['log']}")
    if item["max_active_requests"] > 1 and item["request_count"] > 1 and max(batch_sizes) <= 1:
        raise RuntimeError(f"continuous configuration never formed batch>1: {batch_path}")
    membership_churn = any(left != right for left, right in zip(memberships, memberships[1:]))
    actual_lengths = {metadata["output"] for metadata in request_metadata.values()}
    churn_required = (
        item["request_count"] > item["max_active_requests"]
        or len(actual_lengths) > 1
        or len(set(item["arrival_iterations"])) > 1
    )
    if churn_required and not membership_churn:
        raise RuntimeError(f"dynamic membership churn was required but absent in {batch_path}")
    total_padded_capacity = sum(padded_capacity_tokens)
    weighted_padding_ratio = sum(paddings) / total_padded_capacity if total_padded_capacity else 0.0
    batch_run = {
        "config_id": item["config_id"],
        "category": item["category"],
        "repeat": repeat,
        "batch_count": len(batches),
        "decode_tokens": len(decode_events),
        "mean_batch_size": statistics.mean(batch_sizes),
        "median_batch_size": statistics.median(batch_sizes),
        "p95_batch_size": percentile(batch_sizes, 95),
        "max_batch_size": max(batch_sizes),
        "mean_capacity_utilization": statistics.mean(
            value / max_batch_capacity for value in batch_sizes
        ),
        "multi_request_batch_fraction": sum(value > 1 for value in batch_sizes) / len(batch_sizes),
        "padding_tokens": sum(paddings),
        "padding_ratio": weighted_padding_ratio,
        "kv_gather_ms": sum(gather_ms_values),
        "kv_scatter_ms": sum(scatter_ms_values),
        "kv_gather_bytes": sum(gather_bytes_values),
        "kv_scatter_bytes": sum(scatter_bytes_values),
        "kv_staging_zero_bytes": sum(staging_zero_values),
        "mean_backend_total_ms": statistics.mean(backend_total_values),
        "mean_batch_service_ms": statistics.mean(batch_service_values),
        "buffer_allocation_count": 0,
        "peak_batch_scratch_bytes": max(scratch_values),
        "peak_batch_scratch_allocated_bytes": max(allocated_values),
        "decode_tokens_per_enqueue": len(decode_events) / len(batches),
    }
    return {
        "mean_batch_size": statistics.mean(batch_sizes),
        "p95_batch_size": percentile(batch_sizes, 95),
        "max_batch_size_observed": max(batch_sizes),
        "mean_batch_capacity_utilization": batch_run["mean_capacity_utilization"],
        "multi_request_batch_fraction": batch_run["multi_request_batch_fraction"],
        "padding_ratio": weighted_padding_ratio,
        "total_kv_gather_ms": sum(gather_ms_values),
        "total_kv_scatter_ms": sum(scatter_ms_values),
        "total_kv_gather_bytes": sum(gather_bytes_values),
        "total_kv_scatter_bytes": sum(scatter_bytes_values),
        "total_kv_staging_zero_bytes": sum(staging_zero_values),
        "mean_backend_total_ms": statistics.mean(backend_total_values),
        "mean_batch_service_ms": statistics.mean(batch_service_values),
        "peak_batch_scratch_bytes": max(scratch_values),
        "peak_batch_scratch_allocated_bytes": max(allocated_values),
        "batch_run": batch_run,
        "membership_churn": membership_churn,
    }


def aggregate_stat_rows(
    records: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in key_fields)].append(record)
    result: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            field: value for field, value in zip(key_fields, key)
        }
        out["repeats"] = len(rows)
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            out[f"mean_{metric}"] = f"{statistics.mean(values):.6f}"
            out[f"median_{metric}"] = f"{statistics.median(values):.6f}"
            out[f"stdev_{metric}"] = (
                f"{statistics.stdev(values) if len(values) > 1 else 0.0:.6f}"
            )
            out[f"p95_{metric}"] = f"{percentile(values, 95):.6f}"
        result.append(out)
    return result


def aggregate_performance(
    run_records: Sequence[dict[str, Any]],
    configurations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    performance_ids = {
        item["config_id"] for item in configurations
        if not item["category"].lower().startswith("correctness")
    }
    selected = [row for row in run_records if row["config_id"] in performance_ids]
    expected_repeats = {
        item["config_id"]: int(item["repeats"])
        for item in configurations if item["config_id"] in performance_ids
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row["config_id"]), str(row["variant"]))].append(row)
    required = {(config_id, variant) for config_id in performance_ids for variant in VARIANTS}
    if set(grouped) != required:
        raise RuntimeError(f"performance aggregate matrix mismatch: {sorted(required - set(grouped))}")
    for (config_id, variant), rows in grouped.items():
        repeats = sorted(int(row["repeat"]) for row in rows)
        expected = list(range(1, expected_repeats[config_id] + 1))
        if repeats != expected or len(rows) < MIN_REPEATS:
            raise RuntimeError(f"performance repeats are incomplete for {config_id}/{variant}")

    summaries = aggregate_stat_rows(
        selected,
        RUN_METRICS,
        key_fields=("config_id", "category", "variant", "request_count", "max_active_requests"),
    )
    summary_lookup = {(str(row["config_id"]), str(row["variant"])): row for row in summaries}
    run_lookup = {
        (str(row["config_id"]), str(row["variant"]), int(row["repeat"])): row
        for row in selected
    }
    for config_id in sorted(performance_ids):
        serial_summary = summary_lookup[(config_id, "serial")]
        continuous_summary = summary_lookup[(config_id, "continuous")]
        repeats = expected_repeats[config_id]
        for metric in RUN_METRICS:
            serial_summary[f"aggregate_mean_change_{metric}_pct"] = "0.000000"
            serial_summary[f"paired_mean_change_{metric}_pct"] = "0.000000"
            serial_summary[f"paired_median_change_{metric}_pct"] = "0.000000"
            baseline_mean = float(serial_summary[f"mean_{metric}"])
            candidate_mean = float(continuous_summary[f"mean_{metric}"])
            continuous_summary[f"aggregate_mean_change_{metric}_pct"] = (
                f"{(candidate_mean / baseline_mean - 1.0) * 100.0:.6f}"
                if baseline_mean else ("0.000000" if candidate_mean == 0.0 else "")
            )
            changes: list[float] = []
            for repeat in range(1, repeats + 1):
                baseline = float(run_lookup[(config_id, "serial", repeat)][metric])
                candidate = float(run_lookup[(config_id, "continuous", repeat)][metric])
                if baseline:
                    changes.append((candidate / baseline - 1.0) * 100.0)
                elif candidate == 0.0:
                    changes.append(0.0)
            continuous_summary[f"paired_mean_change_{metric}_pct"] = (
                f"{statistics.mean(changes):.6f}" if changes else ""
            )
            continuous_summary[f"paired_median_change_{metric}_pct"] = (
                f"{statistics.median(changes):.6f}" if changes else ""
            )
    variant_order = {"serial": 0, "continuous": 1}
    summaries.sort(key=lambda row: (str(row["config_id"]), variant_order[str(row["variant"])]))
    return summaries


def aggregate_batches(
    batch_runs: Sequence[dict[str, Any]], configurations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    performance_ids = {
        item["config_id"] for item in configurations
        if not item["category"].lower().startswith("correctness")
    }
    selected = [row for row in batch_runs if row["config_id"] in performance_ids]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["config_id"])].append(row)
    if set(grouped) != performance_ids:
        raise RuntimeError(f"batch aggregate matrix mismatch: {sorted(performance_ids - set(grouped))}")
    for item in configurations:
        if item["config_id"] not in performance_ids:
            continue
        rows = grouped[item["config_id"]]
        repeats = sorted(int(row["repeat"]) for row in rows)
        if repeats != list(range(1, int(item["repeats"]) + 1)) or len(rows) < MIN_REPEATS:
            raise RuntimeError(f"batch repeats are incomplete for {item['config_id']}")
    return aggregate_stat_rows(
        selected,
        BATCH_RUN_METRICS,
        key_fields=("config_id", "category"),
    )


def aggregate_nvml(
    run_dir: Path, expected_keys: Iterable[tuple[str, str, int]]
) -> list[dict[str, Any]]:
    path = run_dir / "raw" / "nvml.csv"
    rows = read_csv(path, NVML_COLUMNS)
    monotonic = [finite_float(row["monotonic_seconds"], "monotonic_seconds", path) for row in rows]
    if monotonic != sorted(monotonic):
        raise RuntimeError(f"NVML monotonic_seconds moved backwards in {path}")
    expected_phases = {f"{variant}:{config_id}:r{repeat}" for variant, config_id, repeat in expected_keys}
    observed_phases = {row["phase"] for row in rows}
    missing_phases = sorted(expected_phases.difference(observed_phases))
    if missing_phases:
        raise RuntimeError(f"NVML trace lacks process phases: {missing_phases[:10]}")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    phase_re = re.compile(r"^(serial|continuous):([A-Za-z0-9][A-Za-z0-9._-]*):r(\d+)$")
    for row in rows:
        phase = row["phase"]
        match = phase_re.fullmatch(phase)
        if match:
            grouped[(match.group(1), match.group(2))].append(row)
        else:
            grouped[(phase, "")].append(row)
    numeric_fields = (
        "temperature.gpu", "power.draw", "power.limit", "clocks.current.sm",
        "clocks.current.memory", "memory.used", "memory.total", "utilization.gpu",
        "utilization.memory",
    )
    result: list[dict[str, Any]] = []
    for (variant, config_id), samples in sorted(grouped.items()):
        out: dict[str, Any] = {
            "variant_or_phase": variant,
            "config_id": config_id,
            "samples": len(samples),
        }
        memory_values: list[float] = []
        for field in numeric_fields:
            values: list[float] = []
            for row in samples:
                raw = row.get(field, "").strip()
                if not raw or "N/A" in raw:
                    continue
                values.append(finite_float(raw, field, path))
            out[f"min_{field}"] = f"{min(values):.6f}" if values else ""
            out[f"mean_{field}"] = f"{statistics.mean(values):.6f}" if values else ""
            out[f"max_{field}"] = f"{max(values):.6f}" if values else ""
            if field == "memory.used":
                memory_values = values
        if not memory_values:
            raise RuntimeError(f"NVML group {variant}/{config_id} has no numeric memory.used")
        out["memory_used_delta_mib"] = f"{max(memory_values) - min(memory_values):.6f}"
        result.append(out)
    return result


def analyze_raw(
    run_dir: Path,
    config: dict[str, Any],
    protocol: dict[str, Any],
    configurations: Sequence[dict[str, Any]],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    matrix = validate_raw_matrix(run_dir, configurations)
    process_rows, process_summary = validate_process_matrix(run_dir, configurations, matrix)
    goldens = load_goldens(config, repo_root)
    used_case_ids = {case_id for item in configurations for case_id in item["request_case_ids"]}
    if not used_case_ids.issubset(goldens):
        raise RuntimeError(f"configuration cases lack goldens: {sorted(used_case_ids - set(goldens))}")

    run_records: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    batch_runs: list[dict[str, Any]] = []
    observed_multi_batch = False
    observed_slot_reuse = False
    observed_membership_churn = False
    round_membership_pairs: dict[
        tuple[str, int], dict[str, tuple[tuple[int, tuple[str, ...]], ...]]
    ] = defaultdict(dict)
    config_by_id = {item["config_id"]: item for item in configurations}
    for key in sorted(matrix, key=lambda value: (value[1], value[2], VARIANTS.index(value[0]))):
        variant, config_id, repeat = key
        item = config_by_id[config_id]
        run_record, correctness_rows, auxiliary = validate_request_and_timeline(
            variant=variant,
            item=item,
            repeat=repeat,
            paths=matrix[key],
            protocol=protocol,
            goldens=goldens,
            process_row=process_rows[key],
            repo_root=repo_root,
        )
        run_records.append(run_record)
        correctness.extend(correctness_rows)
        if auxiliary["batch_run"] is not None:
            batch_runs.append(auxiliary["batch_run"])
        if variant == "continuous":
            observed_multi_batch |= auxiliary["observed_multi_batch"]
            observed_slot_reuse |= auxiliary["observed_slot_reuse"]
            observed_membership_churn |= auxiliary["observed_membership_churn"]
        round_membership_pairs[(config_id, repeat)][variant] = auxiliary[
            "decode_round_membership"
        ]
    if not observed_multi_batch:
        raise RuntimeError("v1.0 evidence never formed a multi-request GPU batch")
    if not observed_slot_reuse:
        raise RuntimeError("v1.0 evidence never reused a completed request slot")
    if not observed_membership_churn:
        raise RuntimeError("v1.0 evidence never changed dynamic batch membership")
    for key, pair in round_membership_pairs.items():
        if set(pair) != set(VARIANTS) or pair["serial"] != pair["continuous"]:
            raise RuntimeError(
                f"serial/continuous scheduler-round membership differs for {key}"
            )

    expected_correctness = sum(
        int(item["request_count"]) * int(item["repeats"]) * len(VARIANTS)
        for item in configurations
    )
    if len(correctness) != expected_correctness or any(int(row["exact_match"]) != 1 for row in correctness):
        raise RuntimeError(
            f"correctness matrix mismatch: {len(correctness)} rows vs {expected_correctness} expected"
        )
    paired_candidates: dict[tuple[str, int, str], dict[str, str]] = defaultdict(dict)
    for row in correctness:
        key = (str(row["config_id"]), int(row["repeat"]), str(row["request_id"]))
        paired_candidates[key][str(row["variant"])] = str(row["candidate_sequence"])
    for key, pair in paired_candidates.items():
        if set(pair) != set(VARIANTS) or pair["serial"] != pair["continuous"]:
            raise RuntimeError(f"serial/continuous token pair differs for {key}")

    performance = aggregate_performance(run_records, configurations)
    batch_summary = aggregate_batches(batch_runs, configurations)
    memory = aggregate_nvml(run_dir, matrix.keys())
    return {
        "run_records": run_records,
        "correctness": correctness,
        "performance": performance,
        "batch": batch_summary,
        "memory": memory,
        "process": process_summary,
        "matrix_runs": len(matrix),
    }


SUMMARY_NAMES = (
    "correctness_summary.csv",
    "performance_summary.csv",
    "batch_summary.csv",
    "memory_summary.csv",
    "process_summary.csv",
)


def render_readme(
    config: dict[str, Any], configurations: Sequence[dict[str, Any]], analysis: dict[str, Any]
) -> str:
    performance_configurations = sum(
        not item["category"].lower().startswith("correctness") for item in configurations
    )
    correctness_configurations = len(configurations) - performance_configurations
    lines = [
        f"# MiniLLM-RT v1.0 benchmark run `{config['run_id']}`",
        "",
        "> Immutable continuous-batching evidence. Never regenerate this directory in place; use a new run ID.",
        "",
        f"- Protocol: `{config['protocol']}`",
        f"- Performance configurations: {performance_configurations}",
        f"- Long-correctness configurations: {correctness_configurations}",
        f"- Raw serial/continuous processes: {analysis['matrix_runs']}",
        f"- Correctness comparisons against HF long golden/prefix: {len(analysis['correctness'])} (0 mismatch)",
        f"- Measured repeats per performance variant/configuration: {config['repeats']}",
        "- Warmup: one representative full generation per process (the primary max-active batch plus uncovered prompt/output representatives), not one enqueue and not a duplicate of every request.",
        "- Primary scheduler-visible TTFT/TPOT/latency: derived from `timeline.end_ms` service-completion timestamps. Batched rows intentionally share the DecodeBatch return time; this is not a per-row sampler timestamp.",
        "- Continuous policy: every active request is padded to the iteration maximum past length; this run does not claim a bucket policy.",
        "- Enqueue gate: every continuous batch row advances the TensorRT enqueue counter by exactly one; serial decode tokens equal serial decode enqueues.",
        "- Byte gates: KV gather/scatter, staging memset, H2D/D2H and working scratch are recomputed from ModelSpec and per-batch shapes.",
        "- Scheduler gate: request slots can join, finish, leave and be reused; the final timeline row owns zero live resources.",
        "",
        "Evidence files:",
        "",
        "- `environment.json`: hardware/software state and operator notes",
        "- `model_manifest.json`: model, ModelSpec and Engine hashes",
        "- `benchmark_config.json`: exact A/B matrix, inputs, arrivals and limits",
        "- `correctness_summary.csv`: every request/variant/repeat versus HF",
        "- `performance_summary.csv`: 10-repeat statistics and paired continuous-vs-serial changes",
        "- `batch_summary.csv`: actual batch size, utilization, padding, copy and scratch evidence",
        "- `memory_summary.csv`: global NVML samples (not per-process attribution)",
        "- `process_summary.csv`: process wall-time statistics",
        "- `artifact_manifest.json`: SHA-256 binding for every raw and top-level evidence file",
        "",
        "Large request/timeline/batch CSVs and logs remain under `raw/`.",
        "",
    ]
    return "\n".join(lines)


def build_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    raw = [
        artifact_row(path, run_dir)
        for path in sorted((run_dir / "raw").rglob("*")) if path.is_file()
    ]
    evidence_names = (
        "environment.json", "model_manifest.json", "benchmark_config.json",
        *SUMMARY_NAMES, "README.md",
    )
    evidence = [artifact_row(run_dir / name, run_dir) for name in evidence_names]
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_artifacts": raw,
        "evidence": evidence,
    }


def verify_artifact_manifest(run_dir: Path) -> None:
    path = run_dir / "artifact_manifest.json"
    payload = load_json(path)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported artifact manifest: {path}")
    raw = payload.get("raw_artifacts")
    evidence = payload.get("evidence")
    if not isinstance(raw, list) or not raw or not isinstance(evidence, list) or not evidence:
        raise RuntimeError(f"artifact manifest is empty: {path}")
    for family, records in (("raw", raw), ("evidence", evidence)):
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError(f"invalid {family} artifact record in {path}")
            relative = str(record.get("path", ""))
            if not relative or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError(f"invalid/duplicate {family} artifact path {relative!r}")
            seen.add(relative)
            artifact = run_dir / relative
            if not artifact.is_file() or artifact.stat().st_size != int(record.get("size_bytes", -1)):
                raise RuntimeError(f"{family} artifact disappeared or changed size: {artifact}")
            if sha256_file(artifact) != record.get("sha256"):
                raise RuntimeError(f"{family} artifact hash changed: {artifact}")
    regenerated = build_artifact_manifest(run_dir)
    if payload != regenerated:
        raise RuntimeError("artifact manifest file set/order differs from current run evidence")


def command_aggregate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    generated = [run_dir / name for name in (*SUMMARY_NAMES, "README.md", "artifact_manifest.json")]
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite immutable v1.0 evidence: {existing}")
    config, _model, _environment, protocol, configurations = validate_manifests(run_dir)
    analysis = analyze_raw(run_dir, config, protocol, configurations)
    # No file is written before every raw family and every correctness gate has passed.
    write_csv_immutable(run_dir / "correctness_summary.csv", analysis["correctness"])
    write_csv_immutable(run_dir / "performance_summary.csv", analysis["performance"])
    write_csv_immutable(run_dir / "batch_summary.csv", analysis["batch"])
    write_csv_immutable(run_dir / "memory_summary.csv", analysis["memory"])
    write_csv_immutable(run_dir / "process_summary.csv", analysis["process"])
    write_text_immutable(run_dir / "README.md", render_readme(config, configurations, analysis))
    write_json_immutable(run_dir / "artifact_manifest.json", build_artifact_manifest(run_dir))
    print(
        f"validated {analysis['matrix_runs']} processes, "
        f"{len(analysis['correctness'])} HF comparisons and "
        f"{len(analysis['batch'])} continuous-batch configurations"
    )
    print(f"immutable v1.0 evidence: {run_dir}")


def command_verify(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    if args.offline_evidence and not args.final:
        raise RuntimeError("--offline-evidence requires --final")
    offline_repo_root = args.repo.expanduser().resolve() if args.offline_evidence else None
    config, _model, _environment, protocol, configurations = validate_manifests(
        run_dir, offline_repo_root
    )
    if not args.final:
        print(f"prepared v1.0 manifests verified: {run_dir}")
        return
    required = [*SUMMARY_NAMES, "README.md", "artifact_manifest.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"final v1.0 bundle is incomplete: {missing}")
    verify_artifact_manifest(run_dir)
    analysis = analyze_raw(
        run_dir, config, protocol, configurations, offline_repo_root
    )
    expected_summaries = {
        "correctness_summary.csv": analysis["correctness"],
        "performance_summary.csv": analysis["performance"],
        "batch_summary.csv": analysis["batch"],
        "memory_summary.csv": analysis["memory"],
        "process_summary.csv": analysis["process"],
    }
    for name, rows in expected_summaries.items():
        assert_csv_matches(run_dir / name, rows, name)
    expected_readme = render_readme(config, configurations, analysis)
    if (run_dir / "README.md").read_text(encoding="utf-8") != expected_readme:
        raise RuntimeError("frozen README does not match regenerated evidence")
    # Recheck after regeneration to make it explicit that verification performs no writes.
    verify_artifact_manifest(run_dir)
    if args.offline_evidence:
        print(
            "final immutable v1.0 evidence verified offline: "
            f"{run_dir} (external runtime/Engine/ONNX files not rehashed)"
        )
    else:
        print(f"final immutable v1.0 run bundle verified: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate immutable v1.0 continuous-batching evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate", help="validate raw evidence and freeze summaries")
    aggregate.add_argument("--run-dir", type=Path, required=True)
    aggregate.set_defaults(func=command_aggregate)
    verify = subparsers.add_parser("verify", help="verify prepared or final immutable evidence")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--final", action="store_true")
    verify.add_argument(
        "--offline-evidence",
        action="store_true",
        help=(
            "replay a final published bundle without requiring the original external "
            "runtime/Engine/ONNX files; their frozen identity metadata is still checked"
        ),
    )
    verify.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing tracked configs/inputs for offline evidence replay",
    )
    verify.set_defaults(func=command_verify)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileExistsError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
