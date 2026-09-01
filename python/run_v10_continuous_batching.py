#!/usr/bin/env python3
"""Run the immutable v1.0 serial-vs-continuous-batching benchmark matrix.

The measured A/B comparison intentionally uses the same prefill Engine, the
same batch-capable decode Engine, the same request order, and the same logical
scheduler capacity for both variants.  Only the decode submission policy
changes: ``scheduler`` issues B=1 enqueues while ``continuous_batch`` issues
one dynamic batched enqueue per scheduler iteration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from qwen_common import DEFAULT_ARTIFACT_ROOT, DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION


DEFAULT_SPEC = Path("configs/qwen2_5_0_5b_fp16.spec")
DEFAULT_WORKLOADS = Path("inputs/qwen_v10/workloads.json")
DEFAULT_ANALYZER = Path("python/analyze_v10_continuous_batching.py")
PROTOCOL = "minillm-rt-v10-continuous-batch-ab-v2"
VARIANTS = ("serial", "continuous")
PROCESS_FIELDS = (
    "sequence_index",
    "kind",
    "config_id",
    "variant",
    "repeat",
    "ab_order",
    "request_count",
    "max_active_requests",
    "total_requested_tokens",
    "process_wall_ms",
    "return_code",
    "started_at_utc",
    "completed_at_utc",
    "requests_csv",
    "timeline_csv",
    "batches_csv",
    "log",
    "command",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_from(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def parse_int_file(path: Path) -> list[int]:
    values = [int(item) for item in path.read_text(encoding="utf-8").split()]
    if not values:
        raise ValueError(f"input ID file is empty: {path}")
    return values


class ArtifactHasher:
    """Hash large Engine/model artifacts once, even if referenced repeatedly."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, Any]] = {}

    def describe(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        cached = self._cache.get(path)
        if cached is not None:
            return dict(cached)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
        self._cache[path] = result
        return dict(result)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_checked(command: Sequence[str], *, cwd: Path, log: Path | None = None) -> float:
    started = time.perf_counter()
    if log is None:
        completed = subprocess.run(list(command), cwd=cwd, check=False)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            stream.write("command: " + shlex.join(command) + "\n")
            stream.flush()
            completed = subprocess.run(
                list(command), cwd=cwd, check=False, stdout=stream, stderr=subprocess.STDOUT
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        suffix = f"; see {log}" if log is not None else ""
        raise RuntimeError(f"command failed ({completed.returncode}){suffix}: {shlex.join(command)}")
    return elapsed_ms


def run_logged(command: Sequence[str], *, cwd: Path, log: Path) -> tuple[float, int, str, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("command: " + shlex.join(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            list(command), cwd=cwd, check=False, stdout=stream, stderr=subprocess.STDOUT
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, completed.returncode, started_at, utc_now()


class GpuTrace:
    """Best-effort global NVML telemetry; it is not process-exclusive memory."""

    FIELDS = (
        "timestamp,name,pstate,temperature.gpu,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory,memory.used,memory.total,"
        "utilization.gpu,utilization.memory"
    )

    def __init__(self, output: Path, interval_seconds: float):
        self.output = output
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.phase = "setup"
        self.lock = threading.Lock()
        self.sampled = threading.Condition()
        self.sampled_phases: set[str] = set()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase
        self.wake_event.set()

    def wait_for_sample(self, phase: str, timeout_seconds: float) -> None:
        with self.sampled:
            observed = self.sampled.wait_for(
                lambda: phase in self.sampled_phases, timeout=timeout_seconds
            )
        if not observed:
            raise RuntimeError(f"NVML sampler did not acknowledge phase {phase!r}")

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="v10-gpu-trace", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval * 4.0))

    def _run(self) -> None:
        fields = self.FIELDS.split(",")
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["monotonic_seconds", "phase", *fields])
            while not self.stop_event.is_set():
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={self.FIELDS}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                with self.lock:
                    phase = self.phase
                if completed.returncode == 0:
                    for line in completed.stdout.strip().splitlines():
                        writer.writerow(
                            [f"{time.monotonic():.6f}", phase]
                            + [item.strip() for item in line.split(",")]
                        )
                    stream.flush()
                    with self.sampled:
                        self.sampled_phases.add(phase)
                        self.sampled.notify_all()
                self.wake_event.wait(self.interval)
                self.wake_event.clear()


@dataclass(frozen=True)
class Workload:
    case_id: str
    golden_case_id: str
    input_path: Path
    golden_path: Path
    prompt_length: int
    golden_tokens: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    stopped_on_eos: bool
    input_artifact: dict[str, Any]
    golden_artifact: dict[str, Any]


def base_case_id(case_id: str) -> str:
    match = re.fullmatch(r"(perf_p(?:8|32|128|256))_o128", case_id)
    if match:
        return match.group(1)
    if case_id == "eos_instruction_o16":
        return "eos_instruction"
    raise ValueError(f"unsupported v1.0 workload case: {case_id}")


def load_workloads(
    repo: Path, manifest_path: Path, hasher: ArtifactHasher
) -> tuple[dict[str, Workload], dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("v1.0 workload manifest must use schema_version=1")
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("workload model identity/revision does not match the pinned Qwen model")

    workloads: dict[str, Workload] = {}
    golden_artifacts: list[dict[str, Any]] = []
    for entry in manifest.get("cases", []):
        golden_case_id = str(entry["case_id"])
        case_id = base_case_id(golden_case_id)
        input_path = resolve_from(repo, Path(str(entry["input_ids_path"])))
        golden_path = (manifest_path.parent / "goldens" / f"{golden_case_id}.json").resolve()
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        if golden.get("case_id") != golden_case_id:
            raise ValueError(f"golden case identity mismatch: {golden_path}")
        if golden.get("model_id") != MODEL_ID or golden.get("revision") != MODEL_REVISION:
            raise ValueError(f"golden model identity/revision mismatch: {golden_path}")
        input_ids = parse_int_file(input_path)
        if [int(value) for value in golden.get("input_ids", [])] != input_ids:
            raise ValueError(f"golden input IDs do not match {input_path}")
        prompt_length = int(golden["prompt_length"])
        if prompt_length != len(input_ids):
            raise ValueError(f"prompt length mismatch for {case_id}")
        requested = int(entry["requested_new_tokens"])
        if requested != int(golden["requested_new_tokens"]):
            raise ValueError(f"golden output capacity mismatch for {case_id}")
        generated = tuple(int(value) for value in golden["generated_ids"])
        if len(generated) != int(golden["actual_new_tokens"]):
            raise ValueError(f"golden generated length mismatch for {case_id}")
        input_artifact = hasher.describe(input_path)
        golden_artifact = hasher.describe(golden_path)
        golden_artifacts.append(golden_artifact)
        workloads[case_id] = Workload(
            case_id=case_id,
            golden_case_id=golden_case_id,
            input_path=input_path,
            golden_path=golden_path,
            prompt_length=prompt_length,
            golden_tokens=generated,
            eos_token_ids=tuple(int(value) for value in golden["eos_token_ids"]),
            stopped_on_eos=bool(golden["stopped_on_eos"]),
            input_artifact=input_artifact,
            golden_artifact=golden_artifact,
        )

    required = {"perf_p8", "perf_p32", "perf_p128", "perf_p256", "eos_instruction"}
    if set(workloads) != required:
        raise ValueError(f"v1.0 workloads must be exactly {sorted(required)}")
    return workloads, hasher.describe(manifest_path), golden_artifacts


def parse_model_spec(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"malformed ModelSpec line: {raw}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def expected_tokens(workload: Workload, output_limit: int) -> list[int]:
    if output_limit <= 0:
        raise ValueError("request output limits must be positive")
    if output_limit > 128 and workload.case_id != "eos_instruction":
        raise ValueError(f"output limit exceeds the long golden for {workload.case_id}")
    return list(workload.golden_tokens[:output_limit])


def make_configuration(
    *,
    config_id: str,
    category: str,
    description: str,
    case_ids: list[str],
    output_limits: list[int],
    arrivals: list[int],
    max_active_requests: int,
    repeats: int,
    workloads: dict[str, Workload],
) -> dict[str, Any]:
    if not (len(case_ids) == len(output_limits) == len(arrivals)):
        raise ValueError(f"configuration vector length mismatch: {config_id}")
    if not case_ids or arrivals != sorted(arrivals):
        raise ValueError(f"invalid request order/arrival trace: {config_id}")
    bindings: list[dict[str, Any]] = []
    for index, (case_id, output_limit, arrival) in enumerate(
        zip(case_ids, output_limits, arrivals, strict=True)
    ):
        workload = workloads[case_id]
        expected = expected_tokens(workload, output_limit)
        finish = "eos" if expected and expected[-1] in workload.eos_token_ids else "length"
        bindings.append(
            {
                "request_id": f"request_{index}",
                "case_id": case_id,
                "golden_case_id": workload.golden_case_id,
                "input_ids_path": str(workload.input_path),
                "input_ids_sha256": workload.input_artifact["sha256"],
                "golden_path": str(workload.golden_path),
                "golden_sha256": workload.golden_artifact["sha256"],
                "prompt_length": workload.prompt_length,
                "output_limit": output_limit,
                "expected_output_length": len(expected),
                "expected_finish_reason": finish,
                "arrival_iteration": arrival,
            }
        )
    return {
        "config_id": config_id,
        "category": category,
        "description": description,
        "request_count": len(case_ids),
        "max_active_requests": max_active_requests,
        # request_case_ids are the immutable HF-golden identities consumed by
        # the analyzer.  workload_case_ids retain the shorter logical aliases
        # used to assemble repeated request mixes in this runner.
        "request_case_ids": [workloads[case_id].golden_case_id for case_id in case_ids],
        "workload_case_ids": case_ids,
        "input_paths": [str(workloads[case_id].input_path) for case_id in case_ids],
        "prompt_lengths": [workloads[case_id].prompt_length for case_id in case_ids],
        "output_limits": output_limits,
        "arrival_iterations": arrivals,
        "repeats": repeats,
        "request_bindings": bindings,
    }


def build_configurations(
    workloads: dict[str, Workload], repeats: int, max_batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    performance: list[dict[str, Any]] = []
    for request_count in (1, 2, 4, 8, 16):
        active = min(request_count, max_batch_size)
        performance.append(
            make_configuration(
                config_id=f"equal_n{request_count}",
                category="performance",
                description=(
                    f"{request_count} equal-length requests; prompt=32, output=32, active={active}"
                ),
                case_ids=["perf_p32"] * request_count,
                output_limits=[32] * request_count,
                arrivals=[0] * request_count,
                max_active_requests=active,
                repeats=repeats,
                workloads=workloads,
            )
        )

    mixed_cases = ["perf_p8", "perf_p32", "perf_p128", "perf_p256"] * 4
    mixed_limits = [16, 32] * 8
    for active in (1, 4, 8):
        performance.append(
            make_configuration(
                config_id=f"mixed_a{active}",
                category="performance",
                description=f"16 mixed-length/mixed-output requests; all-at-once, active={active}",
                case_ids=list(mixed_cases),
                output_limits=list(mixed_limits),
                arrivals=[0] * 16,
                max_active_requests=active,
                repeats=repeats,
                workloads=workloads,
            )
        )

    staggered_arrivals = [0, 0, 0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]
    performance.append(
        make_configuration(
            config_id="staggered_a8",
            category="performance",
            description="16 mixed requests; active=8 with a fixed staggered arrival trace",
            case_ids=list(mixed_cases),
            output_limits=list(mixed_limits),
            arrivals=staggered_arrivals,
            max_active_requests=8,
            repeats=repeats,
            workloads=workloads,
        )
    )

    correctness = make_configuration(
        config_id="correctness_long",
        category="correctness",
        description="heterogeneous long-output, independent completion, and EOS correctness gate",
        case_ids=[
            "perf_p8",
            "perf_p32",
            "perf_p128",
            "perf_p256",
            "perf_p8",
            "perf_p32",
            "perf_p128",
            "perf_p256",
            "eos_instruction",
        ],
        output_limits=[16, 32, 128, 64, 32, 16, 64, 128, 16],
        arrivals=[0] * 9,
        max_active_requests=8,
        repeats=1,
        workloads=workloads,
    )
    return performance, correctness


def read_request_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "request_id",
            "prompt_len",
            "output_len",
            "finish_reason",
            "token_sequence",
            "arrival_iteration",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"request CSV has an invalid schema: {path}")
        return list(reader)


def validate_request_csv(
    path: Path, configuration: dict[str, Any], workloads: dict[str, Workload]
) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"runtime did not create request evidence: {path}")
    rows = read_request_rows(path)
    bindings = configuration["request_bindings"]
    if len(rows) != len(bindings):
        raise RuntimeError(f"request row count mismatch in {path}")
    for row, binding in zip(rows, bindings, strict=True):
        workload = workloads[str(binding["case_id"])]
        expected = expected_tokens(workload, int(binding["output_limit"]))
        actual = [int(value) for value in row["token_sequence"].split() if value]
        checks = {
            "request_id": str(binding["request_id"]),
            "prompt_len": str(binding["prompt_length"]),
            "output_len": str(len(expected)),
            "finish_reason": str(binding["expected_finish_reason"]),
            "arrival_iteration": str(binding["arrival_iteration"]),
        }
        for field, expected_value in checks.items():
            if row[field] != expected_value:
                raise RuntimeError(
                    f"{path}: {row['request_id']} {field}={row[field]!r}, "
                    f"expected {expected_value!r}"
                )
        if actual != expected:
            mismatch = next(
                (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                min(len(actual), len(expected)),
            )
            raise RuntimeError(
                f"{path}: {row['request_id']} token mismatch at generated step {mismatch}"
            )
    return rows


def validate_pair(serial: list[dict[str, str]], continuous: list[dict[str, str]]) -> None:
    fields = ("request_id", "prompt_len", "output_len", "finish_reason", "token_sequence")
    if len(serial) != len(continuous):
        raise RuntimeError("serial/continuous request counts differ")
    for serial_row, continuous_row in zip(serial, continuous, strict=True):
        for field in fields:
            if serial_row[field] != continuous_row[field]:
                raise RuntimeError(
                    f"serial/continuous mismatch for {serial_row['request_id']} field {field}"
                )


def write_process_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROCESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def runtime_command(
    *,
    binary: Path,
    prefill_engine: Path,
    decode_batch_engine: Path,
    model_spec: Path,
    configuration: dict[str, Any],
    variant: str,
    warmup: int,
    max_batch_size: int,
    kv_block_size: int,
    kv_num_blocks: int,
    request_csv: Path,
    timeline_csv: Path,
    batch_csv: Path | None,
) -> list[str]:
    active = int(configuration["max_active_requests"])
    command = [
        str(binary),
        "--model",
        str(prefill_engine),
        "--decode-model",
        str(decode_batch_engine),
        "--model-spec",
        str(model_spec),
        "--request-inputs",
        ",".join(configuration["input_paths"]),
        "--request-max-new-tokens",
        ",".join(str(value) for value in configuration["output_limits"]),
        "--request-arrival-iterations",
        ",".join(str(value) for value in configuration["arrival_iterations"]),
        "--mode",
        "scheduler" if variant == "serial" else "continuous_batch",
        "--provider",
        "tensorrt",
        "--sampling",
        "host",
        "--max-new-tokens",
        str(max(configuration["output_limits"])),
        "--warmup-runs",
        str(warmup),
        "--max-active-requests",
        str(active),
        "--max-tokens-per-step",
        str(active if variant == "serial" else 1),
        "--max-batch-size",
        str(max_batch_size),
        "--kv-block-size",
        str(kv_block_size),
        "--kv-num-blocks",
        str(kv_num_blocks),
        "--timeline-csv",
        str(timeline_csv),
        "--request-summary-csv",
        str(request_csv),
    ]
    if variant == "continuous":
        if batch_csv is None:
            raise ValueError("continuous variant requires batch evidence output")
        command.extend(["--batch-csv", str(batch_csv)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("results/runs/v10"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=Path("build/minillm_rt"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    # Runtime warmup units are representative full generations, not individual
    # enqueues: the main max-active representative batch plus any uncovered
    # prompt/output representatives.  One pass exercises the relevant shapes
    # and materializes the size-specific staging buffers before measurement.
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-pool-tokens", type=int, default=4096)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}", args.run_id) is None:
        raise ValueError(
            "run ID must be 3-80 characters using letters, digits, dot, underscore, or hyphen"
        )
    if args.warmup != 1 or args.repeats < 10:
        raise ValueError(
            "official v1.0 runs require exactly one representative-generation warmup "
            "and repeats>=10"
        )
    if args.max_batch_size != 8:
        raise ValueError("the official v1.0 protocol fixes --max-batch-size=8")
    if args.kv_block_size <= 0 or args.kv_pool_tokens <= 0:
        raise ValueError("KV pool and block sizes must be positive")
    if args.kv_pool_tokens % args.kv_block_size != 0:
        raise ValueError("KV pool tokens must be divisible by KV block size")
    if args.gpu_sample_interval <= 0:
        raise ValueError("GPU sample interval must be positive")

    repo = args.repo.resolve()
    run_root = resolve_from(repo, args.run_root)
    run_dir = run_root / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_dir}")

    binary = resolve_from(repo, args.binary)
    model_dir = resolve_from(repo, args.model_dir)
    artifact_root = resolve_from(repo, args.artifact_root)
    model_spec = resolve_from(repo, args.model_spec)
    workload_manifest = resolve_from(repo, args.workloads)
    analyzer = resolve_from(repo, args.analyzer)
    engines = artifact_root / "engines"
    prefill_engine = engines / "qwen2_5_0_5b_prefill.plan"
    decode_batch_engine = engines / "qwen2_5_0_5b_decode_batch.plan"
    decode_batch_inspection = engines / "qwen2_5_0_5b_decode_batch.engine.json"
    required = (
        binary,
        model_spec,
        workload_manifest,
        prefill_engine,
        decode_batch_engine,
        decode_batch_inspection,
        model_dir / "config.json",
        model_dir / "generation_config.json",
        model_dir / "tokenizer.json",
        model_dir / "model.safetensors",
    )
    if not args.dry_run:
        required += (analyzer, repo / "python/capture_environment.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v1.0 benchmark artifacts: " + ", ".join(missing))

    hasher = ArtifactHasher()
    inspection = json.loads(decode_batch_inspection.read_text(encoding="utf-8"))
    if (
        inspection.get("mode") != "decode_batch"
        or int(inspection.get("max_batch_size", 0)) != args.max_batch_size
        or int(inspection.get("max_sequence_length", 0)) != 512
        or inspection.get("precision_contract") != "strongly_typed_fp16_onnx"
    ):
        raise ValueError("decode_batch Engine inspection has the wrong semantic/profile contract")
    inspected_engine = inspection.get("engine")
    inspected_onnx = inspection.get("onnx")
    if not isinstance(inspected_engine, dict) or not isinstance(inspected_onnx, dict):
        raise ValueError("decode_batch Engine inspection lacks Engine/ONNX records")
    engine_artifact = hasher.describe(decode_batch_engine)
    if (
        inspected_engine.get("sha256") != engine_artifact["sha256"]
        or int(inspected_engine.get("size_bytes", -1)) != engine_artifact["size_bytes"]
    ):
        raise ValueError("decode_batch Engine does not match its semantic inspection")
    onnx_artifact = hasher.describe(Path(str(inspected_onnx.get("path", ""))))
    if (
        inspected_onnx.get("sha256") != onnx_artifact["sha256"]
        or int(inspected_onnx.get("size_bytes", -1)) != onnx_artifact["size_bytes"]
    ):
        raise ValueError("decode_batch ONNX does not match its semantic inspection")
    workloads, workload_artifact, golden_artifacts = load_workloads(
        repo, workload_manifest, hasher
    )
    performance_configs, correctness_config = build_configurations(
        workloads, args.repeats, args.max_batch_size
    )
    configurations = performance_configs + [correctness_config]
    model_spec_values = parse_model_spec(model_spec)
    if int(model_spec_values.get("engine_max_sequence_length", "0")) < 384:
        raise ValueError("ModelSpec Engine capacity is too small for p256 + o128 correctness")

    runtime_artifact = hasher.describe(binary)
    model_spec_artifact = hasher.describe(model_spec)
    benchmark_config = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at_utc": utc_now(),
        "run_id": args.run_id,
        "immutable_output": True,
        "operator_notes": args.notes,
        "runner_command": shlex.join(sys.argv),
        "warmup_runs": args.warmup,
        "warmup_generations": args.warmup,
        "warmup_unit": "representative_full_generation",
        "repeats": args.repeats,
        "correctness_repeats": 1,
        "variants": list(VARIANTS),
        "alternating_ab_order": True,
        "configuration_order": "correctness_gate_then_repeat-major_rotated; paired A/B alternates by repeat",
        "raw_variant_directories": {variant: f"raw/{variant}" for variant in VARIANTS},
        "raw_logs_directory": "raw/logs",
        "raw_process_runs": "raw/process_runs.csv",
        "raw_nvml": "raw/nvml.csv",
        "runtime_binary": runtime_artifact,
        "model_spec_artifact": model_spec_artifact,
        "workloads_manifest": workload_artifact,
        "golden_artifacts": golden_artifacts,
        "workload_registry": [
            {
                "case_id": workload.case_id,
                "golden_case_id": workload.golden_case_id,
                "prompt_length": workload.prompt_length,
                "max_golden_tokens": len(workload.golden_tokens),
                "stopped_on_eos": workload.stopped_on_eos,
                "eos_token_ids": list(workload.eos_token_ids),
                "input_ids": workload.input_artifact,
                "golden": workload.golden_artifact,
            }
            for workload in workloads.values()
        ],
        "execution": {
            "provider": "tensorrt",
            "prefill": "serial",
            "serial_decode": "one_request_per_enqueue",
            "continuous_decode": "one_dynamic_batch_per_iteration",
            "serial_max_tokens_per_step_policy": (
                "upper_bound_equals_max_active_requests;"
                "runtime_caps_each_round_to_round_start_active_requests"
            ),
            "round_semantics": "each_round_start_active_request_advances_at_most_once",
            "continuous_max_tokens_per_step": 1,
            "max_batch_size": args.max_batch_size,
            "block_size": args.kv_block_size,
            "pool_tokens": args.kv_pool_tokens,
            "kv_block_size": args.kv_block_size,
            "kv_pool_tokens": args.kv_pool_tokens,
            "kv_num_blocks": args.kv_pool_tokens // args.kv_block_size,
            "comparison_engine_policy": "identical_prefill_and_decode_batch_engines",
            "nvml_scope": "global_device_not_process_exclusive",
        },
        "expected_process_runs": len(performance_configs) * args.repeats * 2 + 2,
        "performance_config_ids": [config["config_id"] for config in performance_configs],
        "correctness_config_ids": [correctness_config["config_id"]],
        "configurations": configurations,
    }

    selected_metadata = []
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        path = model_dir / name
        if path.is_file():
            selected_metadata.append(hasher.describe(path))
    engine_manifests = []
    for path in (
        engines / "qwen2_5_0_5b_prefill.engine.json",
        engines / "qwen2_5_0_5b_decode_batch.engine.json",
        engines / "qwen2_5_0_5b_decode_batch.build.json",
    ):
        if path.is_file():
            engine_manifests.append(hasher.describe(path))
    model_manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir),
        "precision": "fp16-strongly-typed",
        "model_spec": model_spec_values,
        "model_spec_artifact": model_spec_artifact,
        "engines": {
            "prefill": hasher.describe(prefill_engine),
            "decode_batch": hasher.describe(decode_batch_engine),
        },
        "engine_manifests": engine_manifests,
        "selected_model_metadata": selected_metadata,
    }

    if args.dry_run:
        example = runtime_command(
            binary=binary,
            prefill_engine=prefill_engine,
            decode_batch_engine=decode_batch_engine,
            model_spec=model_spec,
            configuration=correctness_config,
            variant="continuous",
            warmup=args.warmup,
            max_batch_size=args.max_batch_size,
            kv_block_size=args.kv_block_size,
            kv_num_blocks=args.kv_pool_tokens // args.kv_block_size,
            request_csv=run_dir / "raw/continuous/correctness_long_r1_requests.csv",
            timeline_csv=run_dir / "raw/continuous/correctness_long_r1_timeline.csv",
            batch_csv=run_dir / "raw/continuous/correctness_long_r1_batches.csv",
        )
        print(
            f"validated v1.0 protocol: {len(performance_configs)} performance configs, "
            f"{benchmark_config['expected_process_runs']} total processes"
        )
        print(f"immutable run directory: {run_dir}")
        print("example command: " + shlex.join(example))
        return

    for directory in (
        run_dir / "raw/serial",
        run_dir / "raw/continuous",
        run_dir / "raw/logs",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "benchmark_config.json", benchmark_config)
    write_json(run_dir / "model_manifest.json", model_manifest)
    run_checked(
        [
            sys.executable,
            str(repo / "python/capture_environment.py"),
            "--output",
            str(run_dir / "environment.json"),
            "--repo",
            str(repo),
            "--notes",
            args.notes,
        ],
        cwd=repo,
    )

    process_path = run_dir / "raw/process_runs.csv"
    process_rows: list[dict[str, Any]] = []
    write_process_rows(process_path, process_rows)
    trace = GpuTrace(run_dir / "raw/nvml.csv", args.gpu_sample_interval)
    trace.start()
    validated_rows: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    sequence_index = 0

    def execute(configuration: dict[str, Any], variant: str, repeat: int, order: str) -> None:
        nonlocal sequence_index
        config_id = str(configuration["config_id"])
        stem = f"{config_id}_r{repeat}"
        variant_dir = run_dir / "raw" / variant
        request_csv = variant_dir / f"{stem}_requests.csv"
        timeline_csv = variant_dir / f"{stem}_timeline.csv"
        batch_csv = variant_dir / f"{stem}_batches.csv" if variant == "continuous" else None
        log = run_dir / "raw/logs" / f"{variant}_{stem}.log"
        command = runtime_command(
            binary=binary,
            prefill_engine=prefill_engine,
            decode_batch_engine=decode_batch_engine,
            model_spec=model_spec,
            configuration=configuration,
            variant=variant,
            warmup=args.warmup,
            max_batch_size=args.max_batch_size,
            kv_block_size=args.kv_block_size,
            kv_num_blocks=args.kv_pool_tokens // args.kv_block_size,
            request_csv=request_csv,
            timeline_csv=timeline_csv,
            batch_csv=batch_csv,
        )
        phase = f"{variant}:{config_id}:r{repeat}"
        trace.set_phase(phase)
        trace.wait_for_sample(phase, timeout_seconds=max(5.0, args.gpu_sample_interval * 4.0))
        elapsed_ms, return_code, started_at, completed_at = run_logged(
            command, cwd=repo, log=log
        )
        sequence_index += 1
        process_rows.append(
            {
                "sequence_index": sequence_index,
                "kind": configuration["category"],
                "config_id": config_id,
                "variant": variant,
                "repeat": repeat,
                "ab_order": order,
                "request_count": configuration["request_count"],
                "max_active_requests": configuration["max_active_requests"],
                "total_requested_tokens": sum(configuration["output_limits"]),
                "process_wall_ms": f"{elapsed_ms:.6f}",
                "return_code": return_code,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "requests_csv": str(request_csv.relative_to(run_dir)),
                "timeline_csv": str(timeline_csv.relative_to(run_dir)),
                "batches_csv": str(batch_csv.relative_to(run_dir)) if batch_csv else "",
                "log": str(log.relative_to(run_dir)),
                "command": shlex.join(command),
            }
        )
        write_process_rows(process_path, process_rows)
        if return_code != 0:
            raise RuntimeError(f"runtime failed ({return_code}); see {log}")
        if not timeline_csv.is_file() or timeline_csv.stat().st_size == 0:
            raise RuntimeError(f"runtime did not create timeline evidence: {timeline_csv}")
        if variant == "continuous" and (
            batch_csv is None or not batch_csv.is_file() or batch_csv.stat().st_size == 0
        ):
            raise RuntimeError(f"runtime did not create batch evidence: {batch_csv}")
        rows = validate_request_csv(request_csv, configuration, workloads)
        validated_rows[(config_id, variant, repeat)] = rows

    try:
        # Fail fast on long heterogeneous correctness before spending time on
        # the repeated performance matrix.
        for variant in VARIANTS:
            execute(correctness_config, variant, 1, "serial>continuous")
        validate_pair(
            validated_rows[("correctness_long", "serial", 1)],
            validated_rows[("correctness_long", "continuous", 1)],
        )

        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(performance_configs)
            ordered_configs = performance_configs[offset:] + performance_configs[:offset]
            if repeat % 2 == 0:
                ordered_configs.reverse()
            variant_order = VARIANTS if repeat % 2 == 1 else tuple(reversed(VARIANTS))
            order_label = ">".join(variant_order)
            for configuration in ordered_configs:
                for variant in variant_order:
                    execute(configuration, variant, repeat, order_label)
                config_id = str(configuration["config_id"])
                validate_pair(
                    validated_rows[(config_id, "serial", repeat)],
                    validated_rows[(config_id, "continuous", repeat)],
                )
    finally:
        trace.stop()

    run_checked(
        [sys.executable, str(analyzer), "aggregate", "--run-dir", str(run_dir)], cwd=repo
    )
    run_checked(
        [
            sys.executable,
            str(analyzer),
            "verify",
            "--run-dir",
            str(run_dir),
            "--final",
        ],
        cwd=repo,
    )
    print(f"completed immutable v1.0 continuous-batching benchmark: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
