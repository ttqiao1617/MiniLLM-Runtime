#!/usr/bin/env python3
"""Run the immutable Qwen v0.9 full-recompute/KV/scheduler benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from qwen_common import DEFAULT_ARTIFACT_ROOT, DEFAULT_MODEL_DIR, MODEL_ID, MODEL_REVISION


DEFAULT_SPEC = Path("configs/qwen2_5_0_5b_fp16.spec")
DEFAULT_WORKLOADS = Path("inputs/qwen_v09/workloads.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


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
        suffix = f"; see {log}" if log else ""
        raise RuntimeError(f"command failed ({completed.returncode}){suffix}: {shlex.join(command)}")
    return elapsed_ms


class GpuTrace:
    FIELDS = (
        "timestamp,name,pstate,temperature.gpu,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory,memory.used,memory.total"
    )

    def __init__(self, output: Path, interval_seconds: float):
        self.output = output
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.phase = "setup"
        self.lock = threading.Lock()

    def set_phase(self, value: str) -> None:
        with self.lock:
            self.phase = value

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="v09-gpu-trace", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(5.0, self.interval * 4))

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
                self.stop_event.wait(self.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=Path("results/runs/v09"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--binary", type=Path, default=Path("build/minillm_rt"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--scheduler-repeats", type=int, default=10)
    parser.add_argument("--scheduler-active", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--scheduler-block-sizes", type=parse_int_list, default=[8, 16, 32])
    parser.add_argument("--scheduler-pool-tokens", type=int, default=2048)
    parser.add_argument("--notes", default="")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    run_dir = (args.run_root / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable run: {run_dir}")
    if args.warmup < 5 or args.repeats < 10:
        raise ValueError("official v0.9 runs require warmup>=5 and repeats>=10")
    if args.scheduler_repeats < 10:
        raise ValueError("official v0.9 scheduler runs require at least 10 repeats")

    engines = args.artifact_root / "engines"
    baseline_engine = engines / "qwen2_5_0_5b_logits.plan"
    prefill_engine = engines / "qwen2_5_0_5b_prefill.plan"
    decode_engine = engines / "qwen2_5_0_5b_decode.plan"
    required = (
        args.binary,
        args.model_spec,
        args.workloads,
        baseline_engine,
        prefill_engine,
        decode_engine,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing benchmark artifacts: " + ", ".join(missing))

    manifest = json.loads(args.workloads.read_text(encoding="utf-8"))
    controlled_cases = [
        case for case in manifest["cases"] if case["kind"] == "length_controlled_synthetic"
    ]
    if [int(case["prompt_length"]) for case in controlled_cases] != [8, 32, 128, 256]:
        raise RuntimeError("official workload must contain controlled prompt lengths 8/32/128/256")
    real_case_ids = {"kv_cache_en", "eos_instruction"}
    real_cases = [case for case in manifest["cases"] if case["case_id"] in real_case_ids]
    if {str(case["case_id"]) for case in real_cases} != real_case_ids:
        raise RuntimeError("official workload must contain kv_cache_en and eos_instruction real prompts")
    cases = controlled_cases + real_cases
    for case in cases:
        path = repo / str(case["input_ids_path"])
        case["resolved_input_ids"] = str(path.resolve())
        case["golden"] = str((repo / "inputs/qwen_v09/goldens" / f"{case['case_id']}.json").resolve())
        if not path.is_file() or not Path(case["golden"]).is_file():
            raise FileNotFoundError(f"missing input/golden for {case['case_id']}")
    eos_case = next(case for case in real_cases if case["case_id"] == "eos_instruction")
    scheduler_cases = controlled_cases * 2 + [eos_case]
    scheduler_output_limits = [8, 16, 32, 32, 16, 8, 32, 16, 16]

    prepare = [
        sys.executable,
        "python/analyze_v09_qwen.py",
        "prepare",
        "--run-dir",
        str(run_dir),
        "--run-id",
        args.run_id,
        "--model-id",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--precision",
        "fp16-strongly-typed",
        "--model-dir",
        str(args.model_dir.resolve()),
        "--model-spec",
        str(args.model_spec.resolve()),
        "--baseline-engine",
        str(baseline_engine.resolve()),
        "--prefill-engine",
        str(prefill_engine.resolve()),
        "--decode-engine",
        str(decode_engine.resolve()),
        "--runtime-binary",
        str(args.binary.resolve()),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--scheduler-active",
        ",".join(str(value) for value in args.scheduler_active),
        "--scheduler-block-sizes",
        ",".join(str(value) for value in args.scheduler_block_sizes),
        "--scheduler-repeats",
        str(args.scheduler_repeats),
        "--scheduler-request-count",
        str(len(scheduler_cases)),
        "--scheduler-case-ids",
        ",".join(str(case["case_id"]) for case in scheduler_cases),
        "--scheduler-output-limits",
        ",".join(str(value) for value in scheduler_output_limits),
        "--scheduler-pool-tokens",
        str(args.scheduler_pool_tokens),
        "--notes",
        args.notes,
        "--runner-command",
        shlex.join(sys.argv),
    ]
    for case in cases:
        prepare.extend(
            [
                "--input-spec",
                f"{case['case_id']}:{case['prompt_length']}:{case['requested_new_tokens']}:"
                f"{case['resolved_input_ids']}",
                "--golden",
                case["golden"],
            ]
        )
    if args.dry_run:
        print(shlex.join(prepare))
        print("dry-run: benchmark commands omitted after validated manifest")
        return

    run_checked(prepare, cwd=repo)
    run_checked(
        [
            sys.executable,
            "python/capture_environment.py",
            "--output",
            str(run_dir / "environment.json"),
            "--repo",
            str(repo),
            "--notes",
            args.notes,
        ],
        cwd=repo,
    )

    raw_performance = run_dir / "raw/performance"
    raw_scheduler = run_dir / "raw/scheduler"
    raw_logs = run_dir / "raw/logs"
    for directory in (raw_performance, raw_scheduler, raw_logs):
        directory.mkdir(parents=True, exist_ok=True)
    trace = GpuTrace(run_dir / "raw/nvml/gpu_trace.csv", args.gpu_sample_interval)
    process_rows: list[dict[str, object]] = []
    trace.start()
    try:
        for case in cases:
            case_id = str(case["case_id"])
            prompt = int(case["prompt_length"])
            output = int(case["requested_new_tokens"])
            for repeat in range(1, args.repeats + 1):
                variants = ["full", "kv"] if repeat % 2 else ["kv", "full"]
                for variant in variants:
                    stem = f"v09_{variant}_{case_id}_p{prompt}_o{output}_r{repeat}"
                    csv_path = raw_performance / f"{stem}.csv"
                    log_path = raw_logs / f"{stem}.log"
                    command = [
                        str(args.binary.resolve()),
                        "--model",
                        str((baseline_engine if variant == "full" else prefill_engine).resolve()),
                        "--model-spec",
                        str(args.model_spec.resolve()),
                        "--input-ids",
                        str(case["resolved_input_ids"]),
                        "--mode",
                        "naive" if variant == "full" else "kv_cache",
                        "--provider",
                        "tensorrt",
                        "--sampling",
                        "host",
                        "--max-new-tokens",
                        str(output),
                        "--warmup-runs",
                        str(args.warmup),
                        "--output-csv",
                        str(csv_path),
                    ]
                    if variant == "kv":
                        command[3:3] = ["--decode-model", str(decode_engine.resolve())]
                    trace.set_phase(stem)
                    elapsed_ms = run_checked(command, cwd=repo, log=log_path)
                    process_rows.append(
                        {
                            "kind": "performance",
                            "case_id": case_id,
                            "variant": variant,
                            "repeat": repeat,
                            "process_wall_ms": f"{elapsed_ms:.6f}",
                            "csv": str(csv_path),
                            "log": str(log_path),
                        }
                    )

        if args.scheduler_repeats:
            request_paths = ",".join(str(case["resolved_input_ids"]) for case in scheduler_cases)
            # Mixed output limits exercise independent request completion without
            # pretending that serial round-robin enqueue is GPU continuous batching.
            output_limits = scheduler_output_limits
            configurations = [
                (active, block_size)
                for active in args.scheduler_active
                for block_size in args.scheduler_block_sizes
            ]
            for repeat in range(1, args.scheduler_repeats + 1):
                offset = (repeat - 1) % len(configurations)
                ordered_configurations = configurations[offset:] + configurations[:offset]
                if repeat % 2 == 0:
                    ordered_configurations.reverse()
                for active, block_size in ordered_configurations:
                    if args.scheduler_pool_tokens % block_size != 0:
                        raise ValueError("scheduler pool tokens must be divisible by every block size")
                    num_blocks = args.scheduler_pool_tokens // block_size
                    stem = f"v09_scheduler_a{active}_b{block_size}_r{repeat}"
                    request_csv = raw_scheduler / f"{stem}_requests.csv"
                    timeline_csv = raw_scheduler / f"{stem}_timeline.csv"
                    log_path = raw_logs / f"{stem}.log"
                    command = [
                        str(args.binary.resolve()),
                        "--model", str(prefill_engine.resolve()),
                        "--decode-model", str(decode_engine.resolve()),
                        "--model-spec", str(args.model_spec.resolve()),
                        "--request-inputs", request_paths,
                        "--request-max-new-tokens",
                        ",".join(str(value) for value in output_limits),
                        "--mode", "scheduler",
                        "--provider", "tensorrt",
                        "--sampling", "host",
                        "--max-new-tokens", str(max(output_limits)),
                        "--warmup-runs", str(args.warmup),
                        "--max-active-requests", str(active),
                        "--max-tokens-per-step", str(active),
                        "--kv-block-size", str(block_size),
                        "--kv-num-blocks", str(num_blocks),
                        "--timeline-csv", str(timeline_csv),
                        "--request-summary-csv", str(request_csv),
                    ]
                    trace.set_phase(stem)
                    elapsed_ms = run_checked(command, cwd=repo, log=log_path)
                    process_rows.append(
                        {
                            "kind": "scheduler",
                            "case_id": f"a{active}_b{block_size}",
                            "variant": "serial_round_robin",
                            "repeat": repeat,
                            "process_wall_ms": f"{elapsed_ms:.6f}",
                            "csv": str(request_csv),
                            "log": str(log_path),
                        }
                    )
    finally:
        trace.stop()

    process_csv = run_dir / "raw/process_runs.csv"
    with process_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(process_rows[0]))
        writer.writeheader()
        writer.writerows(process_rows)

    run_checked(
        [sys.executable, "python/analyze_v09_qwen.py", "aggregate", "--run-dir", str(run_dir)],
        cwd=repo,
    )
    raw_artifacts = []
    for path in sorted((run_dir / "raw").rglob("*")):
        if path.is_file():
            raw_artifacts.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    evidence_names = (
        "environment.json",
        "model_manifest.json",
        "benchmark_config.json",
        "correctness_summary.csv",
        "performance_summary.csv",
        "decode_step_summary.csv",
        "scheduler_summary.csv",
        "memory_summary.csv",
        "process_summary.csv",
        "README.md",
    )
    evidence = []
    for name in evidence_names:
        path = run_dir / name
        evidence.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "raw_artifacts": raw_artifacts, "evidence": evidence},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run_checked(
        [
            sys.executable,
            "python/analyze_v09_qwen.py",
            "verify",
            "--run-dir",
            str(run_dir),
            "--final",
        ],
        cwd=repo,
    )
    print(f"completed immutable v0.9 benchmark: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
