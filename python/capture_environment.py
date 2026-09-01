#!/usr/bin/env python3
"""Capture a machine-readable environment snapshot for benchmark evidence.

The collector is deliberately best-effort: an unavailable optional command is
recorded as unavailable instead of making the benchmark setup fail.  The
output itself is immutable by default so an official run cannot silently
replace the environment under which it was measured.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PACKAGE_NAMES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "torch",
    "transformers",
    "safetensors",
    "tensorrt",
)


def command_result(argv: Sequence[str], cwd: Path | None = None) -> dict[str, Any]:
    """Run a diagnostic command without a shell and retain failure context."""
    executable = shutil.which(argv[0])
    if executable is None and not Path(argv[0]).is_file():
        return {"available": False, "argv": list(argv)}
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "argv": list(argv),
            "error": f"{type(exc).__name__}: {exc}",
        }
    result: dict[str, Any] = {
        "available": True,
        "argv": list(argv),
        "returncode": completed.returncode,
    }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        result["stdout"] = stdout
    if stderr:
        result["stderr"] = stderr
    return result


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    text = read_text(Path("/etc/os-release"))
    if not text:
        return values
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def cpu_info() -> dict[str, Any]:
    model = None
    text = read_text(Path("/proc/cpuinfo"))
    if text:
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    return {
        "model": model or platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
    }


def memory_info() -> dict[str, int]:
    result: dict[str, int] = {}
    text = read_text(Path("/proc/meminfo"))
    if not text:
        return result
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    for line in text.splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2 and parts[0] in wanted:
            # Linux reports these values in KiB.
            result[f"{parts[0]}_bytes"] = int(parts[1]) * 1024
    return result


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def python_runtime_details(include_host_identifiers: bool = False) -> dict[str, Any]:
    details: dict[str, Any] = {
        "version": platform.python_version(),
        "executable": sys.executable if include_host_identifiers else Path(sys.executable).name,
        "implementation": platform.python_implementation(),
        "packages": package_versions(),
    }
    try:
        import torch

        details["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "visible_device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # optional dependency/runtime may be broken
        details["torch_runtime"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        import tensorrt

        details["tensorrt_python_version"] = tensorrt.__version__
    except Exception as exc:  # optional dependency/runtime may be broken
        details["tensorrt_python_error"] = f"{type(exc).__name__}: {exc}"
    return details


def gpu_details(include_host_identifiers: bool = False) -> dict[str, Any]:
    identity_field = "uuid," if include_host_identifiers else ""
    query_fields = (
        f"name,{identity_field}driver_version,pstate,persistence_mode,temperature.gpu,"
        "power.draw,power.limit,clocks.current.sm,clocks.current.memory,"
        "memory.total,memory.used,memory.free"
    )
    return {
        "snapshot": command_result(
            [
                "nvidia-smi",
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            ]
        ),
        "query_fields": query_fields.split(","),
        "compute_processes": command_result(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ) if include_host_identifiers else {"redacted": True},
    }


def find_trtexec() -> str:
    found = shutil.which("trtexec")
    if found:
        return found
    candidates = (
        Path("/usr/local/tensorrt/bin/trtexec"),
        Path("/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "trtexec"


def git_details(repo: Path, include_host_identifiers: bool = False) -> dict[str, Any]:
    inside = command_result(["git", "rev-parse", "--is-inside-work-tree"], repo)
    if inside.get("returncode") != 0:
        return {
            "repo": str(repo.resolve()) if include_host_identifiers else repo.name,
            "available": False,
            "probe": inside,
        }
    commit = command_result(["git", "rev-parse", "HEAD"], repo)
    short_commit = command_result(["git", "rev-parse", "--short=12", "HEAD"], repo)
    branch = command_result(["git", "branch", "--show-current"], repo)
    describe = command_result(["git", "describe", "--always", "--dirty", "--tags"], repo)
    status = command_result(["git", "status", "--porcelain=v1"], repo)
    status_text = str(status.get("stdout", ""))
    return {
        "repo": str(repo.resolve()) if include_host_identifiers else repo.name,
        "available": True,
        "commit": commit.get("stdout"),
        "short_commit": short_commit.get("stdout"),
        "branch": branch.get("stdout"),
        "describe": describe.get("stdout"),
        "dirty": bool(status_text),
        "status_porcelain": status_text.splitlines(),
    }


def build_snapshot(
    repo: Path, notes: str, include_host_identifiers: bool = False
) -> dict[str, Any]:
    cuda_version_files = {}
    for candidate in (Path("/usr/local/cuda/version.json"), Path("/usr/local/cuda/version.txt")):
        value = read_text(candidate)
        if value is not None:
            cuda_version_files[str(candidate)] = value
    selected_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "CUDA_MODULE_LOADING",
            "TRT_LOGGER_SEVERITY",
        )
    }
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_release": os_release(),
            "container_detected": Path("/.dockerenv").exists(),
            "cpu": cpu_info(),
            "memory": memory_info(),
        },
        "gpu": gpu_details(include_host_identifiers),
        "cuda": {
            "nvcc": command_result(["nvcc", "--version"]),
            "version_files": cuda_version_files,
        },
        "tensorrt": {
            "trtexec": command_result([find_trtexec(), "--version"]),
        },
        "python": python_runtime_details(include_host_identifiers),
        "git": git_details(repo, include_host_identifiers),
        "selected_environment": selected_environment,
        "collector_argv": sys.argv if include_host_identifiers else [Path(sys.argv[0]).name],
    }


def write_json_atomic(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing environment snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the software, GPU, clock/power and Git state for a benchmark run."
    )
    parser.add_argument("--output", type=Path, required=True, help="environment.json path")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository root")
    parser.add_argument(
        "--notes",
        default="",
        help="Free-form load/power/cooling note supplied by the benchmark operator",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing snapshot (not used by official run scripts)",
    )
    parser.add_argument(
        "--include-host-identifiers",
        action="store_true",
        help="Include hostname, GPU UUID, process list and absolute local paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build_snapshot(
        args.repo.resolve(), args.notes, args.include_host_identifiers
    )
    write_json_atomic(args.output, snapshot, args.overwrite)
    print(f"environment snapshot: {args.output}")


if __name__ == "__main__":
    main()
