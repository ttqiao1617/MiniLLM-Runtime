#!/usr/bin/env python3
"""Build strongly typed TensorRT engines used by Qwen v0.9/v1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from qwen_common import DEFAULT_ARTIFACT_ROOT


ENGINE_FILES = {
    "logits": "qwen2_5_0_5b_logits.plan",
    "prefill": "qwen2_5_0_5b_prefill.plan",
    "decode": "qwen2_5_0_5b_decode.plan",
    "decode_batch": "qwen2_5_0_5b_decode_batch.plan",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_trtexec(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        if path.is_file():
            return path.resolve()
        found = shutil.which(raw)
        if found:
            return Path(found).resolve()
        raise FileNotFoundError(f"trtexec not found: {raw}")
    for candidate in (
        shutil.which("trtexec"),
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
    ):
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError("trtexec not found; pass --trtexec")


def tensor(name: str, *dimensions: int) -> str:
    return f"{name}:{'x'.join(str(value) for value in dimensions)}"


def profiles(
    mode: str,
    maximum: int,
    max_batch_size: int,
    opt_batch_size: int,
) -> tuple[str, str, str]:
    if mode not in {"decode", "decode_batch"}:
        def shapes(sequence: int) -> str:
            entries = [
                tensor("input_ids", 1, sequence),
                tensor("attention_mask", 1, sequence),
                tensor("position_ids", 1, sequence),
            ]
            return ",".join(entries)

        return shapes(1), shapes(min(32, maximum)), shapes(maximum)

    layers = 24

    def shapes(past: int, batch: int = 1) -> str:
        entries = [
            tensor("input_ids", batch, 1),
            tensor("attention_mask", batch, past + 1),
            tensor("position_ids", batch, 1),
        ]
        for layer in range(layers):
            entries.extend(
                (
                    tensor(f"past_key_values.{layer}.key", batch, 2, past, 64),
                    tensor(f"past_key_values.{layer}.value", batch, 2, past, 64),
                )
            )
        return ",".join(entries)

    if mode == "decode_batch":
        return (
            shapes(1, 1),
            shapes(min(128, maximum - 1), opt_batch_size),
            shapes(maximum - 1, max_batch_size),
        )
    return shapes(1), shapes(min(32, maximum - 1)), shapes(maximum - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "fp16_deployment"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT / "engines"
    )
    parser.add_argument("--trtexec")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--opt-batch-size", type=int, default=4)
    parser.add_argument(
        "--mode", action="append", choices=tuple(ENGINE_FILES), help="Repeat to build a subset."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--inspect-existing",
        action="store_true",
        help="Hash and record an existing engine with its reproducible build command.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequence_length < 2:
        raise ValueError("--max-sequence-length must be at least 2")
    if args.max_batch_size < 1 or args.opt_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.opt_batch_size > args.max_batch_size:
        raise ValueError("--opt-batch-size cannot exceed --max-batch-size")
    trtexec = find_trtexec(args.trtexec)
    # Preserve the v0.9 default; v1.0 batch decode is an explicit artifact.
    modes = args.mode or ["logits", "prefill", "decode"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(
        [str(trtexec), "--version"], check=False, capture_output=True, text=True
    )
    version_lines = (version.stdout + "\n" + version.stderr).splitlines()
    version_summary = next(
        (line.strip() for line in version_lines if "TensorRT v" in line),
        "unknown",
    )

    for mode in modes:
        onnx_path = args.onnx_dir / f"qwen2_5_0_5b_{mode}.onnx"
        engine_path = args.output_dir / ENGINE_FILES[mode]
        log_path = engine_path.with_suffix(".build.log")
        manifest_path = engine_path.with_suffix(".build.json")
        inspection_path = engine_path.with_suffix(".engine.json")
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        outputs_to_protect = (engine_path, log_path, manifest_path)
        if args.inspect_existing:
            outputs_to_protect = (inspection_path,)
        for output in outputs_to_protect:
            if output.exists() and not args.force and not args.dry_run:
                raise FileExistsError(f"refusing to overwrite {output}; pass --force")

        minimum, optimum, maximum = profiles(
            mode,
            args.max_sequence_length,
            args.max_batch_size,
            args.opt_batch_size,
        )
        command = [
            str(trtexec),
            f"--onnx={onnx_path.resolve()}",
            f"--saveEngine={engine_path.resolve()}",
            f"--minShapes={minimum}",
            f"--optShapes={optimum}",
            f"--maxShapes={maximum}",
            "--stronglyTyped",
            "--noTF32",
            "--builderOptimizationLevel=3",
            "--profilingVerbosity=detailed",
            "--skipInference",
        ]
        print(" ".join(command))
        if args.dry_run:
            continue
        if args.inspect_existing:
            if not engine_path.is_file():
                raise FileNotFoundError(engine_path)
            inspection = {
                "schema_version": 1,
                "mode": mode,
                "precision_contract": "strongly_typed_fp16_onnx",
                "tf32_enabled": False,
                "max_sequence_length": args.max_sequence_length,
                "max_batch_size": args.max_batch_size if mode == "decode_batch" else 1,
                "opt_batch_size": args.opt_batch_size if mode == "decode_batch" else 1,
                "reproducible_build_command": command,
                "built_by_this_invocation": False,
                "trtexec_version": version_summary,
                "onnx": {
                    "path": str(onnx_path.resolve()),
                    "size_bytes": onnx_path.stat().st_size,
                    "sha256": sha256_file(onnx_path),
                },
                "engine": {
                    "path": str(engine_path.resolve()),
                    "size_bytes": engine_path.stat().st_size,
                    "sha256": sha256_file(engine_path),
                },
            }
            inspection_path.write_text(
                json.dumps(inspection, indent=2) + "\n", encoding="utf-8"
            )
            print(f"recorded existing {mode} engine: {inspection_path}")
            continue

        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elapsed = time.perf_counter() - started
        if process.returncode != 0 or not engine_path.is_file():
            raise RuntimeError(
                f"TensorRT build failed for {mode} (exit {process.returncode}); see {log_path}"
            )
        manifest = {
            "schema_version": 1,
            "mode": mode,
            "precision_contract": "strongly_typed_fp16_onnx",
            "tf32_enabled": False,
            "max_sequence_length": args.max_sequence_length,
            "max_batch_size": args.max_batch_size if mode == "decode_batch" else 1,
            "opt_batch_size": args.opt_batch_size if mode == "decode_batch" else 1,
            "command": command,
            "trtexec_version": version_summary,
            "build_seconds": elapsed,
            "onnx": {
                "path": str(onnx_path.resolve()),
                "size_bytes": onnx_path.stat().st_size,
                "sha256": sha256_file(onnx_path),
            },
            "engine": {
                "path": str(engine_path.resolve()),
                "size_bytes": engine_path.stat().st_size,
                "sha256": sha256_file(engine_path),
            },
            "build_log": str(log_path.resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"built {mode}: {engine_path} ({engine_path.stat().st_size / 2**20:.1f} MiB, "
            f"{elapsed:.1f}s)"
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
