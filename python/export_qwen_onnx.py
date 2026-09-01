#!/usr/bin/env python3
"""Export explicit MiniLLM-RT ONNX contracts for Qwen2.5.

The wrappers intentionally keep the external attention_mask contract 2-D while
building a 4-D additive causal mask internally.  Passing that mask to the HF
model avoids the nested-vmap mask builder which is not traceable by the legacy
PyTorch exporter in Transformers 4.53.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import onnx
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from qwen_common import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MODEL_DIR,
    build_chat,
    load_config,
    load_model,
    load_tokenizer,
    write_model_spec,
)


def causal_mask_4d(
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    value_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert [B,K] padding mask into additive causal [B,1,Q,K]."""
    key_positions = torch.arange(
        attention_mask.shape[-1], device=attention_mask.device, dtype=position_ids.dtype
    ).view(1, 1, -1)
    allowed = key_positions <= position_ids.unsqueeze(-1)
    allowed = allowed & attention_mask.to(torch.bool).unsqueeze(1)
    mask = torch.zeros(allowed.shape, dtype=value_dtype, device=attention_mask.device)
    mask = mask.masked_fill(~allowed, torch.finfo(value_dtype).min)
    return mask.unsqueeze(1)


def decode_padding_mask_4d(
    attention_mask: torch.Tensor,
    value_dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B,1,1,K] for a one-token padded decode query.

    Batched decode appends the current key at physical index ``K - 1`` while
    each row can have a different logical position.  A physical
    ``key_index <= position_id`` test would therefore hide the current token
    for shorter rows.  There are no future keys for a one-token query, so the
    per-row key-padding mask is the complete causal contract.
    """
    allowed = attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
    mask = torch.zeros(allowed.shape, dtype=value_dtype, device=attention_mask.device)
    return mask.masked_fill(~allowed, torch.finfo(value_dtype).min)


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        mask = causal_mask_4d(attention_mask, position_ids, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            cache_position=position_ids[0],
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[:, -1:, :]


class PrefillWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        mask = causal_mask_4d(attention_mask, position_ids, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            cache_position=position_ids[0],
            use_cache=True,
            return_dict=True,
        )
        flat_cache = []
        for key, value in outputs.past_key_values:
            flat_cache.extend((key, value))
        return (outputs.logits[:, -1:, :], *flat_cache)


class WithPastWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, *past):
        legacy_cache = tuple(
            (past[2 * layer], past[2 * layer + 1]) for layer in range(self.num_layers)
        )
        cache = DynamicCache.from_legacy_cache(legacy_cache)
        mask = causal_mask_4d(attention_mask, position_ids, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=position_ids[0],
            use_cache=True,
            return_dict=True,
        )
        flat_cache = []
        for key, value in outputs.past_key_values:
            flat_cache.extend((key, value))
        return (outputs.logits[:, -1:, :], *flat_cache)


class BatchedWithPastWrapper(WithPastWrapper):
    """Padding-aware one-token decode used by the v1.0 batch Engine."""

    def forward(self, input_ids, attention_mask, position_ids, *past):
        legacy_cache = tuple(
            (past[2 * layer], past[2 * layer + 1]) for layer in range(self.num_layers)
        )
        cache = DynamicCache.from_legacy_cache(legacy_cache)
        mask = decode_padding_mask_4d(attention_mask, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask},
            position_ids=position_ids,
            # DynamicCache ignores cache_position.  The explicit per-row
            # position_ids drive RoPE; this singleton only satisfies the HF
            # model signature used during export.
            cache_position=position_ids[0],
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        flat_cache = []
        for key, value in outputs.past_key_values:
            flat_cache.extend((key, value))
        return (outputs.logits[:, -1:, :], *flat_cache)


def cache_names(prefix: str, num_layers: int) -> list[str]:
    return [
        f"{prefix}{layer}.{kind}"
        for layer in range(num_layers)
        for kind in ("key", "value")
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--mode",
        choices=["logits", "prefill", "decode", "decode_batch", "all"],
        default="all",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-sequence-length", type=int, default=8)
    parser.add_argument("--sample-past-length", type=int, default=8)
    parser.add_argument("--engine-max-sequence-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--inspect-existing",
        action="store_true",
        help="Validate existing graphs and refresh export manifests without re-exporting weights.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_info(value) -> dict[str, object]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.dim_param:
            dimensions.append(dimension.dim_param)
        elif dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append(None)
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type).lower(),
        "shape": dimensions,
    }


def write_export_metadata(
    output_path: Path,
    mode: str,
    args: argparse.Namespace,
    elapsed_seconds: float | None,
    inspected_existing: bool,
) -> dict[str, object]:
    graph = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(graph)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "dtype": args.dtype,
        "opset": args.opset,
        "path": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "elapsed_seconds": elapsed_seconds,
        "inspected_existing": inspected_existing,
        "producer_name": graph.producer_name,
        "ir_version": graph.ir_version,
        "inputs": [value_info(value) for value in graph.graph.input],
        "outputs": [value_info(value) for value in graph.graph.output],
    }
    manifest_path = output_path.with_suffix(".export.json")
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --force")
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def export_one(model, config, args, mode: str, output_dir: Path):
    num_layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    dtype = model.dtype
    batch = 1
    sequence = args.sample_sequence_length
    input_ids = torch.arange(1, sequence + 1, dtype=torch.int64).view(batch, sequence)
    attention_mask = torch.ones((batch, sequence), dtype=torch.int64)
    position_ids = torch.arange(sequence, dtype=torch.int64).view(batch, sequence)
    inputs: tuple[torch.Tensor, ...]
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    if mode == "logits":
        wrapper = LogitsOnlyWrapper(model)
        inputs = (input_ids, attention_mask, position_ids)
    elif mode == "prefill":
        wrapper = PrefillWrapper(model)
        inputs = (input_ids, attention_mask, position_ids)
        present_names = cache_names("present.", num_layers)
        output_names.extend(present_names)
        for name in present_names:
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
    else:
        batch = 2 if mode == "decode_batch" else 1
        wrapper = (
            BatchedWithPastWrapper(model, num_layers)
            if mode == "decode_batch"
            else WithPastWrapper(model, num_layers)
        )
        past_len = args.sample_past_length
        input_ids = torch.ones((batch, 1), dtype=torch.int64)
        attention_mask = torch.ones((batch, past_len + 1), dtype=torch.int64)
        position_ids = torch.full((batch, 1), past_len, dtype=torch.int64)
        if mode == "decode_batch" and past_len > 2:
            shorter = past_len - 2
            attention_mask[1, shorter:past_len] = 0
            position_ids[1, 0] = shorter
        past = tuple(
            torch.zeros((batch, kv_heads, past_len, head_dim), dtype=dtype)
            for _ in range(2 * num_layers)
        )
        inputs = (input_ids, attention_mask, position_ids, *past)
        past_names = cache_names("past_key_values.", num_layers)
        present_names = cache_names("present.", num_layers)
        input_names.extend(past_names)
        output_names.extend(present_names)
        # Decode consumes exactly one token; only batch and past/total lengths vary.
        dynamic_axes["input_ids"] = {0: "batch_size"}
        dynamic_axes["position_ids"] = {0: "batch_size"}
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "total_sequence_length"}
        for name in past_names:
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        for name in present_names:
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}

    output_path = output_dir / f"qwen2_5_0_5b_{mode}.onnx"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")
    if args.force:
        output_path.unlink(missing_ok=True)
        Path(str(output_path) + ".data").unlink(missing_ok=True)

    wrapper.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
    elapsed = time.perf_counter() - started
    write_export_metadata(output_path, mode, args, elapsed, False)


def main():
    args = parse_args()
    if args.sample_sequence_length <= 0 or args.sample_past_length <= 0:
        raise ValueError("sample lengths must be positive")
    if args.engine_max_sequence_length <= 1:
        raise ValueError("engine max sequence length must exceed one")

    output_dir = args.output_root / ("fp32_correctness" if args.dtype == "float32" else "fp16_deployment")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_model_spec(
        output_dir / "model.spec",
        args.model,
        args.dtype,
        args.engine_max_sequence_length,
    )
    config = load_config(args.model)
    # Keep ``all`` backward compatible with the v0.9 three-graph pipeline;
    # the v1.0 batch graph is requested explicitly.
    modes = ["logits", "prefill", "decode"] if args.mode == "all" else [args.mode]
    if args.inspect_existing:
        for mode in modes:
            output_path = output_dir / f"qwen2_5_0_5b_{mode}.onnx"
            if not output_path.is_file():
                raise FileNotFoundError(output_path)
            write_export_metadata(output_path, mode, args, None, True)
        print(f"model spec: {manifest}")
        return

    model = load_model(args.model, args.dtype, "cpu")
    for mode in modes:
        export_one(model, config, args, mode, output_dir)
    print(f"model spec: {manifest}")


if __name__ == "__main__":
    main()
