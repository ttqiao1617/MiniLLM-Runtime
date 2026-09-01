#!/usr/bin/env python3
"""Create a deterministic Qwen chat input and HF greedy-decode golden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_common import (
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    MODEL_REVISION,
    build_chat,
    eos_token_ids,
    load_model,
    load_tokenizer,
)


DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_USER = "Briefly explain what a KV cache does in LLM inference."


def greedy_decode(model, encoded, max_new_tokens: int, stop_ids: set[int], use_cache: bool):
    device = encoded["input_ids"].device
    generated = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    position_ids = encoded["position_ids"]
    cache = None
    generated_ids: list[int] = []
    step_top5: list[list[dict[str, float | int]]] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            cached_step = use_cache and cache is not None
            model_ids = generated[:, -1:] if cached_step else generated
            model_positions = position_ids[:, -1:] if cached_step else position_ids
            outputs = model(
                input_ids=model_ids,
                attention_mask=attention_mask,
                position_ids=model_positions,
                past_key_values=cache,
                use_cache=use_cache,
                return_dict=True,
            )
            last = outputs.logits[:, -1, :].to(torch.float32)
            values, indices = torch.topk(last, k=5, dim=-1)
            next_id = int(indices[0, 0].item())
            step_top5.append([
                {"token_id": int(token_id), "logit": float(value)}
                for token_id, value in zip(indices[0].tolist(), values[0].tolist())
            ])
            generated_ids.append(next_id)
            next_token = torch.tensor([[next_id]], dtype=torch.int64, device=device)
            generated = torch.cat((generated, next_token), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)
            position_ids = torch.cat((position_ids, position_ids[:, -1:] + 1), dim=-1)
            cache = outputs.past_key_values if use_cache else None
            if next_id in stop_ids:
                break
    return generated, generated_ids, step_top5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--input-ids",
        type=Path,
        help="Optional pre-tokenized workload; bypasses chat-template construction.",
    )
    parser.add_argument("--case-id", default="qwen_chat")
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("inputs/qwen_golden.json"))
    parser.add_argument("--ids-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.dtype == "float16" and args.device == "cpu":
        raise ValueError("float16 baseline requires --device cuda")

    tokenizer = load_tokenizer(args.model)
    if args.input_ids is None:
        rendered, encoded = build_chat(tokenizer, args.system, args.user)
    else:
        values = [int(value) for value in args.input_ids.read_text(encoding="utf-8").split()]
        if not values:
            raise ValueError(f"empty token-ID workload: {args.input_ids}")
        input_ids = torch.tensor([values], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": torch.arange(len(values), dtype=torch.int64).unsqueeze(0),
        }
        rendered = None
    model = load_model(args.model, args.dtype, args.device)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    position_ids = encoded["position_ids"].to(args.device)
    stop_ids = set(eos_token_ids(args.model))
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    generated, generated_ids, step_top5 = greedy_decode(
        model, model_inputs, args.max_new_tokens, stop_ids, use_cache=True
    )
    _, recompute_ids, _ = greedy_decode(
        model, model_inputs, args.max_new_tokens, stop_ids, use_cache=False
    )
    if generated_ids != recompute_ids:
        raise RuntimeError(
            f"HF correctness gate failed: cache={generated_ids}, no-cache={recompute_ids}"
        )

    input_list = encoded["input_ids"][0].tolist()
    data = {
        "case_id": args.case_id,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "system": args.system if args.input_ids is None else None,
        "user": args.user if args.input_ids is None else None,
        "source_input_ids": str(args.input_ids.resolve()) if args.input_ids else None,
        "rendered_chat": rendered,
        "input_ids": input_list,
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "position_ids": encoded["position_ids"][0].tolist(),
        "prompt_length": len(input_list),
        "requested_new_tokens": args.max_new_tokens,
        "generated_ids": generated_ids,
        "no_cache_generated_ids": recompute_ids,
        "cache_vs_no_cache_match": True,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "full_text": tokenizer.decode(generated[0].tolist(), skip_special_tokens=False),
        "eos_token_ids": sorted(stop_ids),
        "step_top5": step_top5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ids_output = args.ids_output
    if ids_output is None and args.input_ids is None:
        ids_output = Path("inputs/qwen_chat_ids.txt")
    if ids_output is not None:
        ids_output.parent.mkdir(parents=True, exist_ok=True)
        ids_output.write_text(" ".join(str(x) for x in input_list) + "\n", encoding="utf-8")
    print(f"golden: {args.output}")
    print(f"input IDs: {ids_output or args.input_ids}")
    print(f"prompt length: {len(input_list)}")
    print(f"generated IDs: {' '.join(str(x) for x in generated_ids)}")
    print(f"generated text: {data['generated_text']!r}")


if __name__ == "__main__":
    main()
