# Single-copy KV cache and fused staging

## Scope

The TensorRT decode engine consumes dense `[B,H,S,D]` past tensors and produces
full `[B,H,S+1,D]` present tensors. MiniLLM-RT previously kept two full-capacity
KV slabs per request and issued per-tensor, per-request copies to gather the
batch and rebuild the next slab after every token.

This change separates the persistent request layout from the transient engine
layout:

- one fixed-stride `[tensor,H,capacity,D]` slab is persistent per request;
- one CUDA gather kernel packs every layer/key/value/request and writes mixed-
  length padding as zero in the same pass;
- TensorRT still writes full present tensors into two reusable contiguous
  staging slabs;
- one CUDA scatter kernel copies only the last K/V row back to each request's
  persistent slab. The prefix is never recopied or flipped.

This is an in-place append to the persistent request cache. It is not an alias
of TensorRT's output binding and it is not PagedAttention: the current engine
still computes a dense full-present output and logical scheduler blocks are a
capacity ledger rather than a physical page table.

## Protocol

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM 12.0, normal 55 W laptop policy
- software: CUDA 12.9, TensorRT 10.11.0.33
- model: Qwen2.5-0.5B-Instruct, strongly typed FP16
- workload: four concurrent requests, prompt lengths 8/32/128/256, output 32
- scheduler: continuous batch, active=4, batch cap=4, device argmax
- warmup: one complete generation before each measured run
- order: main/branch, branch/main, main/branch, branch/main, main/branch,
  branch/main
- build: `CMAKE_CUDA_ARCHITECTURES=native`

The main baseline is commit `88c61ea`. Each row below is a fresh process and
reloads the same prefill/decode engines. The paired statistic preserves the
interleaved order instead of selecting fast laptop samples.

## Raw paired makespan

| Pair | Order | Main (ms) | Fused KV (ms) | Improvement |
| ---: | --- | ---: | ---: | ---: |
| 1 | M-F | 461.539 | 288.080 | 37.58% |
| 2 | F-M | 459.786 | 251.649 | 45.27% |
| 3 | M-F | 524.313 | 541.238 | -3.23% |
| 4 | F-M | 489.164 | 257.272 | 47.41% |
| 5 | M-F | 498.848 | 328.957 | 34.06% |
| 6 | F-M | 492.914 | 303.933 | 38.34% |

Main makespan median was 491.039 ms; fused-KV median was 296.007 ms (-39.7%).
The median of the six paired improvements was 37.96%, with 5/6 pairs winning.
The losing pair is retained: GPU enqueue time, not either fused kernel, grew in
that run.

Steady-batch values below first average all batches after the first enqueue in
each process, then take the median across six processes per implementation.

| Metric | Main | Fused KV | Change |
| --- | ---: | ---: | ---: |
| persistent KV bytes, four requests | 13,762,560 | 6,881,280 | -50.0% |
| logical gather bytes / batch | 5,971,968 | 5,971,968 | unchanged |
| scatter bytes / batch | 6,021,120 | 49,152 | -99.18% / 122.5x smaller |
| padding-zero bytes / batch | 13,344,768 | 7,372,800 | -44.76% |
| gather time / batch | 2.683 ms | 0.162 ms | -93.97% |
| scatter time / batch | 4.133 ms | 0.0037 ms | -99.91% |
| backend time / batch | 14.605 ms | 8.104 ms | -44.51% |

The gather byte count is intentionally unchanged because the dense TensorRT
contract still requires every valid past token. The gain comes from replacing
many `cudaMemcpy2DAsync` calls plus a full memset with one vectorized kernel.
The scatter traffic changes fundamentally: 48 layer/key/value tensors × four
requests × one FP16 `[2,64]` row = 49,152 bytes per decode batch.

## Correctness and safety gates

- all 1,536 generated tokens across the 12 performance runs matched main;
- an additional 4-request × 128-token run matched main for all 512 tokens;
- the stored Hugging Face 32-token golden prefix matched for p8/p32/p128/p256;
- the CUDA unit test covers mixed past lengths and capacities, exact gather
  values, zero padding, new-row append, and preservation of prefix/future rows;
- Compute Sanitizer memcheck reported `0 errors` and `0 bytes leaked`;
- CTest passed 6/6 and the real B=1 plus B=4 TensorRT paths both completed.

## Next physical-layout step

The next meaningful optimization is not another host-side block abstraction.
It is a decode attention path that reads a block table directly and emits only
new K/V, removing both dense gather and full-present staging. That requires an
attention plugin/native kernel or a different TensorRT engine contract; these
results do not claim that work is already complete.
