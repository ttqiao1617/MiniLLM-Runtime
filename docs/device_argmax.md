# Device argmax: RTX 5060 Laptop A/B

## Why this optimization exists

Qwen2.5-0.5B has a vocabulary of 151,936 tokens. The original continuous-batch
path copied `[B, 151936]` FP16 logits to the host, converted every value to
FP32, and then ran four serial CPU argmax loops. At B=4 that is 1,215,488 bytes
per decode step even though the scheduler only needs four token IDs.

`device_argmax.cu` runs one deterministic CUDA reduction per batch row and
copies four `int32` results. Equal logits choose the lowest token ID, matching
the host `GreedySampler` contract.

## Protocol

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, SM 12.0, 8,151 MiB
- software: CUDA 12.9, TensorRT 10.11.0.33
- model: Qwen2.5-0.5B-Instruct, strongly typed FP16
- workload: four concurrent requests, prompt lengths 8/32/128/256, output 32
- scheduler: continuous batch, active=4, batch cap=4
- warmup: one complete 32-token generation before each measured run
- order: H-D, D-H, H-D, D-H, H-D, D-H
- build: `CMAKE_CUDA_ARCHITECTURES=native`

The laptop was left at its normal power policy. This deliberately reports the
observed variability instead of selecting only fast samples.

## Raw paired makespan

| Pair | Order | Host (ms) | Device (ms) | Improvement |
| ---: | --- | ---: | ---: | ---: |
| 1 | H-D | 864.499 | 664.252 | 23.16% |
| 2 | D-H | 655.645 | 617.635 | 5.80% |
| 3 | H-D | 875.548 | 828.952 | 5.32% |
| 4 | D-H | 716.836 | 463.844 | 35.29% |
| 5 | H-D | 632.035 | 777.867 | -23.07% |
| 6 | D-H | 882.961 | 402.817 | 54.38% |

Host makespan median was 790.668 ms; device median was 640.944 ms (-18.9%).
The median of the six paired improvements was 14.48%, with 5/6 pairs winning.
Across steady decode batches, device argmax averaged 0.251 ms.

The invariant result is the data movement: 1,215,488 bytes becomes 16 bytes per
B=4 decode batch, a 75,968x reduction. This also removes host FP16-to-FP32
conversion and four CPU vocabulary scans.

## Correctness gates

- all 1,536 generated tokens in the 12 performance runs matched the first host
  reference sequence;
- an additional 4-request × 128-token run matched host and device for all 512
  tokens;
- both modes matched the stored Hugging Face golden for p8, p128, and p256;
- for p32, both host and device first diverged from the HF golden at generated
  token 89 while remaining identical to each other. This is an existing
  TensorRT batched-FP16 numerical boundary, not a device reduction regression.

The project therefore keeps `--sampling host` as a reference path and treats
host/device token equality as the device-kernel gate. HF comparisons remain a
separate end-to-end engine gate.
