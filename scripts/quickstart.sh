#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

workspace=${MINILLM_WORKSPACE:-/workspace/trtllm_workspace}
model_dir="$workspace/qwen2.5-0.5b-instruct"
artifact_root="$workspace/minillm-rt/qwen2.5-0.5b-instruct"
engine_dir="$artifact_root/engines"
run_id=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-v10"}

check_environment() {
  if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "error: the provided ONNX Runtime SDK requires x86_64" >&2
    exit 1
  fi

  local tool
  for tool in python3 cmake nvidia-smi; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: required command not found: $tool" >&2
      exit 1
    fi
  done

  local trtexec_path
  trtexec_path=$(command -v trtexec || true)
  if [[ -z "$trtexec_path" && -x /usr/local/tensorrt/bin/trtexec ]]; then
    trtexec_path=/usr/local/tensorrt/bin/trtexec
  fi
  if [[ -z "$trtexec_path" ]]; then
    echo "error: trtexec was not found on PATH or under /usr/local/tensorrt" >&2
    exit 1
  fi

  local gpu_memory
  gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
  if [[ ! "$gpu_memory" =~ ^[0-9]+$ ]]; then
    echo "error: failed to read GPU memory with nvidia-smi" >&2
    exit 1
  fi
  if ((gpu_memory < 7000)); then
    echo "warning: only ${gpu_memory} MiB GPU memory is available; Engine build may fail" >&2
  fi

  python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("tensorrt_llm", "tensorrt", "torch", "transformers", "onnx"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")
PY
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "trtexec: $trtexec_path"
  echo "workspace: $workspace"
  echo "environment check passed"
}

download_model() {
  python3 python/download_qwen.py --output "$model_dir"
}

export_graph() {
  local mode=$1
  local path="$artifact_root/fp16_deployment/qwen2_5_0_5b_${mode}.onnx"
  [[ -f "$path" ]] && return
  python3 python/export_qwen_onnx.py \
    --model "$model_dir" --output-root "$artifact_root" \
    --dtype float16 --mode "$mode" --engine-max-sequence-length 512
}

build_engine() {
  local mode=$1
  local path="$engine_dir/qwen2_5_0_5b_${mode}.plan"
  local inspection="$engine_dir/qwen2_5_0_5b_${mode}.engine.json"
  if [[ ! -f "$path" ]]; then
    python3 python/build_qwen_engines.py \
      --onnx-dir "$artifact_root/fp16_deployment" \
      --output-dir "$engine_dir" --mode "$mode"
  fi
  if [[ ! -f "$inspection" ]]; then
    python3 python/build_qwen_engines.py \
      --onnx-dir "$artifact_root/fp16_deployment" \
      --output-dir "$engine_dir" --mode "$mode" \
      --inspect-existing --force
  fi
}

prepare_artifacts() {
  export_graph prefill
  export_graph decode_batch
  build_engine prefill
  build_engine decode_batch
}

prepare_v09_artifacts() {
  export_graph logits
  export_graph prefill
  export_graph decode
  build_engine logits
  build_engine prefill
  build_engine decode
}

build_runtime() {
  ./scripts/build.sh
  ctest --test-dir build --output-on-failure
}

run_smoke_test() {
  ./build/minillm_rt \
    --model "$engine_dir/qwen2_5_0_5b_prefill.plan" \
    --decode-model "$engine_dir/qwen2_5_0_5b_decode_batch.plan" \
    --model-spec configs/qwen2_5_0_5b_fp16.spec \
    --request-inputs inputs/qwen_v09/perf_p8_ids.txt,inputs/qwen_v09/perf_p32_ids.txt,inputs/qwen_v09/perf_p128_ids.txt,inputs/qwen_v09/perf_p256_ids.txt \
    --request-max-new-tokens 16,16,16,16 \
    --mode continuous_batch --provider tensorrt \
    --max-active-requests 4 --max-batch-size 4 \
    --kv-block-size 16 --kv-num-blocks 256 \
    --timeline-csv /tmp/minillm_timeline.csv \
    --request-summary-csv /tmp/minillm_requests.csv \
    --batch-csv /tmp/minillm_batches.csv
}

run_benchmark() {
  python3 python/run_v10_continuous_batching.py \
    --run-id "$run_id" --model-dir "$model_dir" --artifact-root "$artifact_root"
}

run_v09_benchmark() {
  python3 python/run_v09_qwen.py \
    --run-id "$run_id" --model-dir "$model_dir" --artifact-root "$artifact_root"
}

verify_runtime() {
  check_environment
  build_runtime
  run_smoke_test
}

case "${1:-all}" in
  check) check_environment ;;
  model) download_model ;;
  artifacts) prepare_artifacts ;;
  artifacts-v09) prepare_v09_artifacts ;;
  build) build_runtime ;;
  test) ctest --test-dir build --output-on-failure ;;
  smoke) run_smoke_test ;;
  verify) verify_runtime ;;
  benchmark) run_benchmark ;;
  benchmark-v09) run_v09_benchmark ;;
  all)
    check_environment
    download_model
    prepare_artifacts
    build_runtime
    run_smoke_test
    ;;
  *)
    echo "usage: $0 {check|all|model|artifacts|artifacts-v09|build|test|smoke|verify|benchmark|benchmark-v09}" >&2
    exit 2
    ;;
esac
