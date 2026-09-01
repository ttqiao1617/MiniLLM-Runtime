ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.26.0
ARG ONNXRUNTIME_SHA256=1254da24fb389cf39dc0ff3451ab48301740ffbfcbaf646849df92f80ee92c57
ARG EXPECTED_TRTLLM_VERSION=1.0.0
ARG EXPECTED_TENSORRT_PREFIX=10.11

# `release:latest` is mutable. Fail early instead of silently building with an
# untested TensorRT-LLM/TensorRT combination. A recorded image digest remains
# the preferred way to reproduce the original container.
RUN python3 -c \
    'from importlib.metadata import version; import sys; actual = version("tensorrt_llm"); expected = sys.argv[1]; assert actual == expected, f"TensorRT-LLM {actual} != expected {expected}; use the recorded BASE_IMAGE digest"' \
    "${EXPECTED_TRTLLM_VERSION}" \
    && python3 -c \
    'from importlib.metadata import version; import sys; actual = version("tensorrt"); expected = sys.argv[1]; assert actual.startswith(expected), f"TensorRT {actual} does not start with {expected}; use the recorded BASE_IMAGE digest"' \
    "${EXPECTED_TENSORRT_PREFIX}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
      -o /tmp/onnxruntime.tgz \
    && echo "${ONNXRUNTIME_SHA256}  /tmp/onnxruntime.tgz" | sha256sum --check --strict \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && ln -s "/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime \
    MINILLM_WORKSPACE=/workspace/trtllm_workspace \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/MiniLLM-RT
COPY requirements.txt /tmp/minillm-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/minillm-requirements.txt

COPY . .
RUN ./scripts/build.sh && ctest --test-dir build --output-on-failure

CMD ["bash"]
