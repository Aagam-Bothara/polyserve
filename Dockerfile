# PolyServe on the official vLLM image, which brings CUDA, PyTorch and vLLM.
# vLLM 0.29 needs an NVIDIA driver 580+; on older drivers build with
#   --build-arg BASE_IMAGE=vllm/vllm-openai:v0.11.0
# CI builds this file on python:3.12-slim to check its own steps without pulling the vLLM image.
#
#   docker build -t polyserve .
#   docker run --gpus all --ipc=host -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.polyserve:/root/.polyserve \
#     polyserve serve Qwen/Qwen2.5-3B-Instruct
#
# The two volumes keep downloaded models and calibrated profiles across restarts, so calibration runs
# once per machine. llama.cpp is not included; mount a llama-server binary and set LLAMA_SERVER to use it.
ARG BASE_IMAGE=vllm/vllm-openai:v0.29.0
FROM ${BASE_IMAGE}

WORKDIR /opt/polyserve
COPY pyproject.toml README.md LICENSE ./
COPY polyserve ./polyserve
RUN pip install --no-cache-dir ".[nvml]"

EXPOSE 8000
# Inside a container, only a server listening on every interface can be reached through -p; outside one,
# PolyServe listens on this machine alone. Publish with -p 127.0.0.1:8000:8000 to keep the container local too.
ENV POLYSERVE_HOST=0.0.0.0
ENTRYPOINT ["polyserve"]
CMD ["--help"]
