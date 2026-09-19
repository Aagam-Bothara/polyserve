# Security

PolyServe is alpha software (0.1.x), maintained by one person. This page says how to report a vulnerability and, more usefully, what PolyServe does and does not protect, so you can decide how to deploy it.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub instead: the repository's **Security** tab, then **Report a vulnerability**. Include the version (`pip show polyserve`), the command you ran, and what someone could do with it.

There is no guaranteed response time; reports are read and answered as soon as possible. Fixes go into the latest release only.

## What PolyServe protects, and what it does not

**There is no authentication.** Nothing checks who is calling: not the OpenAI-compatible `/v1/*` routes, not `/polyserve/profile`, `/polyserve/drift` or `/health`. There are no rate limits and no request-size limits either.

**It listens on this machine only, unless told otherwise.** `polyserve serve` binds `127.0.0.1`. `--host 0.0.0.0`, or `POLYSERVE_HOST`, exposes it to every network the machine can reach, and it warns when it starts that way. Before doing that, put a reverse proxy in front of it that authenticates and rate-limits. The Docker image sets `POLYSERVE_HOST=0.0.0.0`, because inside a container that is the only address `-p` can reach; publish with `-p 127.0.0.1:8000:8000` to keep a container local as well.

**`/polyserve/profile` describes the machine.** It returns the hardware (GPU and CPU models, memory, driver versions), the model, the engine's launch arguments, which can include local file paths, and the calibration results. It does not return secrets from the environment: the launch environment it stores holds only the variables PolyServe sets itself, such as `CUDA_VISIBLE_DEVICES` and `NCCL_P2P_DISABLE`, never the process environment, so tokens such as `HF_TOKEN` stay out of it.

**The proxy changes streaming requests.** It adds `stream_options: {"include_usage": true}` to a streaming request that did not set `include_usage` itself, so the drift report can count tokens. Everything else passes through unchanged.

**Traffic monitoring never reads text.** The drift report records token counts, concurrency and time to first token. It never records prompts or answers.

**Engines run locally, as you.** vLLM, SGLang and llama.cpp run as child processes listening on `127.0.0.1`, with the permissions of the user who started PolyServe; their own vulnerabilities belong upstream. PolyServe never passes `--trust-remote-code`, so a model repository cannot run code through it.

**Some weights can come from someone other than the model's author.** The model you name, and any draft model speculative decoding adds, come from that model's own organisation on the Hub. Two kinds of file can come from a third party: pre-quantized 4-bit and 8-bit checkpoints, which are off unless you allow them (`--quant auto,awq,gptq,w8a8`), and llama.cpp's GGUF files, which it may download from known quantizers such as bartowski or TheBloke. PolyServe checks what those files are, from a checkpoint's quantization config, not who published them. They are safetensors or GGUF, which hold data rather than code, so the risk is altered answers rather than code execution. With `--backend vllm` or `--backend sglang` and the default `--quant auto`, PolyServe downloads only from the named model's own organisation.

**Power control acts on the whole machine.** `--power` and `polyserve power` set GPU power caps and clock locks through the driver, which needs root, and they apply to every process using that GPU. A crash can leave them set; `polyserve power reset` undoes that.
