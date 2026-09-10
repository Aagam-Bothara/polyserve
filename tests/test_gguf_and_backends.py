from __future__ import annotations

import pytest

from polyserve.backends import get_backend, registry
from polyserve.gguf import GGUF_QUANTS, estimate_gguf_bytes, match_gguf_files
from polyserve.models import Config


def test_match_gguf_files_picks_single_file_quants():
    files = [
        "README.md",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q6_K.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_S.gguf",
        "Llama-3.2-3B-Instruct-Q6_K-00001-of-00002.gguf",
        "Llama-3.2-3B-Instruct-f16.gguf",
    ]
    m = match_gguf_files(files, GGUF_QUANTS)
    assert m == {
        "Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Q5_K_M": "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        "Q6_K": "Llama-3.2-3B-Instruct-Q6_K.gguf",
        "Q8_0": "Llama-3.2-3B-Instruct-Q8_0.gguf",
    }


def test_match_gguf_files_case_insensitive_and_nested():
    m = match_gguf_files(["quants/llama-3b.q4_k_m.gguf"], ("Q4_K_M",))
    assert m == {"Q4_K_M": "quants/llama-3b.q4_k_m.gguf"}
    assert match_gguf_files(["model-Q4_K_M.bin"], ("Q4_K_M",)) == {}


def test_estimate_gguf_bytes_orders_by_quant():
    p = 3_210_000_000
    sizes = [estimate_gguf_bytes(p, q) for q in GGUF_QUANTS]
    assert sizes == sorted(sizes)
    assert 1.8e9 < sizes[0] < 2.1e9  # Q4_K_M of a 3B model is ~2.0 GB


def test_registry_has_all_v1_backends():
    assert set(registry()) == {"vllm", "sglang", "llamacpp-cuda", "llamacpp-cpu", "vllm-cpu"}
    with pytest.raises(KeyError):
        get_backend("mlx")


def test_vllm_launch_args(hw_a100, prepared_vllm):
    be = get_backend("vllm")
    cfg = Config(backend="vllm", quant="fp8", ctx=8192, batch=128, gpu_memory_utilization=0.9)
    spec = be.launch_spec(cfg, prepared_vllm, 9000)
    args = spec.args
    assert "vllm.entrypoints.openai.api_server" in args
    for flag, val in (("--port", "9000"), ("--max-model-len", "8192"), ("--max-num-seqs", "128"),
                      ("--gpu-memory-utilization", "0.90"), ("--quantization", "fp8")):
        assert args[args.index(flag) + 1] == val
    assert "--dtype" not in args


def test_sglang_launch_args(hw_a100, prepared_vllm):
    be = get_backend("sglang")
    cfg = Config(backend="sglang", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.88)
    args = be.launch_spec(cfg, prepared_vllm, 9001).args
    assert "sglang.launch_server" in args
    assert args[args.index("--mem-fraction-static") + 1] == "0.88"
    assert args[args.index("--dtype") + 1] == "bfloat16"


@pytest.mark.usefixtures("no_network")
def test_llamacpp_launch_args_split_context_across_slots(hw_gtx1080, prepared_llamacpp):
    be = get_backend("llamacpp-cuda")
    cfg = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29, n_batch=512)
    spec = be.launch_spec(cfg, prepared_llamacpp, 9002)
    args = spec.args
    assert args[0].endswith("llama-server")
    assert args[args.index("-c") + 1] == str(4096 * 4)
    assert args[args.index("-np") + 1] == "4"
    assert args[args.index("-ngl") + 1] == "29"
    assert "-fa" in args


@pytest.mark.usefixtures("no_network")
def test_llamacpp_refuses_unmaterialized_gguf(hw_gtx1080, prepared_llamacpp):
    be = get_backend("llamacpp-cuda")
    prepared_llamacpp.gguf_paths["Q4_K_M"] = "hf://bartowski/x/x-Q4_K_M.gguf"
    cfg = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=2048, batch=1, n_gpu_layers=29)
    with pytest.raises(RuntimeError, match="not materialized"):
        be.launch_spec(cfg, prepared_llamacpp, 9003)


def test_llamacpp_cpu_sets_threads_and_hides_gpu(hw_cpu, prepared_llamacpp, monkeypatch):
    import polyserve.backends.llamacpp as lc

    monkeypatch.setattr(lc, "llama_server_binary", lambda: "/opt/llama-server")
    be = get_backend("llamacpp-cpu")
    cfg = be.default_config(hw_cpu, prepared_llamacpp)
    assert cfg.n_gpu_layers == 0 and cfg.extra["threads"] == 8
    spec = be.launch_spec(cfg, prepared_llamacpp, 9004)
    assert spec.env["CUDA_VISIBLE_DEVICES"] == ""
    assert spec.args[spec.args.index("-t") + 1] == "8"


def test_vllm_cpu_env(hw_cpu_avx512, prepared_vllm):
    be = get_backend("vllm-cpu")
    cfg = Config(backend="vllm-cpu", quant="bf16", ctx=4096, batch=16)
    spec = be.launch_spec(cfg, prepared_vllm, 9005)
    assert spec.env["VLLM_TARGET_DEVICE"] == "cpu"
    assert int(spec.env["VLLM_CPU_KVCACHE_SPACE"]) >= 4


def test_candidate_grids_are_bounded(hw_a100, hw_rtx4090, hw_gtx1080, prepared_vllm, prepared_llamacpp):
    n_vllm = len(get_backend("vllm").candidate_configs(hw_a100, prepared_vllm))
    n_sgl_a100 = len(get_backend("sglang").candidate_configs(hw_a100, prepared_vllm))
    n_sgl_ada = len(get_backend("sglang").candidate_configs(hw_rtx4090, prepared_vllm))
    n_lc = len(get_backend("llamacpp-cuda").candidate_configs(hw_gtx1080, prepared_llamacpp))
    assert n_vllm == 2 * 3 * 3 * 3  # {bf16, fp8} x gmu x ctx x batch
    assert n_sgl_a100 == 1 * 3 * 3 * 3 and n_sgl_ada == 2 * 3 * 3 * 3  # SGLang fp8 needs cc >= 8.9
    assert n_lc == 4 * 3 * 3 * 3  # 4 quants x ngl x ctx x n_parallel
    assert 100 <= n_vllm + n_sgl_ada + n_lc <= 250  # "~144 grid points" ballpark per machine


@pytest.mark.usefixtures("no_network")
def test_prepare_records_weight_sizes(hw_a100, hw_gtx1080, spec):
    p = get_backend("vllm").prepare(spec, hw_a100)
    assert set(p.weights_bytes) == {"bf16", "fp8"} and p.weights_bytes["fp8"] * 2 == p.weights_bytes["bf16"]
    q = get_backend("llamacpp-cuda").prepare(spec, hw_gtx1080)
    assert set(q.weights_bytes) == set(GGUF_QUANTS)
    assert all(v.startswith("convert://") for v in q.gguf_paths.values())  # hub search stubbed to empty
