from __future__ import annotations

import pytest

from polyserve import memcal
from polyserve.memlog import MeasuredMemory, merge, parse_llamacpp, parse_log, parse_sglang, parse_vllm
from polyserve.memory import DEFAULT_MARGIN_FRACTION, MemoryModel, estimate, safety_margin
from polyserve.models import Config, GiB, MemoryEstimate, MemoryObservation, MiB, TrialMetrics, TrialResult

VLLM_LOG = """
INFO 09-10 18:14:01 [model_runner.py] Model loading took 6.4265 GiB memory and 8.16 seconds
INFO 09-10 18:14:05 [worker.py] Memory profiling takes 3.12 seconds. Total non KV cache memory: 7.85GiB; torch peak memory increase: 1.02GiB; non-torch forward increase memory: 0.41GiB; weights memory: 6.43GiB.
INFO 09-10 18:14:05 [worker.py] Available KV cache memory: 11.34 GiB
INFO 09-10 18:14:06 [kv_cache_utils.py] GPU KV cache size: 353,296 tokens
INFO 09-10 18:14:06 [kv_cache_utils.py] Maximum concurrency for 4,096 tokens per request: 86.25x
"""

LLAMACPP_LOG = """
load_tensors: offloading 36 repeating layers to GPU
load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
load_tensors:        CUDA0 model buffer size =  2003.50 MiB
llama_context: n_ctx = 16384
llama_kv_cache_unified:      CUDA0 KV buffer size =   576.00 MiB
llama_context:      CUDA0 compute buffer size =   300.25 MiB
llama_context:  CUDA_Host compute buffer size =    16.01 MiB
"""

SGLANG_LOG = """
[2026-09-10] Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.20 GB, mem usage=6.21 GB.
[2026-09-10] KV Cache is allocated. #tokens: 123,456, K size: 2.12 GB, V size: 2.12 GB
[2026-09-10] Memory pool end. avail mem=8.10 GB
"""


def test_parse_vllm_components():
    m = parse_vllm(VLLM_LOG)
    assert m.source == "vllm-log"
    assert m.weights_mb == pytest.approx(6.43 * 1024)
    assert m.non_kv_mb == pytest.approx(7.85 * 1024)
    assert m.workspace_mb == pytest.approx((7.85 - 6.43) * 1024)
    assert m.kv_mb == pytest.approx(11.34 * 1024) and m.kv_tokens == 353_296
    assert m.details["torch_peak_increase_mb"] == pytest.approx(1.02 * 1024)


def test_parse_llamacpp_splits_device_and_host():
    m = parse_llamacpp(LLAMACPP_LOG)
    assert m.source == "llamacpp-log"
    assert m.weights_mb == pytest.approx(2003.5) and m.kv_mb == pytest.approx(576.0)
    assert m.workspace_mb == pytest.approx(300.25) and m.kv_tokens == 16384
    assert m.details["host_weights_mb"] == pytest.approx(308.23)
    assert m.details["host_compute_mb"] == pytest.approx(16.01)


def test_parse_llamacpp_cpu_only():
    text = "load_tensors:   CPU model buffer size =  2311.73 MiB\nllama_kv_cache_unified:        CPU KV buffer size =   256.00 MiB\nllama_context:        CPU compute buffer size =   120.00 MiB\n"
    m = parse_llamacpp(text)
    assert m.weights_mb == pytest.approx(2311.73) and m.kv_mb == 256.0 and m.workspace_mb == 120.0


def test_parse_sglang_and_unknown():
    m = parse_sglang(SGLANG_LOG)
    assert m.source == "sglang-log" and m.weights_mb == pytest.approx(6.21 * 1024)
    assert m.kv_mb == pytest.approx(4.24 * 1024) and m.kv_tokens == 123_456
    assert parse_log("mlx", "anything").source == "none"
    assert parse_vllm("nothing here").source == "none"


def test_merge_subtracts_baseline_and_labels_source():
    m = merge(MeasuredMemory(), device_peak_mb=20_000.0, baseline_mb=350.0, telemetry_source="llmtrace")
    assert m.device_peak_mb == 19_650.0 and m.source == "nvml" and m.total_mb == 19_650.0
    log = parse_llamacpp(LLAMACPP_LOG)
    m2 = merge(log, device_peak_mb=None, baseline_mb=None, telemetry_source="none")
    assert m2.source == "llamacpp-log" and m2.total_mb == pytest.approx(2003.5 + 576.0 + 300.25)


def _trial(backend, key_batch, predicted_mb, measured_mb, measured_ws=None, error=None, launched=True):
    cfg = Config(backend=backend, quant="Q4_K_M", ctx=4096, batch=key_batch, n_gpu_layers=37)
    ws = 768 * MiB
    weights = int((predicted_mb * MiB) * 0.6)
    kv = int(predicted_mb * MiB) - weights - ws
    pred = MemoryEstimate(config_key=cfg.key(), weights=weights, kv_cache=kv, runtime_workspace=ws,
                          safety_margin=512 * MiB, total=int(predicted_mb * MiB) + 512 * MiB, budget=20 * GiB,
                          feasible=True)
    meas = MeasuredMemory(device_peak_mb=measured_mb, workspace_mb=measured_ws, source="nvml") if measured_mb else None
    return TrialResult(config=cfg, stage="t", metrics=TrialMetrics(requests=1, output_tokens=1, failed=0 if launched else 1),
                       launched=launched, error=error, memory=MemoryObservation(predicted=pred, measured=meas))


def test_analysis_reports_error_bias_and_recommendations():
    trials = [
        _trial("llamacpp-cuda", 1, 3000, 2900, measured_ws=280),   # over by 3.4%
        _trial("llamacpp-cuda", 4, 3600, 3700, measured_ws=310),   # under by 2.7%
        _trial("llamacpp-cuda", 8, 4400, 4200, measured_ws=330),   # over by 4.8%
        _trial("llamacpp-cuda", 8, 9000, None, error="CUDA error: out of memory", launched=False),
        _trial("vllm", 64, 20_000, 19_900, measured_ws=1400),
    ]
    obs = memcal.observations_from_trials(trials, "hw1")
    assert len(obs) == 5 and sum(o.oom for o in obs) == 1
    cals = memcal.analyse(obs)
    lc = cals["llamacpp-cuda"]
    assert lc.n == 4 and lc.n_measured == 3 and lc.ooms == 1
    assert lc.mape_pct == pytest.approx((3.45 + 2.70 + 4.76) / 3, abs=0.1)
    assert lc.worst_under_pct == pytest.approx(-2.70, abs=0.05)
    assert lc.fitted_workspace_mb == 330  # p95 of [280, 310, 330]
    assert lc.current_workspace_mb == 768
    assert lc.recommended_margin_fraction == pytest.approx(0.027 + 0.02, abs=0.001)
    v = cals["vllm"]
    assert v.worst_under_pct == 0.0 and v.recommended_margin_fraction == memcal.MIN_MARGIN_FRACTION
    md = memcal.render_markdown(cals, hardware="RTX 3090")
    assert "| llamacpp-cuda | 4 | 3 |" in md and "768 → 330 MB" in md and "1 OOM" in md
    assert "No trials" in memcal.render_markdown({})


def test_overrides_roundtrip_and_planner_uses_them(tmp_home, hw_a100, prepared_llamacpp):
    from polyserve.backends import get_backend

    be = get_backend("llamacpp-cuda")
    assert not be.memory_model(hw_a100).calibrated
    assert be.memory_model(hw_a100).runtime_workspace == be.runtime_workspace_bytes
    from polyserve.hardware import hardware_hash

    hh = hardware_hash(hw_a100)
    cal = memcal.BackendCalibration(backend="llamacpp-cuda", n=3, n_measured=3, fitted_workspace_mb=330.0,
                                    recommended_margin_fraction=0.047)
    path = memcal.save_overrides(hh, {"llamacpp-cuda": cal, "vllm": memcal.BackendCalibration(backend="vllm")})
    assert path == tmp_home / "memory-model.json"
    assert memcal.workspace_override(hh, "llamacpp-cuda") == 330 * MiB
    assert memcal.margin_override(hh, "llamacpp-cuda") == 0.047
    assert memcal.workspace_override(hh, "vllm") is None  # nothing fitted -> not written
    assert memcal.load_overrides("other-machine") == {}
    mm = be.memory_model(hw_a100)
    assert mm.calibrated and mm.runtime_workspace == 330 * MiB and mm.margin_fraction == 0.047
    cfg = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29)
    est = estimate(hw_a100, prepared_llamacpp, cfg, mm)
    assert est.runtime_workspace == 330 * MiB
    assert est.safety_margin == safety_margin(hw_a100.gpu.vram_free_bytes, 0.047)
    assert safety_margin(4 * GiB, 0.02) == 512 * MiB  # floor still applies
    assert MemoryModel(1, lambda c: 0).margin_fraction == DEFAULT_MARGIN_FRACTION


def test_runner_records_observation(tmp_path):
    """The real subprocess runner attaches prediction + measurement to every trial."""
    from tests.test_supervisor import FakeBackend, _cfg
    from polyserve.calibrate.search import SubprocessTrialRunner
    from polyserve.calibrate.workload import Workload
    from polyserve.models import ArchInfo, ModelSpec, PreparedModel
    from tests.conftest import make_hw

    hw = make_hw("cpu")
    arch = ArchInfo(num_layers=2, hidden_size=64, num_attention_heads=2, num_kv_heads=2, head_dim=32, num_params=1000)
    pm = PreparedModel(spec=ModelSpec(hf_id="x/y"), backend="fake", arch=arch, weights_bytes={"none": 1000})
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    runner = SubprocessTrialRunner(backends={"fake": FakeBackend()}, models={"fake": pm}, hw=hw, workload=wl,
                                   log_dir=tmp_path, startup_timeout=15)
    res = runner.run(_cfg(), "quant")
    assert res.ok and res.memory is not None
    assert res.memory.predicted is not None and res.memory.predicted.weights == 1000
    assert res.memory.measured is not None
    assert res.memory.measured.source in ("psutil", "rapl", "none")
