# Benchmarks

Every number PolyServe claims, and how it was measured. The short version is in the [README](../README.md#results); how calibration works is in [usage.md](usage.md).

> **Correction (September 2026).** Earlier versions of this section reported RTX 3090 and CPU throughput measured with a harness that sent the same prompts at every concurrency level of a trial. The engine's prompt cache then made every later level's prefill nearly free, so those numbers were inflated (one configuration read 3638 tok/s with replayed prompts and 1814 with fresh ones). They have been removed; the git history has them, and the raw files stay in [benchmarks/results/](../benchmarks/results/) for the memory figures below, which prompt caching does not affect. Every throughput number here now comes from the fixed harness.

## Re-measured on an A40

One A40 (48 GB, Ampere, cc 8.6) on RunPod Secure Cloud, with vLLM 0.11.0 plus FlashInfer 0.3.1 and llama.cpp built from source on 12 September 2026. The model is Qwen2.5-3B-Instruct unless stated, the objective is `balanced`, and every concurrency level gets fresh prompts. The multi-GPU rows come from a second pod with two A40s linked only by PCIe. Raw results, ablations and quality data are in [benchmarks/strategies/](../benchmarks/strategies/), and [SUMMARY.md](../benchmarks/strategies/SUMMARY.md) has every table in full (`benchmarks/summarize_strategies.py` regenerates it).

**PolyServe's pick against stock settings**, measured minutes apart on the same card. Stock means `vllm serve <model>`, the same with `--quantization fp8`, and `llama-server -m <Q4_K_M>`:

| workload | PolyServe pick | tok/s | vs stock vLLM | vs stock vLLM fp8 | vs stock llama.cpp |
|---|---|---|---|---|---|
| `generation` | vLLM GPTQ 4-bit, batch 256 | 1157 | **+111%** | +38% | +514% |
| `chat-system` | vLLM GPTQ 4-bit, ctx 16k, batch 256 | 733 | **+69%** | +34% | +372% |
| `chat` | vLLM GPTQ 4-bit, batch 64 | 783 | **+67%** | +36% | +402% |
| `rag-shared` | vLLM AWQ 4-bit, ctx 32k, batch 64 | 480 | **+44%** | +27% | stock fails (4k context) |
| `high-concurrency` | vLLM AWQ 4-bit, batch 256 | 1786 | +6% | +60% | stock misses SLO |
| `rag` | vLLM bf16, ctx 32k | 106 | +1% | +134% | not measured |
| `chat`, **Qwen2.5-7B** | vLLM GPTQ 4-bit, batch 256 | 495 | **+113%** | +144% | +413% |
| `chat-system`, llama.cpp only | Q5_K_M, 8 slots, n-gram speculation | 455 | | | **+190%** |

**Read this before the table: on vLLM the gain is 4-bit weights, and 4-bit is not free.** Where PolyServe picked a 4-bit checkpoint, swapping it for fp8 at the same settings cost 23–29% of the throughput. On GSM8K those checkpoints cost Qwen2.5-3B 4.1–5.0 points of accuracy, where fp8 cost 2.4; on Qwen2.5-7B they cost 0.7–1.1 points, within noise (see the quality tables below). Locked to `--quant bf16,fp8`, PolyServe's gain over stock vLLM falls to +54% on `generation`, +27% on `chat` and +10% on `rag-shared`, and on `high-concurrency` it would serve bf16, matching stock. On Qwen2.5-7B `chat`, fp8 could not keep time to first token under the 500 ms ceiling above concurrency 4. Within that ceiling it measured below bf16, so there the 4-bit pick led fp8 by 144%, and locked to `bf16,fp8` PolyServe would serve bf16 and roughly match stock. Because of that quality cost, `--quant auto` no longer picks 4-bit checkpoints. The runs above allowed them (that is `--quant auto,awq,gptq` today), and the `bf16,fp8` numbers in this paragraph are what the default does now. The stock llama.cpp column is dominated by its one-slot default: most of that gap is concurrency, not tuning.

**What each strategy was worth.** `benchmarks/ablate_strategies.py` flips one strategy of the pick at a time and measures both back to back:

| strategy (vLLM) | `chat` | `chat-system` | `rag-shared` | `high-concurrency` | `generation` |
|---|---|---|---|---|---|
| 4-bit weights, against fp8 at the same settings | +33% | did not fit* | +31% | +33% | +40% |
| prefix caching, on against off | | **+129%** | **+409%** | | |
| fp8 KV cache | −3.7% | −2.1% | +0.5% | −4.5% | −3.8% |
| n-gram speculative decoding | −6.9% | −19.7% | −9.4% | +0.1% | −12.1% |

\* The ablation did not yet step the batch down when fp8 weights did not fit at the pick's batch; it does now.

- **Prefix caching** is the largest single effect: on `rag-shared` it cut time to first token from 1377 to 386 ms. vLLM caches prefixes by default, so it is not a gain over stock; PolyServe keeps it on and now measures it.
- **The fp8 KV cache and vLLM's n-gram speculation** never beat noise on these synthetic prompts (on real text the fp8 cache did; see [Real text, and a card where memory is tight](#real-text-and-a-card-where-memory-is-tight)), and the tie-break (fewer strategies wins inside 2%) kept them out of every pick. On llama.cpp the result was different. Its `ngram-mod` speculation won on `chat-system` (time per token 16.3 → 6.8 ms), and `--kv-unified` measured 12.7% better still. The staged search never tried that combination, because it had tested prefix flags before speculation joined the leader. Tuning one dimension at a time has limits, and this is one of them; the `--combine` stage, added since, measures such pairs.
- **Two GPUs** (`high-concurrency`, two PCIe A40s): two replicas gave 3216 tok/s against 1845 on one GPU (+74%). Tensor parallelism gave 1560, slower than one GPU. `--layout auto` picked replicas. Disaggregated prefill/decode on `rag` lost to one engine (102 against 105 tok/s, time to first token 3.1 s against 1.3 s), and `--phases auto` kept one engine.
- **Qwen2.5-7B `chat`**: the 4-bit pick reached 495 tok/s against 232 for stock vLLM, with time per token 11.9 ms against 31.3. The fp8 KV cache cost 1.5% and n-gram speculation 40%, and neither was picked.
- **Run-to-run variation** was 1–3% for vLLM and up to 10% for llama.cpp. The same llama.cpp pick measured 508, 455 and 427 tok/s across calibration, comparison and ablation.

**Quality**, from `benchmarks/quality_check.py`, perplexity on the same public-domain book at every precision:

| model | bf16 | fp8 | AWQ 4-bit | GPTQ 4-bit |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 5.278 | +1.1% | +25.9% | +31.6% |
| Qwen2.5-7B-Instruct | 2.290 | +1.6% | +32.9% | +36.4% |

llama.cpp's own 4-bit fared better on the same text. Against Q8_0, Q6_K scored +0.9%, Q5_K_M +3.8% and Q4_K_M +10.1% (llama-perplexity, 3B). The book is almost certainly in the models' training data, so these losses partly measure lost memorisation.

**Task accuracy**, from `benchmarks/task_quality.py`: all 1319 GSM8K test problems, zero-shot with the chat template, greedy decoding, graded on the final number (every answer ended in one). Each precision is compared with bf16 problem by problem, with an exact McNemar test on the problems one got right and the other did not:

| model | bf16 | fp8 | AWQ 4-bit | GPTQ 4-bit |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 87.2% | 84.8% (−2.4, p = 0.002) | 82.2% (−5.0, p < 0.001) | 83.1% (−4.1, p < 0.001) |
| Qwen2.5-7B-Instruct | 91.6% | 91.4% (−0.2, p = 0.76) | 90.9% (−0.7, p = 0.35) | 90.5% (−1.1, p = 0.14) |

Each accuracy's 95% interval is about ±2 points. On 3B every quantization cost real accuracy: 2.4 points for fp8, 4–5 for the 4-bit checkpoints. On 7B no difference could be told apart from noise, including the 4-bit checkpoints whose perplexity rose 33–36%. So the book overstated 4-bit's damage at 7B and understated fp8's at 3B. 4-bit stays opt-in because it measurably hurts the 3B model and this is one task on one model family; on this evidence `--quant auto,awq,gptq` is a reasonable choice at 7B. These ran on vLLM 0.11 on the A40, where fp8 is weight-only (Marlin). On Ada fp8 also quantizes activations; on an L4 that cost Qwen2.5-7B nothing either (91.3% against 91.4% at bf16, 31 problems lost and 30 gained, p = 1.0). bf16 on the L4 scored 91.4% against 91.6% on the A40, so a few problems of difference is what changing the GPU alone does. llama.cpp's GGUF formats have not been graded.

## Real text, and a card where memory is tight

Measured on 12–13 September 2026 with vLLM 0.29.0 on the A40 (Qwen2.5-3B-Instruct) and vLLM 0.11.0 on an NVIDIA L4 (24 GB, Ada, cc 8.9, about 300 GB/s; Qwen2.5-7B-Instruct), default `--quant auto`, objective `balanced`. Stock rows are `vllm serve <model>`, the same with `--quantization fp8`, and `llama-server` with one slot.

**Real-text workloads on the A40** (ShareGPT first turns; type hints added to HumanEval functions; sentences with numbers copied out of CNN/DailyMail articles):

| workload | PolyServe pick | tok/s | vs stock vLLM | stock vLLM fp8 | vs stock llama.cpp |
|---|---|---|---|---|---|
| `sharegpt` | vLLM bf16, fp8 KV cache, batch 512 | 1891 | +10.3% | fails to start (vLLM 0.29, Ampere) | +895% |
| `code-edit` | vLLM bf16, fp8 KV cache, batch 256 | 483 | +5.2% | fails to start | +158% |
| `extract` | vLLM bf16, batch 256, Qwen2.5-0.5B draft model | 731 | **+66.5%** | fails to start | +320% |

What each strategy was worth, flipped one at a time and measured back to back:

| strategy (vLLM, A40) | `sharegpt` | `code-edit` | `extract` |
|---|---|---|---|
| fp8 KV cache, on against off | +8.7% | +12.5% | −16.3%* |
| of which FlashInfer attention alone | +0.5% | +5.4% | −3.5%* |
| 4-bit weights (AWQ / GPTQ), not allowed by default | +110% / +109% | +117% / +122% | +19% / +23% |
| n-gram speculative decoding | −87% | −72% (best at one user, where it was +80%) | −68% (calibration) |
| draft-model speculative decoding (Qwen2.5-0.5B) | −27% | −11% | +13.5% (in the pick) |

\* The `extract` column flips the first run's pick, the fp8 KV cache with the draft model (628 tok/s in the comparison, +43% over stock). With the draft model on, the fp8 cache cost 16%: the draft model with an unquantized cache measured 728 tok/s against 609. The search had adopted the fp8 cache first (442 → 526 tok/s on its own) and measured the draft model only on top of it, and the combination stage only ever added changes, so it never tried taking the cache away again. A second run that also allowed single changes, still ranked by summed gains, picked the same configuration (638 tok/s): all eight of its combinations kept the fp8 cache. The stage now first measures the leader with each adopted change undone, and a third run found the configuration on its second undo trial: 726 tok/s in calibration and 731 in the comparison, the row in the table above.

- On Ampere the fp8 KV cache runs as e5m2 through FlashInfer, so turning it on also changes the attention kernel. A variant with the unquantized cache on FlashInfer separates the two: on `code-edit` each is worth about half, on `sharegpt` nearly all of it is the cache.
- 4-bit weights doubled throughput on real text, against +31–40% on the synthetic presets. On Qwen2.5-3B they cost 4–5 GSM8K points, which is why `--quant auto` still leaves them out.
- n-gram speculation has a sharp crossover. On `code-edit`, where the answer repeats much of the prompt, it cut time per token for one user from 12.9 to 7.4 ms; at four users it rose to 69 ms and throughput fell to a quarter. On `sharegpt`, where answers rarely repeat the prompt, it lost everywhere. The search measures at the workload's concurrency and rejected it both times.
- `extract` is where the search earned its keep: the answer copies sentences out of the prompt, so a small draft model guesses it well, and the pick ran 66.5% ahead of stock vLLM, which uses no draft model. The fp8 cache, worth +19% on its own, got in the draft model's way, and only undoing it found that (footnote above).
- The combination stage ran on all three workloads (4, 4 and 8 combinations). None beat the leader by more than the 2% noise band; on `extract` the first version missed a configuration 19% faster, and with the undo step a rerun found it.

**Qwen2.5-7B on an L4 (24 GB).** Weights take 14.2 GB in bf16 and 7.1 GB in fp8, against a 22 GB budget.

| workload | PolyServe pick | tok/s | stock vLLM (bf16) | stock vLLM fp8 | stock llama.cpp |
|---|---|---|---|---|---|
| `chat` | vLLM fp8, ctx 8k, batch 64 | 195 | misses the 50 ms time-per-token ceiling at every concurrency (58–62 ms) | 195 | 49 |
| `sharegpt` | vLLM fp8, fp8 KV cache, prefill budget 16k, ctx 8k, batch 64 | 738 | 410 (PolyServe +80%) | 701 (PolyServe +5.3%) | 24 of 88 requests failed with one slot |

- Against stock settings the pick wins outright: stock vLLM serves bf16 and cannot meet the `chat` ceiling on this card at all, and on `sharegpt` it trails by 80%. Against someone who already passes `--quantization fp8` the pick ties on `chat` (every other knob measured within noise, 192–197 tok/s) and leads by 5.3% on `sharegpt`, where it added the fp8 cache and a larger prefill budget. That 5.3% cost latency: `balanced` maximises throughput under the 1000 ms TTFT ceiling, and the pick used most of it (985 ms against 605 for stock fp8).
- The precision choice depends on the card and the latency target. 4-bit AWQ decoded fastest here (255 tok/s at 8 users, 20–24 ms per token) but its prefill was slower, and time to first token passed the 500 ms ceiling from 4 users up (509 and 857 ms). fp8 prefills on the L4's fp8 tensor cores and stayed inside it. On the A40, where vLLM 0.29 offers no fp8, 4-bit was the fastest option by 2×.
- Flipped one at a time on `sharegpt`, the fp8 cache was worth 7.9% on the L4 too (681 tok/s without it), where it is native e4m3 with no change of attention kernel. n-gram speculation measured −1.2% here on vLLM 0.11, against −87% on the A40 on vLLM 0.29 with the e5m2 cache on FlashInfer, so that collapse belongs to that setup rather than to n-gram lookup as such. 4-bit reached 1026 tok/s at 32 users but 1.6 s to first token, so under the 1000 ms ceiling it qualified only at 8 users (373 tok/s).
- On GSM8K, fp8 cost Qwen2.5-7B nothing on the L4 either, where it also quantizes activations: 91.3% against 91.4% (31 problems lost, 30 gained).
- Memory bound less than expected. Qwen2.5-7B's grouped-query attention keeps its KV cache small (57 KB per token), so even bf16 had room for the concurrency these workloads reach; bandwidth was the binding limit, and fp8 halves the bytes per step.

**CPU only** (Qwen2.5-0.5B-Instruct, `default` workload, llama.cpp on the A40 pod's Xeon Gold 6342 with the GPU hidden). The container sees 96 CPUs but may use 7.65, and PolyServe first gave llama.cpp 48 threads: 27 tok/s and 24 s to first token. The CPU probe now honours the cgroup quota and the affinity mask; with 7 threads the same configuration ran at 114 tok/s and 1.7 s. That still broke the 500 ms TTFT ceiling, so the pick was one slot with n-gram speculation: 92.2 tok/s against 67.5 for stock `llama-server` (+36.5%), both at about 475 ms. The synthetic prompts repeat themselves, which flatters n-gram speculation, so treat that gain as an upper bound.

## Your own prompts, SGLang, and a time budget

One A40 on 13–14 September 2026: vLLM 0.29.0 and SGLang 0.5.19 (in its own environment, found through `SGLANG_PYTHON`), Qwen2.5-3B-Instruct, objective `balanced`. The prompts were 300 drawn from Dolly-15k (CC BY-SA 3.0), a dataset no preset uses, passed with `--workload-file` and shaped by the `chat` template (1, 4 and 8 concurrent users). They are not committed; `benchmarks/make_dolly_prompts.py` rebuilds the same file. Every row was measured 3 times, interleaved.

**SGLang, first launch through PolyServe** (smoke test, one short trial each): bf16 230.7 tok/s, AWQ 444.4, GPTQ 443.6, fp8_e5m2 KV cache 224.4, radix cache off 218.3 (time to first token 90 against 63 ms, since the smoke prompts share a prefix). Every flag PolyServe passes was accepted.

**PolyServe against stock settings on your own prompts:**

| row | median tok/s | runs | spread | TTFT |
|---|---|---|---|---|
| **PolyServe:** vLLM bf16, fp8 KV cache, Qwen2.5-0.5B draft model, batch 64 | **547** | 566 / 536 / 547 | 5.6% | 120 ms |
| stock vLLM 0.29 | 496 (PolyServe +10.3%) | 503 / 496 / 478 | 4.9% | 54 ms |
| stock SGLang 0.5.19 | 473 (PolyServe +15.6%) | 473 / 475 / 473 | 0.3% | 54 ms |

- The gain is outside run-to-run noise: PolyServe's slowest run beat both stock rows' fastest.
- SGLang led the first trial (490 against 471 tok/s for vLLM, with lower time to first token and 10% less energy per token). vLLM took the lead with the fp8 cache (521), then a 0.5B draft model (581). Undoing the batch step-up with the draft model on measured 603, the best trial of the search, and became the pick: the combination stage's undo step at work on a fresh workload. n-gram speculation collapsed to 141 tok/s, as on this card before.
- The pick scored 603 tok/s in calibration and 547 in the comparison: choosing the best of many noisy trials flatters it, which is why the comparison re-measures with repeats.
- The draft model roughly doubled time to first token (120 against 54 ms), inside the `chat` preset's 500 ms ceiling.
- Later stages vary only the current leader, so SGLang was never measured with the fp8 cache or a draft model once vLLM led. Whether it would have won with them is unknown.
- Flipped one at a time (single runs, back to back): without the draft model 509 tok/s against the pick's 571, so the draft model is worth about 12%, nearly all of the gain over stock vLLM. Without the fp8 cache 565 (−1%, within noise once the draft model is on, though it was worth 4% before the draft model joined); with the unquantized cache but FlashInfer attention kept 541. With 4-bit weights, which `--quant auto` leaves out for quality, GPTQ reached 712 (+25%) and AWQ 625 (+9%).

**SGLang alone on `sharegpt`** (`--backend sglang`, one run per row): PolyServe tuned SGLang to bf16 with the fp8_e5m2 KV cache at batch 64, 1952 tok/s against 1799 for stock SGLang (+8.5%, time to first token 364 against 318 ms). As on vLLM, the fp8 cache was the step that paid (1790 → 1948 tok/s in calibration); batch 16 left requests queueing (5.9 s to first token). SGLang's backend offers no speculative decoding yet, so none was tried. For scale only, since it came from another session: stock vLLM measured 1714 on the same workload and card.

**The same with `--budget 10m`:** calibration stopped after 6 trials in 580 s (precision, memory and batch for each engine) and skipped 7, the stages where the full run found its gain. The pick was plain vLLM bf16 at batch 64: 504 tok/s (504 / 505 / 503) against stock vLLM's 506 (518 / 503 / 506), which the comparison flagged as within run-to-run noise, and 474 for stock SGLang.

## Memory planner accuracy

Each trial compares the planner's memory estimate with actual allocations reported by NVML and the backend's startup log. Run `polyserve memory-report` to see the comparison, or add `--apply` to fit the planner's constants to your machine.

Measured on the A40 runs above, 146 trials with a memory reading across both pods:

| backend | scored on | trials | mean abs error | bias | worst under-prediction | worst over-prediction |
|---|---|---|---|---|---|---|
| vLLM | weights + workspace | 101 | **16.6%** | +10.9% | −26.5% | +104.8% |
| llama.cpp (CUDA) | peak device memory | 45 | **14.1%** | +13.3% | −18.7% | +23.1% |

The planner under-predicted vLLM's non-KV memory by up to 26.5%: the workspace fitted from these trials is 2.4 GB against its 1.5 GB constant. `polyserve memory-report --apply` therefore raises the safety margin from 5% to 15%.

What the planner is really judged on is whether a configuration it admits then runs out of memory, and here it did three times. vLLM at batch 512 with the fp8 KV cache failed twice, because its sampler warm-up needs memory the planner does not model. Both failures were on vLLM 0.11; on vLLM 0.29 the same shape started on every real-text workload, so calibration now retries an out-of-memory start-up with less memory reserved rather than assuming a buffer size. A two-GPU tensor-parallel engine at 95% memory utilisation failed on NCCL's buffers; tensor-parallel shapes are now capped at 90%. The report counted none of these, because it read only the last lines of the engine log. Start-up failures now carry the engine's out-of-memory message, so the next run counts them.

The two backends are scored on different quantities on purpose. vLLM and SGLang size their KV pool to fill `gpu_memory_utilization × VRAM`, so their peak memory is a policy choice, not a requirement; scoring against it compares two different things. On the A40 vLLM's KV pool was **4.0× larger** than the planner budgeted, which is why the conservative `PAGED_KV_FRACTION` did not reject workable configurations. llama.cpp allocates exactly what it is asked for, so it is scored on peak memory; its error is mostly over-prediction, from a hand-set 768 MB workspace constant against the 349 MB `polyserve memory-report --apply` fitted from the A40 trials.

## Performance predictor

The predictor uses a roofline model with three parameters per backend: bandwidth efficiency α, compute efficiency β and a per-step overhead. `polyserve fit` learns them from one machine's own trials and checks itself by leaving one trial out at a time. On the A40 runs:

| machine | backend | trials | throughput error, fitted | throughput error, defaults | rank correlation (Spearman ρ) |
|---|---|---|---|---|---|
| one A40 | llama.cpp (CUDA) | 117 | **22.9%** | 36.3% | **0.89** |
| one A40 | vLLM | 219 | 28.5% | **22.8%** | **0.88** |
| two A40s | vLLM | 168 | 35.9% | **11.5%** | **0.94** |

Fitting helped llama.cpp and hurt vLLM. vLLM's trials mix 3B and 7B models, 4-bit weights, prefix-cached prompts and prefill-bound workloads, which three parameters cannot all follow, and the fit chases time-to-first-token errors of 65–121% at the expense of throughput. `polyserve fit --apply` therefore keeps a backend's defaults whenever they predict better than the fit. The search needs the ranking more than the exact throughput, and a rank correlation of 0.88–0.94 is what lets it skip quantizations that cannot win. Time to first token is not predicted usefully: scheduling, queueing and prefix caching dominate it, and the model represents none of them.

## Not yet measured

These are gaps, not claims. In rough order of how much they would change the conclusions:

1. **Task quality beyond one benchmark.** On GSM8K, quantization cost Qwen2.5-3B 2–5 points and Qwen2.5-7B about one, within noise, and fp8 with activation quantization on Ada cost 7B nothing. Other tasks (code, long-context retrieval), other model families, llama.cpp's GGUF formats and Hopper are ungraded, so whether 4-bit checkpoints could return to `--quant auto` for larger models is still open.
2. **A model where memory truly binds.** Qwen2.5-7B on a 24 GB L4 was bandwidth-bound, not memory-bound: its grouped-query KV cache is small, and the presets reach at most 32 concurrent requests. A larger model on the same card (a 14B in fp8, or a model without grouped-query attention), or long-context traffic at high concurrency, is where batch and cache sizing would decide the result.
3. **Other accelerators.** A100, A30 and a pre-Turing card (GTX 1080) are untested, so the compute-capability branch in the selector has never run on real hardware. SGLang has run on one A40 only (0.5.19, see [Your own prompts, SGLang, and a time budget](#your-own-prompts-sglang-and-a-time-budget)), on one workload, and never with speculative decoding, which its backend does not offer yet.
4. **Beating an expert, not just the defaults.** Where the pick won big, it chose what an informed user could also pass by hand: fp8 on the L4, a draft model and an fp8 cache on `extract`. Measured against someone who already knows those flags, the rest of the search moved throughput by a few percent. Its value is knowing which of them pay on this card and this traffic, and what they cost in quality. The search can also miss: on `extract` the ablation found a configuration 19% faster than the pick, and only then did the combination stage learn to undo adopted changes (a rerun found it). What else a staged search misses is known only as far as the ablations reach.
5. **Energy tuning on real hardware.** `--power` has only run against a simulated NVML. It needs root on the host, so it has to be measured on a machine you control; `benchmarks/measure_power.sh` runs the whole measurement in one command and restores the GPU afterwards. Whether the energy-optimal point sits near 70% of full power for these workloads, and whether it differs between prefill-heavy `rag` and decode-heavy `generation`, is still a prediction.
6. **Disaggregated prefill and decode at a scale where it could pay.** On two PCIe-linked A40s with a 3B model it ran end to end but lost to one engine (102 against 105 tok/s, time to first token 3.1 s against 1.3 s), and one of the two pairs tried failed a quarter of its requests with KV blocks the decode engine never pulled. Published gains come from larger models, NVLink or RDMA between the engines, and heavier prefill contention, none of which was available here.
7. **Speculative decoding by load.** On real text on the A40 (vLLM 0.29, fp8 cache on FlashInfer) n-gram speculation made one user up to 80% faster and collapsed from four users up; on the L4 (vLLM 0.11, native fp8 cache) it neither helped nor collapsed on `sharegpt`. Which part of the A40 setup causes the collapse is unmeasured. A draft model paid on `extract`. A server that switches speculation on only at low load would get the single-user gain without the collapse; PolyServe picks one setting per workload. llama.cpp's `ngram-mod` has not been measured on real text.
8. **Memory outside vLLM's reservation.** At batch 512 with the fp8 cache and 95% memory utilization, vLLM 0.11 ran out of memory at start-up twice; vLLM 0.29 started that shape every time it was tried. The planner does not model that memory, and the evidence gives no single size for it, so calibration now retries such a failure with 5 points less memory reserved (down to 0.85) instead. A planner term would need start-up logs from several engine versions.
