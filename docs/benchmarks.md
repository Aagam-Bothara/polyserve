# Benchmarks

Every number PolyServe claims, and how it was measured. The [README](../README.md#results) has three headlines, [At a glance](#at-a-glance) has the main results in one table, and [Limits](#limits) says what they do not show; how calibration works is in [usage.md](usage.md).

> **Correction (September 2026).** Earlier versions of this section reported RTX 3090 and CPU throughput measured with a harness that sent the same prompts at every concurrency level of a trial. The engine's prompt cache then made every later level's prefill nearly free, so those numbers were inflated (one configuration read 3638 tok/s with replayed prompts and 1814 with fresh ones). They have been removed; the git history has them, and the raw files stay in [benchmarks/results/](../benchmarks/results/) for the memory figures below, which prompt caching does not affect. Every throughput number here now comes from the fixed harness.

## At a glance

Qwen2.5 and Llama 3.1 models on rented GPUs, objective `balanced`, PolyServe's default `--quant auto` (no 4-bit weights), vLLM 0.29 on the A40, A100, H100 and RTX 4090 and vLLM 0.11 on the L4. The latency ceiling is judged on the 95th-percentile time to first token, PolyServe's default. The Llama rows, your own prompts and the L4 on `sharegpt` were calibrated that way on 14–15 September, and every row there was measured 3 times, interleaved (twice on the H100). The other rows are single runs judged by the median and **re-scored** from their recorded measurements with `benchmarks/rescore_ttft.py`, not re-measured; where that changed a number, the median's follows in brackets, and a calibration judged at p95 could choose differently. Full tables, per-strategy ablations and methods are in the sections below.

| Machine, model | Workload | PolyServe pick | vs stock `vllm serve` | vs stock with `--quantization fp8` |
|---|---|---|---|---|
| A40, Llama 3.1 8B | **prompts calibration never saw** (calibrated on 300 Dolly prompts, measured on 300 others) | vLLM bf16 + 1B draft model | **+93%**, and +97% over stock SGLang (505 against 262 and 257 tok/s); +67% on OpenAssistant prompts | fp8 weights fail on Ampere |
| A100 80 GB, Llama 3.1 8B | the same | vLLM bf16 + 1B draft model | **+59%** (1153 against 724); +47% on OpenAssistant prompts | fp8 weights fail on Ampere |
| H100 NVL, Llama 3.1 8B | the same | fp8 + 1B draft model, 8k prefill budget | +87% (2164 against 1155) | **+24%** (against 1743); +8% on OpenAssistant prompts |
| RTX 4090 24 GB, Llama 3.1 8B | the same | SGLang fp8 | stock fails to start (its 131k context does not fit); +71% over stock SGLang (765 against 448) | fails to start |
| A40, 3B | `extract` (copy facts out of news articles) | bf16 + 0.5B draft model | **+66.5%** | stock fp8 fails to start (vLLM 0.29 on Ampere) |
| A40, 3B | `sharegpt` (real chat first turns) | bf16 + fp8 KV cache, batch 512 | +10.3% | fails to start |
| A40, 3B | `code-edit` (add type hints to functions) | bf16 + fp8 KV cache | +5.2%† | fails to start |
| A40, 3B | **your own prompts** (300 from Dolly-15k, `--workload-file`), vLLM and SGLang both candidates | vLLM bf16 + fp8 KV cache + 0.5B draft model, 16k prefill budget | **+22.8%**, and +30.3% over stock SGLang (619 against 504 and 475 tok/s, p95 157 ms); ranges do not overlap | fails to start |
| A40, 3B | the same prompts, recalibrated with suffix decoding installed; measured on 300 others | vLLM bf16 + fp8 KV cache + suffix decoding, 8k prefill budget | **+122%**, and +110% over stock SGLang (994 against 448 and 472 tok/s) | fails to start |
| L4 24 GB, 7B | `chat` | fp8 weights | stock misses the 50 ms/token target at any load | tie, 107 tok/s at 4 users (195 at 8) |
| L4 24 GB, 7B | `sharegpt` | fp8 weights, 8k prefill budget | **+71%**, 219 against 128 tok/s at 8 users (by the median: +80%, 738 at 32) | tie, 219 against 212 |
| CPU container (7.65 cores), 0.5B | `default` | llama.cpp, 1 slot + n-gram speculation | misses the 500 ms ceiling by 18 ms at p95, which stock `llama-server` meets (+36.5% by the median*) | |

\* Synthetic prompts, which flatter n-gram speculation; treat it as an upper bound.

† A single run, inside the run-to-run spread that repeated runs showed on real text (up to about 6%), so not yet a reliable gain; `compare --repeats 3` would settle it.

- **The right precision depends on the card and the latency target.** On the L4, fp8 beat 4-bit because 4-bit's slower prefill broke the first-token target; on the A40, where vLLM 0.29 offers no fp8, 4-bit ran at twice bf16.
- **Quality was measured, in a separate experiment.** `benchmarks/task_quality.py` graded all 1,319 GSM8K problems at each precision, paired against bf16: on Qwen2.5-3B fp8 cost 2.4 points and 4-bit 4–5 (all significant); on 7B none cost a measurable amount. That is why `--quant auto` leaves 4-bit out; `--quant auto,awq,gptq` puts it back (it doubled throughput on real text). PolyServe does not grade answers while it calibrates, so `--quant` is where you decide what it may trade. An 8-bit W8A8 checkpoint cost Qwen2.5-3B 1.3 points, not significant, and ran 38–48% faster than bf16 on an A40, where vLLM 0.29 has no fp8 weights; it is opt-in (`--quant auto,w8a8`) until a second model agrees.
- **Judged by the tail, a card promises less.** By the median, 8 of 20 recorded picks sent more than 1 request in 20 past the time-to-first-token ceiling; at p95 they serve fewer users at once (the L4 on `sharegpt`: 738 tok/s at 32 users by the median, 219 at 8 calibrated at p95). Stock settings queue worse at the tail as often as not, so the lead over stock grew in some rows (on an A40 on `high-concurrency`, +6% became +20%) and vanished in others (stock fp8 on the L4). [The re-scored tables](#judged-at-the-95th-percentile).
- **Some strategies only pay together, and some get in each other's way.** On `extract` the fp8 cache helped on its own but slowed the draft model. The search finds that by undoing each change it adopted.
- **Speculative decoding depends on load, content and the proposer.** On `extract` on an A40, suffix decoding ran 2.6× plain bf16 at 8 users and 1.6× at 32, and vLLM's n-gram lookup, which collapsed from four users up on its CPU proposer, ran 1.7× bf16 at 8 users on the GPU proposer (single trials). On your own prompts the draft model was the pick at p95; with arctic-inference installed, a recalibration picked suffix decoding and measured +122% on prompts it never saw. PolyServe measures at your workload's concurrency.

## Hardware and engines measured

| Hardware | Backends tried | Measured |
|---|---|---|
| NVIDIA, compute capability ≥ 7.5 | vLLM, SGLang, llama.cpp (CUDA) | vLLM 0.11 and 0.29 and llama.cpp on an A40 and an L4, two-GPU layouts on a pair of A40s; vLLM 0.29 and SGLang 0.5.19 (from its own environment via `SGLANG_PYTHON`) on an A40, an A100, an RTX 4090 and, vLLM only, an H100 NVL |
| NVIDIA, compute capability < 7.5 | llama.cpp (CUDA) | **never benchmarked** |
| x86 CPU | llama.cpp; vLLM-CPU with AVX-512 | llama.cpp in a CPU container; vLLM-CPU **never benchmarked** |

Linux, Python 3.10–3.13.

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
- **Run-to-run variation** was 1–3% for vLLM on these synthetic prompts and up to 10% for llama.cpp. On real text, where answers stop when the model does, three interleaved runs spread by up to 5.6% (see [Your own prompts, SGLang, and a time budget](#your-own-prompts-sglang-and-a-time-budget)), so single-run gains of a few percent there are not conclusive. The same llama.cpp pick measured 508, 455 and 427 tok/s across calibration, comparison and ablation.

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

W8A8 weights (int8 weights and activations, `RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8`), graded on 14 September on vLLM 0.29 against a bf16 run of 86.4% on the same card: 85.1% (−1.3, 66 lost and 49 gained, p = 0.14). The two bf16 runs (87.2% and 86.4%) differ in engine version; each comparison is paired within its own run.

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
- The pick scored 603 tok/s in calibration and 547 in the comparison: choosing the best of many noisy trials flatters it, which is why the comparison re-measures with repeats. Of the five picks a comparison has re-measured, the other four came within 1.4% of their calibration number. Calibration now re-measures its best three before choosing and reports the fresh number (`--confirm`; on the p95 re-run the pick measured 601 tok/s in the search and 616 re-measured).
- The draft model roughly doubled time to first token (120 against 54 ms), inside the `chat` preset's 500 ms ceiling.
- Later stages then varied only the current leader, so SGLang was never measured with the fp8 cache once vLLM led; whether it would have won with it is unknown. After the batch stage SGLang's best (490 tok/s) was 2% behind vLLM's (500), so the search now tunes it as well: every engine within 10% of the leader gets the later stages. SGLang's backend offers no speculative decoding, so a draft model was never an option for it. On the p95 re-run SGLang was tuned alongside vLLM and reached 518 tok/s; vLLM with the draft model won.
- Flipped one at a time (single runs, back to back): without the draft model 509 tok/s against the pick's 571, so the draft model is worth about 12%, nearly all of the gain over stock vLLM. Without the fp8 cache 565 (−1%, within noise once the draft model is on, though it was worth 4% before the draft model joined); with the unquantized cache but FlashInfer attention kept 541. With 4-bit weights, which `--quant auto` leaves out for quality, GPTQ reached 712 (+25%) and AWQ 625 (+9%).

**SGLang alone on `sharegpt`** (`--backend sglang`, one run per row): PolyServe tuned SGLang to bf16 with the fp8_e5m2 KV cache at batch 64, 1952 tok/s against 1799 for stock SGLang (+8.5%, time to first token 364 against 318 ms). As on vLLM, the fp8 cache was the step that paid (1790 → 1948 tok/s in calibration); batch 16 left requests queueing (5.9 s to first token). SGLang's backend offers no speculative decoding yet, so none was tried. For scale only, since it came from another session: stock vLLM measured 1714 on the same workload and card.

**The same with `--budget 10m`:** calibration stopped after 6 trials in 580 s (precision, memory and batch for each engine) and skipped 7, the stages where the full run found its gain. The pick was plain vLLM bf16 at batch 64: 504 tok/s (504 / 505 / 503) against stock vLLM's 506 (518 / 503 / 506), which the comparison flagged as within run-to-run noise, and 474 for stock SGLang.

## The p95 search on hardware, and the new options

One A40 (vLLM 0.29.0, SGLang 0.5.19 through `SGLANG_PYTHON`) and one L4 (vLLM 0.11.0) on 14–15 September 2026, with everything built since the previous run switched on: the p95 ceiling, the busiest level measured first, runner-up tuning, the confirmation stage, the new speculative methods and W8A8 weights. Two L4 hosts failed first (downloads fell to 4.6 MB/s on one; the other lost its GPU mid-run, NVML "Unknown Error") and were replaced. The raw files are in [results-dolly/p95](../benchmarks/strategies/results-dolly/p95), [results-l4/p95](../benchmarks/strategies/results-l4/p95), [smoke-a40](../benchmarks/strategies/smoke-a40) and [task-quality-3b-w8a8.json](../benchmarks/strategies/task-quality-3b-w8a8.json).

**Your own prompts, judged at p95** (the Dolly-15k file above, vLLM and SGLang both candidates, every row 3 interleaved runs):

| row | median tok/s | runs | TTFT p95 |
|---|---|---|---|
| **PolyServe:** vLLM bf16, fp8 KV cache, 0.5B draft model, 16k prefill budget, batch 64 | **619** | 619 / 619 / 617 | 157 ms |
| stock vLLM 0.29 | 504 (PolyServe +22.8%) | 505 / 500 / 504 | 97 ms |
| stock SGLang 0.5.19 | 475 (PolyServe +30.3%) | 475 / 475 / 474 | 67 ms |

- The gain is outside run-to-run noise: PolyServe's slowest run beat both stock rows' fastest.
- The pick is the same kind as the median chose on these prompts before (a draft model on the fp8 cache), and it measured more: 619 tok/s against 547. Two things changed at once and are not separated: the warm-up, which keeps a first-use stall out of every measured level, and a 16k prefill budget in the pick. The stock rows barely moved (504 against 496, 475 against 473).
- The draft model met the 500 ms ceiling at 8 users with room to spare, at a p95 of 190–207 ms across its trials. GPU n-gram lookup qualified at 8 users too (487 tok/s) but trailed the fp8 cache (515).
- SGLang came within 10% of vLLM after the batch stage and got the later stages as well. Its best reached 518 tok/s; its backend has no speculative decoding, so it could not follow vLLM's draft model.
- SGLang with a 16k prefill budget ran out of memory at start-up with 93% of the GPU reserved, and the retry at 88% started. It did the same in the afternoon run: the first two times the retry has fired on hardware.
- The confirmation stage re-measured the best three: the pick measured 601 tok/s in the search and 616 re-measured.
- The calibration ran 35 trials in 46 minutes, against 21 trials in 53 minutes before the busiest level was measured first.

**The first p95 calibration of the same prompts** ([p95-old-warmup](../benchmarks/strategies/results-dolly/p95-old-warmup)) came out a tie: 523 tok/s (489 / 523 / 535) against 495 for stock vLLM and 492 for stock SGLang. It still used the old warm-up, two short requests to one slot, while the busiest level already ran first, so kernels that compile on first use did so during the level that is scored: the draft model's p95 at 8 users was 1213 and 1243 ms in its two trials and GPU n-gram lookup's 1008 and 1013 ms, and both were scored at 4 users. That is the artifact described in [Measuring the busiest level first](#measuring-the-busiest-level-first), fixed before the run above. That afternoon's `--budget 10m` run ([p95-old-warmup-budget](../benchmarks/strategies/results-dolly/p95-old-warmup-budget)) has the same flaw: its 7 trials reached the variations, none qualified at 8 users, and it served 499 tok/s against stock vLLM's 509. The budget's order has not been measured since.

**Qwen2.5-7B on `sharegpt` on the L4, judged at p95** (vLLM 0.11, every row 3 interleaved runs):

| row | median tok/s | runs | TTFT p95 |
|---|---|---|---|
| **PolyServe:** fp8 weights, 8k prefill budget, batch 64 | 219 at 8 users | 219 / 218 / 219 | 436 ms |
| stock vLLM with `--quantization fp8` | 212 at 8 users | 212 / 219 / 212 | 467 ms |
| stock vLLM bf16 | 128 at 8 users (PolyServe +71%) | 128 / 129 / 128 | 660 ms |

- The re-scored figures held: a tie with stock fp8 and +71% over stock bf16, every row at 8 users. None of the three rows kept the p95 under 1000 ms at 32 users.
- In the search the pick scored at 32 users with a p95 of 999 ms against the 1000 ms ceiling; the comparison measured it at 8. Whether a level meets the ceiling is decided by the second-slowest of its 32 requests, so a close call can go either way.
- The calibration ran 11 trials in 30 minutes, against 18 trials in 78 minutes before.

**Every new option on `extract`** (A40, Qwen2.5-3B, one trial per row measured at 1, 8 and 32 users in that order, [smoke-a40](../benchmarks/strategies/smoke-a40)), in tok/s:

| configuration | 1 user | 8 users | 32 users | TTFT p95 at 8 users |
|---|---|---|---|---|
| bf16 | 78 | 444 | 1044 | 490 ms |
| n-gram lookup on the CPU | 178 | 164 | 450 | 598 ms |
| n-gram lookup on the GPU (`ngram_gpu`) | 175 | 771 | 1406 | 1344 ms |
| 0.5B draft model | 142 | 646 | 1221 | 1620 ms |
| draft model, off above 8 running requests | 146 | 654 | 822 | 1657 ms |
| suffix decoding | 264 | 1141 | 1637 | 462 ms |
| fp8 KV cache (FlashInfer) | 77 | 538 | 1523 | 505 ms |
| int8 KV cache (`int8_per_token_head`, Triton) | 77 | 398 | 1040 | 1052 ms |
| W8A8 weights (`RedHatAI/Qwen2.5-3B-Instruct-quantized.w8a8`) | 109 | 659 | 1444 | 412 ms |

- Read the p95 column with care. These trials used the old warm-up and 8 users was each one's first batched level, so options whose kernels compile on first use (GPU n-gram lookup, the draft model, the int8 cache on Triton attention) paid for it there; on the Dolly re-run the draft model's p95 at 8 users fell from about 1.2 s to 0.2 s with the new warm-up. Throughput moves much less, since a one-second stall spreads over a level of 30 s or more.
- n-gram lookup's collapse under load was the CPU proposer. vLLM 0.29 turns async scheduling off for it and keeps it for the GPU proposer, which ran 4.7 times as fast at 8 users and beat plain bf16 at every load. The two differ in more than scheduling, so this does not prove async scheduling is the whole cause.
- Suffix decoding was the fastest option at every load, with a p95 at 8 users like bf16's. Its package, arctic-inference, is a vLLM plugin: with it installed, bf16 measured 78 / 441 / 1040 tok/s, as without it, and a second suffix run measured 263 / 1138 / 1649, within 1% of the first.
- Switching the draft model off above 8 running requests (vLLM's `num_speculative_tokens_per_batch_size`, not a PolyServe option) started, but at 32 users it ran slower than no speculation (822 against 1044 tok/s): the draft model still holds its memory.
- W8A8 weights ran without the kernel failure that stops vLLM 0.29's fp8 weights on this card, 38–48% faster than bf16. On GSM8K they cost Qwen2.5-3B 1.3 points: 85.1% against 86.4% for bf16 in the same run (1122 and 1139 of 1319), 66 problems lost and 49 gained, McNemar p = 0.14, not significant; fp8 cost the same model 2.4 points. They stay opt-in until a second model or task agrees.
- The int8 cache lost to both the unquantized and the fp8 cache, so Ampere is no longer offered it.

## Llama 3.1 8B on four GPUs

A second model family on four cards, on 15 September 2026, asking four things: does the pick hold on prompts calibration never saw, does it beat a rule of thumb and random search given the same time, what does calibration cost, and does measuring the busiest level first still pick the same. The model is Meta-Llama-3.1-8B-Instruct through unsloth's ungated copy (`unsloth/Meta-Llama-3.1-8B-Instruct`, so no Hub token was needed; its draft model is `unsloth/Llama-3.2-1B-Instruct`), served by vLLM 0.29.0 with arctic-inference 0.1.1 for suffix decoding, and by SGLang 0.5.19, on RunPod: an A40 (48 GB, Ampere), an A100 SXM (80 GB, Ampere), an H100 NVL (94 GB, Hopper) and an RTX 4090 (24 GB, Ada). Every calibration ran on the 300 Dolly-15k prompts above. Every comparison was measured on 300 other Dolly prompts that share none with them (`make_dolly_prompts.py --offset 300`) or on 300 OpenAssistant conversation openers (`make_oasst_prompts.py`), with 3 interleaved runs a row (2 on the H100, the dearest card, where SGLang was left out to keep the run short). The raw files are in [results-llama](../benchmarks/strategies/results-llama).

**On prompts the search never saw** (held-out Dolly, median run in tok/s, runs in brackets):

| card | PolyServe's pick | PolyServe | stock vLLM | stock vLLM fp8 | stock SGLang | calibration |
|---|---|---|---|---|---|---|
| A40 | vLLM bf16, 1B draft model | **505** (505 / 505 / 506) | 262 (+93%) | not run* | 257 (+97%) | 29 trials, 56 min |
| A100 | vLLM bf16, 1B draft model | **1153** (1161 / 1153 / 1145) | 724 (+59%) | not run* | 722 (+60%) | 30 trials, 44 min |
| H100 NVL | vLLM fp8, 1B draft model, 8k prefill budget | **2164** (2229 / 2164) | 1155 (+87%) | 1743 (+24%) | not run | 27 trials, 37 min |
| RTX 4090 | SGLang fp8 | **765** (765 / 779 / 764) | fails to start | fails to start | 448 (+71%) | 16 trials, 22 min |
| RTX 4090, after the contender fix below | vLLM fp8, suffix decoding | **807** (822 / 807 / 795) | | | 459 (+76%) | 31 trials, 39 min |

\* vLLM 0.29 cannot start fp8 weights on Ampere (see above); the comparison left the row out.

- The picks held on prompts they were not chosen on: the A100's measured 1133 tok/s in its calibration and 1153 held out, the H100's 2172 and 2164, the 4090's 778 and 765.
- Stock vLLM does not start on the 24 GB card. It reserves room for the model's full 131,072-token context, which does not fit beside 16 GB of bf16 weights, nor beside 8 GB of fp8 ones.
- On the Ampere cards, where vLLM 0.29 has no fp8 weights, the whole gain is the draft model, found on every card where vLLM led. It costs time to first token (the A40's p95 181 ms against 139 for stock vLLM), well inside the 500 ms ceiling. On Qwen2.5-3B on the A40 the same kind of pick gained 22.8%.
- On the H100 the stronger stock row is fp8, and the draft model on top of fp8 added 24%. The confirmation stage moved the pick from 1948 tok/s in the search to 2172 re-measured.

**On OpenAssistant prompts** (the same picks, on a dataset nothing else here uses):

| card | PolyServe | fastest stock | gain |
|---|---|---|---|
| A40 | 440 (440 / 440 / 438) | stock SGLang 264 | +67% |
| A100 | 1054 (1056 / 1047 / 1054) | stock vLLM 717 | +47% |
| H100 NVL | 1902 (1969 / 1902) | stock vLLM fp8 1755 | +8% |
| RTX 4090 (the pick before the fix) | 775 (775 / 775 / 774) | stock SGLang 460 | +68% |

- The gain shrank where the draft model had less to guess from: open-ended chat, against Dolly's many questions about a given passage. Against stock fp8 on the H100 it fell from +24% to +8%; against the fastest stock row on the A40, from +93% to +67%, and on the A100 from +59% to +47%.

**Qwen2.5-3B again, on the held-out prompts.** The same A40 calibrated Qwen2.5-3B afresh on the Dolly prompts, now with arctic-inference installed (the 14 September runs above did not have it, so they could not try suffix decoding). The pick was vLLM bf16 with the fp8 cache, suffix decoding and an 8k prefill budget, after 32 trials in 39 minutes, and on the held-out prompts it measured 994 tok/s (996 / 994 / 993, p95 72 ms) against 448 for stock vLLM (+122%) and 472 for stock SGLang (+110%). The earlier pick, a draft model, measured 619 on the calibration prompts. Suffix decoding was also the fastest option on `extract`; which speculative method wins is a property of the model, the card and the traffic, and here it was worth more than everything else together.

**Against a rule of thumb and random search, given the same time** (`benchmarks/search_baselines.py`: every pick re-measured in one session on the held-out prompts, 3 runs each):

| card | PolyServe | random search, same time | rule of thumb | stock |
|---|---|---|---|---|
| A40 | 505 (29 trials) | 505 and 508 (seeds 0 and 1: 28 and 30 of 720 configurations) | 257 | vLLM 262 |
| A100 | 1150 (30 trials) | 1151 (29 of 960) | 674 | vLLM 724 |
| RTX 4090 | 763 (16 trials) | 1039 (12 of 984) | 666 | SGLang 448 |
| RTX 4090, after the fix, head to head | 813 (31 trials) | 1049 (the same pick) | 646 | |
| RTX 4090, with the explore stage, 66 minutes each | 815 (41 trials, 9 of them explore draws) | 1045 (40 of 984) | 667 | SGLang 448 |

- **Random search did as well as the staged search on the Ampere cards, and better on the 4090.** Every random pick carried the draft model. Speculation has four settings in the space (none, GPU n-gram lookup, suffix decoding and the draft model), so roughly one configuration in four carries it, and 28–30 draws are all but sure to include some, and on these cards the other settings moved throughput by 1% or less.
- **The rule of thumb** (vLLM; fp8 weights where the card runs them; the engine's 8-bit KV cache; the shortest context the workload allows; batch as close to 256 as fits; 90% of memory; no speculation, which costs throughput at that batch size) was no better than stock on the Ampere cards, because the fp8 cache costs throughput there (the A100: 674 against 724). Measuring beat guessing on every card; which search did the measuring mattered less.
- **Why the staged search lost on the 4090.** SGLang in fp8 led vLLM in fp8 by 14% after the batch stage, and the later stages tuned only engines within 10% of the leader, so vLLM's speculative methods were never tried; SGLang's backend offers none. The search now also tunes an engine further behind when a stage has something to try on it and nothing on the leader's engine. Recalibrated that way, the 4090 found vLLM with suffix decoding (909 tok/s in the search, 813 held out), but random search's pick still measured 29% more: the draft model with an int8 KV cache, a 16k prefill budget and a 4k context. The staged search measured the draft model (878 tok/s) and the int8 cache (670) one at a time, and the combination stage joins only changes that came within 5% of the leader on their own, so it never tried them together. One change at a time cannot see an interaction that neither change shows alone; random sampling can stumble on one.
- What the staged search still gives: the same answer on every run, each strategy tried at least once, and a profile that says what each change was worth. What it did not give here: more throughput than random sampling at the same cost.
- **An explore stage did not close the gap.** `--explore on` spends about 30% more trials, after the stages, on random configurations of the leading engine and precision. On the same 4090 its 9 draws (vLLM fp8, the best 869 tok/s against the leader's 921 in the search) found nothing better, and random search given the same 66 minutes still won by 28% on the held-out prompts (1045 against 815). Its first 12 draws, the same as in the earlier run, already held the draft-model combination; the stages had spent their trials on one change at a time. The option stays, off by default.

**Tail latency.** Each level rests on 32 requests. Across 161 measured levels in the Llama calibrations the p95 time to first token ran from 38 to 350 ms, except one draft-model combination on the A100 at 770 ms at 8 users, a clear miss that was scored at 4 users instead. No level had the 500 ms ceiling inside its p95's distribution-free 95% confidence interval, so the close-call rule never re-measured one. The slowest requests of a level came in tight groups (the 770 ms level's interval was 769–770 ms), and for every pick in the comparisons the interval was at most 9 ms wide.

**Also learned on these cards:**

- SGLang 0.5.19 captures prefill CUDA graphs up to its prefill budget. With 8k and 16k budgets on the 24 GB card that capture ran out of GPU memory after the KV pool had left under 3 GB, and the retry with 88% reserved failed too, since the capture sits outside SGLang's memory fraction. The search skipped those trials; the planner does not model that memory.
- On the A40, vLLM 0.29's draft model on the fp8 KV cache (FlashInfer, the Ampere route) failed most requests with "CUDA error: an illegal memory access", where each worked alone. The search dropped those trials.
- The int8 KV cache (`int8_per_token_head`) ran on Hopper and Ada: on the H100 1679–1687 tok/s against 1734 for the fp8 cache, and it was part of random search's winning 4090 configuration.

## Judged at the 95th percentile

Every comparison above judged the `balanced` ceiling on the median time to first token, which lets half the requests run past it. PolyServe now judges it at the 95th percentile by default (`--ttft-percentile 50` restores the median). Every trial recorded both, so the same measurements can be re-scored without a GPU: `benchmarks/rescore_ttft.py` ranks each row under both rules, and its median columns reproduce the tables above. These numbers are **re-scored, not re-measured**, and PolyServe's pick is still the one the median chose; a calibration judged at p95 could choose differently, which only a new run can show.

In 8 of 20 comparisons the load the median chose sent more than 1 request in 20 past the ceiling. Those rows, by the median → at p95:

| machine, model, engine | workload | ceiling | PolyServe | vs stock bf16 | vs stock fp8 |
|---|---|---|---|---|---|
| A40, 3B, vLLM 0.11, 4-bit allowed | `high-concurrency` | 1000 ms | 1786 tok/s at 64 users → 1432 at 32 | +6% → +20% | +60% → +28% |
| A40, 7B, vLLM 0.11, 4-bit allowed | `chat` | 500 ms | 495 at 8 → 315 at 4 | +113% → +148% | +144% → +423% |
| A40, 3B, vLLM 0.11, `--phases auto` (served unified) | `rag` | 1500 ms | 106 at 8 → 46 at 1 | +1% → tie | +134% → +2% |
| two A40s, 3B, vLLM 0.11, two replicas | `high-concurrency` | 1000 ms | 3152 at 128 → 3049 at 64 | +84% → +143% | +134% → +164% |
| A40, 3B, llama.cpp only | `chat-system` | 500 ms | 455 at 4 → 368 at 1 | against stock llama.cpp: +190% → +135% | |
| L4, 7B, vLLM 0.11 | `chat` | 500 ms | 195 at 8 → 107 at 4 | stock misses the TPOT ceiling | tie → tie |
| L4, 7B, vLLM 0.11 | `sharegpt` | 1000 ms | 738 at 32 → 212 at 8 | +80% → +63% | +5% → tie |
| CPU, 0.5B, llama.cpp | `default` | 500 ms | 92 at 1 → misses (518 ms at p95) | against stock llama.cpp: +37% → stock meets it and the pick does not | |

The other 12 kept their load: every real-text pick on the A40 (`extract` three times, `sharegpt` on vLLM and on SGLang, `code-edit`), both Dolly runs, and `chat`, `chat-system`, `generation` and `rag-shared` on the A40 with vLLM 0.11. Some of their stock rows did not: stock fp8 on `chat` and `chat-system` fell back to fewer users, so PolyServe's lead over it grew from +36% to +125% and from +34% to +121%.

What it shows: the median hid queueing. The picks that changed were the ones served at the highest load the median allowed, and judged by the tail they serve fewer users at once. Stock settings queue worse at the tail as often as not, so the lead over stock grew in some rows and shrank in others; where it was already small against the stronger stock row it stayed a tie or became one (`rag`, and the L4 on `chat` and `sharegpt`).

Measured since ([The p95 search on hardware](#the-p95-search-on-hardware-and-the-new-options)): the L4 on `sharegpt` came out as re-scored, 219 tok/s at 8 users and a tie with stock fp8, and on the Dolly prompts, which the re-score left unchanged, PolyServe measured +22.8% over stock vLLM.

## Measuring the busiest level first

A trial measures each of its workload's concurrency levels in turn. Under `throughput` and `balanced` it scores at its fastest level that meets the limits, and throughput rises with load, so calibration now measures the busiest level first and stops at the first level that meets them. `benchmarks/early_stop_check.py` replays six recorded calibrations on an A40 that way, keeping for each trial only the levels a top-down run would have measured:

| calibration | trials | scores changed | same pick | measuring time skipped | of calibration time |
|---|---|---|---|---|---|
| `extract`, run 1 | 26 | 0 | yes | 18 of 32 min | 36% |
| `extract`, run 2 | 26 | 0 | yes | 17 of 32 min | 36% |
| `extract`, run 3 | 26 | 0 | yes | 17 of 32 min | 36% |
| Dolly prompts | 21 | 0 | yes | 15 of 26 min | 29% |
| Dolly prompts, `--budget 10m` | 6 | 0 | yes | 3 of 5 min | 34% |
| SGLang on `sharegpt` | 10 | 0 | yes | 10 of 13 min | 44% |
| Llama 3.1 8B on an RTX 4090, Dolly prompts, `--all-levels` | 12 | 0 | yes | 14 of 19 min | 45% |

No score and no pick changed, at the 95th percentile or at the median; the skipped levels took 35% of calibration time overall. The last row was recorded for this check on 15 September, with the current warm-up and every level of every trial measured (`--all-levels`), on a second model family and card: the same calibration took 22 minutes with the early stop and 31 without, for the same pick. Start-up and warm-up (33–76 s a trial) are untouched. Timed on GPUs since: 28 trials in 34 minutes and 35 in 46 on Dolly prompts on an A40, and 11 in 30 minutes on `sharegpt` on an L4, against 21 in 53 and 18 in 78 before. It can misjudge a trial whose throughput falls as load rises, as n-gram speculation's did; in these tables that happened only to configurations that lost either way. `latency` and `efficiency`, which can prefer a quieter level, and calibrations with a power stage, which compare energy per token over every level, still measure all of them.

What the replay could not show: in the old order the busiest level ran last, after quieter levels had warmed the engine; run first, it pays for kernels that compile on first use. In the recorded quiet-to-busy calibrations that cost fell on the first batched level instead. On Dolly prompts every draft-model trial had a p95 time to first token of 1.3–1.6 s at 4 users and 0.12–0.18 s at 8 users right after. On an A40 on 14 September, with the busiest level first, the draft model, GPU n-gram lookup and the int8 cache on Triton attention showed 0.6–1.2 s at 8 users, and vLLM 0.29 logged "Triton kernel JIT compilation during inference". Two such requests out of 32 set a p95, so under the p95 ceiling that spike alone kept speculation out of the Dolly pick. The warm-up now sends one short request per slot at the busiest level before any level is measured, instead of two requests to one slot; on the Dolly re-run it removed the spike for vLLM: the draft model's p95 at 8 users fell from 1.2 s to 0.19–0.21 s, and GPU n-gram lookup qualified at 8 users too. It has not been checked on SGLang or llama.cpp.

## When a calibration pays for itself

Calibration occupies the GPU without serving. Against the fastest stock setup that met the limits in the same comparison, the pick's extra throughput repays that time after `calibration seconds × stock tok/s ÷ (PolyServe tok/s − stock tok/s)` of serving at the load the comparison scored. `benchmarks/break_even.py` computes it from the recorded comparisons. It prices throughput, for a server kept busy; at light traffic a pick's value is its latency instead, which this does not count.

| comparison | calibration | PolyServe tok/s | fastest stock that met the limits | gain | break-even |
|---|---|---|---|---|---|
| A40, 7B, `chat` (4-bit allowed) | 26 min | 495 | stock vLLM 233 | +112% | 0.4 h |
| A40, 3B, `chat` (4-bit allowed) | 21 min | 783 | stock vLLM fp8 575 | +36% | 1.0 h |
| A40, 3B, `extract` | 46 min | 731 | stock vLLM 439 | +67% | 1.2 h |
| A40, 3B, `generation` (4-bit allowed) | 63 min | 1157 | stock vLLM fp8 837 | +38% | 2.7 h |
| A40, 3B, your own prompts at p95 | 46 min | 619 | stock vLLM 504 | +23% | 3.4 h |
| A40, 3B, SGLang on `sharegpt` | 22 min | 1952 | stock SGLang 1799 | +9% | 4.3 h |
| A40, 3B, `sharegpt` | 44 min | 1891 | stock vLLM 1714 | +10% | 7.1 h |
| A40, 3B, `high-concurrency` (4-bit allowed) | 32 min | 1786 | stock vLLM 1686 | +6% | 9.1 h |
| A40, 3B, `code-edit` | 39 min | 483 | stock vLLM 459 | +5%, single run | 13 h |
| L4, 7B, `sharegpt` at p95 | 30 min | 219 | stock vLLM fp8 212 | tie | never |
| A40, 3B, `rag` | 58 min | 106 | stock vLLM 105 | tie | never |
| L4, 7B, `chat` | 45 min | 195 | stock vLLM fp8 195 | tie | never |
| A40, 3B, your own prompts, `--budget 10m` (old warm-up) | 9 min | 499 | stock vLLM 509 | −2% | never |

- Where the pick beat the fastest stock setup by 20% or more, the calibration paid for itself within 0.4–3.4 hours of busy serving; at 5–10% it took 4–13 hours; a tie never pays it back (the script's arithmetic gives 16 to 1,079 hours for these, which is noise divided by noise).
- The rows marked "4-bit allowed" were picked when `--quant auto` still included 4-bit checkpoints (`--quant auto,awq,gptq` today), which cost the 3B model 4–5 GSM8K points.
- A calibration that finds nothing still costs its half hour. `--budget` caps that cost, at the price of the trials it skips.

**Llama 3.1 8B**, and Qwen2.5-3B recalibrated with suffix decoding available ([above](#llama-31-8b-on-four-gpus)), calibrated on Dolly prompts and measured on prompts calibration never saw:

| comparison | calibration | PolyServe tok/s | fastest stock that met the limits | gain | break-even |
|---|---|---|---|---|---|
| RTX 4090, held-out Dolly | 22 min | 765 | stock SGLang 448 | +71% | 0.5 h |
| RTX 4090, OpenAssistant | 22 min | 775 | stock SGLang 460 | +68% | 0.5 h |
| RTX 4090, held-out Dolly, after the contender fix | 39 min | 807 | stock SGLang 459 | +76% | 0.9 h |
| A40, held-out Dolly | 56 min | 505 | stock vLLM 262 | +93% | 1.0 h |
| A40, OpenAssistant | 56 min | 440 | stock SGLang 264 | +67% | 1.4 h |
| A100, held-out Dolly | 44 min | 1153 | stock vLLM 724 | +59% | 1.2 h |
| A100, OpenAssistant | 44 min | 1054 | stock vLLM 717 | +47% | 1.5 h |
| H100 NVL, held-out Dolly | 37 min | 2164 | stock vLLM fp8 1743 | +24% | 2.6 h |
| H100 NVL, OpenAssistant | 37 min | 1902 | stock vLLM fp8 1755 | +8% | 7.4 h |
| A40, Qwen2.5-3B, held-out Dolly (suffix decoding) | 39 min | 994 | stock SGLang 472 | +110% | 0.6 h |

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

## Limits

- The order of the search has added nothing measured so far. Given the same time, random sampling of the same settings matched the staged search on an A40 (two seeds) and an A100 and beat it by 29% on an RTX 4090, and by 28% again once an explore stage of random draws around the leader was added ([details](#llama-31-8b-on-four-gpus)). What pays is the set of settings tried and how each is measured: a fixed rule of thumb (fp8 weights and KV cache, a short context, batch 256), what an informed user might set without measuring, lost to the pick by 15–97% on the same three cards and did no better than stock settings on the two Ampere ones. Against someone who already knows the winning flags for this card and this traffic PolyServe ties by construction; knowing them takes measuring.
- Never run on real hardware: vLLM-CPU, pre-Turing GPUs, and `--power`, which has only run against a simulated NVML.
- The staged order changes one setting at a time and combines only changes that paid on their own, so it misses interactions: the 4090's random winner paired a draft model with an int8 KV cache and a 16k prefill budget, which neither showed alone. At 8 users a p95 rests on 32 requests, so its second-slowest can decide; a level too close to its ceiling to call is measured again with as many requests, which no Llama calibration needed. The budget's new order has not been measured with the current warm-up.
- Calibration never evaluates answer quality. The quality results come from a separate script, run by hand, on one task (GSM8K) and one model family.
- Measured on two model families (Qwen2.5 and Llama 3.1 8B) and six kinds of machine, for speed; answer quality only on Qwen2.5.
- Calibration takes half an hour to an hour per workload: 22–56 minutes for Llama 3.1 8B on four cards, 30–46 for Qwen2.5, against 53–78 before trials measured their busiest level first. Where the pick beat the fastest stock setup by 20% or more, that time was repaid within 0.4–3.4 hours of busy serving (0.5–2.6 for Llama on held-out prompts), and at 5–10% within 4–13 hours; a tie never repays it ([break-even](#when-a-calibration-pays-for-itself)). `--budget 10m` caps it further, at the price of skipped trials.

## Not yet measured

These are gaps, not claims. In rough order of how much they would change the conclusions:

1. **Task quality beyond one benchmark.** On GSM8K, quantization cost Qwen2.5-3B 2–5 points and Qwen2.5-7B about one, within noise, and fp8 with activation quantization on Ada cost 7B nothing. Other tasks (code, long-context retrieval), other model families, llama.cpp's GGUF formats and Hopper are ungraded, so whether 4-bit checkpoints could return to `--quant auto` for larger models is still open.
2. **A model where memory truly binds.** Qwen2.5-7B on a 24 GB L4 was bandwidth-bound, not memory-bound: its grouped-query KV cache is small, and the presets reach at most 32 concurrent requests. A larger model on the same card (a 14B in fp8, or a model without grouped-query attention), or long-context traffic at high concurrency, is where batch and cache sizing would decide the result.
3. **Other accelerators.** An A100, an H100 NVL and an RTX 4090 ran Llama 3.1 8B on 15 September ([Llama 3.1 8B on four GPUs](#llama-31-8b-on-four-gpus)); an A30 and a pre-Turing card (GTX 1080) are untested, so the compute-capability branch in the selector has never run on real hardware. SGLang 0.5.19 has run on an A40, an A100 and an RTX 4090, on Dolly and OpenAssistant prompts, and never with speculative decoding, which its backend does not offer yet.
4. **Beating an expert, not just the defaults.** Where the pick won big, it chose what an informed user could also pass by hand: fp8 on the L4, a draft model on `extract`, a draft model or suffix decoding on Dolly prompts. Someone who already knows the winning flags for a card and its traffic ties PolyServe. Someone who follows a fixed rule of thumb does not, because the winning flags changed from card to card: the rule lost to the pick by 15–97% on three cards ([Llama 3.1 8B on four GPUs](#llama-31-8b-on-four-gpus)). Whether a human expert's guess, rather than a written rule, would do better has not been measured. The search can also miss: on `extract` the ablation found a configuration 19% faster than the pick, and only then did the combination stage learn to undo adopted changes (a rerun found it). Given the same time, random search over the same settings matched the staged search on an A40 (two seeds) and an A100 and beat it by 29% on an RTX 4090, where it found an interaction the one-change-at-a-time stages cannot see. The value is in measuring on the card and the traffic; the order of the search has not yet shown one of its own. A search that beats random sampling at equal cost would change that; the staged pass followed by random draws around the leader (`--explore on`) did not, on the 4090, where random search still won by 28%. Sampling at random inside the leading engine and precision from the start is the next idea, untested.
5. **Energy tuning on real hardware.** `--power` has only run against a simulated NVML. It needs root on the host, so it has to be measured on a machine you control; `benchmarks/measure_power.sh` runs the whole measurement in one command and restores the GPU afterwards. Whether the energy-optimal point sits near 70% of full power for these workloads, and whether it differs between prefill-heavy `rag` and decode-heavy `generation`, is still a prediction.
6. **Disaggregated prefill and decode at a scale where it could pay.** On two PCIe-linked A40s with a 3B model it ran end to end but lost to one engine (102 against 105 tok/s, time to first token 3.1 s against 1.3 s), and one of the two pairs tried failed a quarter of its requests with KV blocks the decode engine never pulled. Published gains come from larger models, NVLink or RDMA between the engines, and heavier prefill contention, none of which was available here.
7. **Speculative decoding by load.** On real text on the A40 (vLLM 0.29, fp8 cache on FlashInfer) n-gram speculation made one user up to 80% faster and collapsed from four users up; on the L4 (vLLM 0.11, native fp8 cache) it neither helped nor collapsed on `sharegpt`. Which part of the A40 setup causes the collapse is unmeasured. One candidate: vLLM 0.29 runs its CPU n-gram proposer without async scheduling. On `extract` its GPU proposer (`ngram_gpu`), which keeps it, did not collapse, and suffix decoding was faster still ([The p95 search on hardware](#the-p95-search-on-hardware-and-the-new-options)); both have since run inside calibrations, and suffix decoding was the pick for Qwen2.5-3B on Dolly prompts on an A40 and for Llama 3.1 8B on an RTX 4090 ([Llama 3.1 8B on four GPUs](#llama-31-8b-on-four-gpus)). A draft model paid on `extract`. A server that switches speculation on only at low load would get the single-user gain without the collapse; PolyServe picks one setting per workload. llama.cpp's `ngram-mod` has not been measured on real text.
8. **Memory outside vLLM's reservation.** At batch 512 with the fp8 cache and 95% memory utilization, vLLM 0.11 ran out of memory at start-up twice; vLLM 0.29 started that shape every time it was tried. The planner does not model that memory, and the evidence gives no single size for it, so calibration now retries such a failure with 5 points less memory reserved (down to 0.85) instead. The retry has fired twice since, on SGLang with a 16k prefill budget at 93%, and started at 88% both times. A planner term would need start-up logs from several engine versions.
9. **More calibrations at p95.** Dolly prompts (Qwen2.5-3B, and Llama 3.1 8B on four cards) and `sharegpt` on an L4 have been calibrated and compared at p95 ([The p95 search on hardware](#the-p95-search-on-hardware-and-the-new-options), [Llama 3.1 8B on four GPUs](#llama-31-8b-on-four-gpus)); the other tables are re-scored from runs judged by the median. The budget's new order has only run with the old warm-up. W8A8 weights have run as single trials on one workload, not inside a calibration. And at 8 users a p95 rests on 32 requests, so the second-slowest can decide whether a level meets the ceiling; a level whose ceiling falls inside its p95's confidence interval is now measured again with as many requests. Across the Llama calibrations on four cards no p95 came near the ceiling, so it has not fired yet.
