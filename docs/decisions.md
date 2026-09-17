# Decisions, and the measurement behind each

Every design choice here was settled by a measurement, and several were settled against the first guess. This
page is the short answer to each, with the number that decided it and where the full evidence lives. Where the
evidence is thin or absent, it says so; those are the honest weak points, listed again in
[Limits](benchmarks.md#limits) and [Not yet measured](benchmarks.md#not-yet-measured).

## Why not just random search?

Random sampling of the same space is a strong baseline, and it was measured rather than dismissed: given equal
time it matched the staged search on an A40 and an A100 and beat it by 29% on an RTX 4090
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

What is worth keeping is not the ordering but the space: random search samples the 720–984 configurations that
remain after the memory planner drops what cannot fit and the capability filter drops what the engine cannot do.
Run blind over the raw product of settings and most draws fail to launch. Random search here *is* PolyServe
minus the stage order, and it inherits the expensive parts — the planner, the measurement harness, the objective.

The stages buy three things random draws do not: the same answer on every run, where a seed change moves random's
pick; every strategy tried at least once, so the profile says what each setting was worth instead of only naming
a winner; and a sensible order under `--budget`, where the first handful of trials decide everything.

The case where staging should win outright is a space where good configurations are rare, rather than the roughly
one in four that carry speculative decoding on those cards — and that case has since been measured twice, with
two different answers. On a 14B in fp8 on a 24 GB card, where only 2 of 54 configurations fit, the staged search
measured 385 tok/s against random sampling's 369 at equal time, ranges apart. On the same card in the
788-configuration space that also contains 4-bit checkpoints, the two tied: 1412 against 1416, with random search
given 2524 s to the staged search's 3028 s and sampling 3% of the space
([evidence](benchmarks.md#qwen25-14b-where-memory-binds)).

So the honest score is one win in a small space and one tie in a large one, and the prediction held in only half
the cases it was made for. Nothing here supports "the search order is what pays"; what it supports is the
narrower claim these stages were built for — the same answer every run, every strategy tried at least once, and
a profile that says what each setting was worth.

Two ways of doing better were then measured against the recorded calibrations, and neither pays.

**Reaching the answer sooner.** If staging found its pick in a handful of trials while random sampling needed
dozens, the claim would be speed rather than quality. Across 13 recorded calibrations it reaches 95% of its own
final answer at a median of **13 of 25 trials**, having spent 43% of its wall-clock — against the 24 to 44 draws
random search needed. That is roughly a factor of two, not the order of magnitude that would make "calibration in
five minutes" true, and the spread is wide: two runs got there in 3–4 trials, three needed 16–19.

**Sampling from a prior instead of uniformly.** The obvious way to beat random sampling is to draw configurations
in proportion to how good they are predicted to be, using the performance predictor fitted from previous
calibrations — memory across runs being the one advantage random sampling cannot have. Leave one calibration out,
fit on the rest, and ask where the predictor ranks the configuration that actually won: **median rank 15 of ~25,
and never in the top 5 across 8 folds** — and those 25 are already the configurations the staged search chose to
measure, so they are biased towards good ones. A rank correlation of 0.88–0.94 is enough to prune quantizations
that cannot win and nowhere near enough to order near-optimal configurations against each other, which is exactly
what a sampler needs. Sampling on those scores would be worse than uniform.

Both were computed from data already on disk, before spending anything on a GPU, which is the only reason it was
cheap to learn that the search is at its practical ceiling for this space.

## Why measure at all, instead of a rule of thumb?

Because a defensible rule of thumb lost, including to doing nothing. The rule (fp8 weights and KV cache, a short
context, batch 256) came in 56–97% behind the measured pick on three cards, and on both Ampere cards it was
*slower than stock defaults* — 674 against 724 tok/s on an A100 — because an fp8 KV cache costs throughput there
while helping on Ada ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)). The winning setting also changed with
the prompts on one card: suffix decoding on Dolly-15k, a draft model on extraction, an fp8 cache on real chat
([evidence](benchmarks.md#re-measured-on-an-a40)). A fixed rule cannot follow that.

## Why compare against stock defaults at all?

Stock `vllm serve` is what a machine runs when nobody tunes it, so it is the honest zero point, but it is a weak
opponent and the headline should not rest on it. The rule-of-thumb comparison is the one that answers "could an
informed engineer have guessed this?", which is why the README leads with it. Against someone who already knows
the winning flags for this card and this traffic, PolyServe ties by construction — knowing them takes measuring
([limits](benchmarks.md#limits)).

## Why change one setting at a time?

So the profile can attribute the gain: each stage's trials differ from the leader in one dimension, which makes
the table readable as "what each change was worth". The cost is interactions, and that cost is real: on the 4090,
an int8 KV cache and a 16k prefill budget each did nothing alone but were part of a configuration 29% faster with
a draft model. Two mitigations followed, and only the second worked — a stage of random draws around the leader
(`--explore on`) found nothing better in 9 draws, while trying speculation first and tuning everything else per
speculation method closed the gap ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why try speculative decoding before the other settings?

Because the other settings pay differently depending on it, so their order relative to it decides the result. On
the 4090 the int8 cache and the prefill budget were worthless alone and valuable with a draft model. Speculation
first, then the remaining stages once per each of the two best speculation methods, reached 1041 tok/s in 24
trials and 43 minutes where the previous order reached 813 in 31 trials and 66 minutes. On the A40 and A100 it
reached the same picks as before in about two-thirds of the time ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why tune engines that are already behind?

Because a lead after three stages is not a lead after all of them. SGLang led vLLM in fp8 by 14% on the 4090, so
tuning only the leader never tried vLLM's speculative decoding, which SGLang's backend does not offer — and that
is where the card's best configuration was. The search now also tunes an engine further behind when a stage has
something to try on it and nothing on the leader's, and drops an engine that a tuned rival has left more than 10%
behind with nothing more to offer ([design](writeup.md#3-staged-search)).

## Why re-measure the best three configurations at the end?

Because the best of many noisy trials is flattered by the choosing. Of five picks re-measured before this stage
existed, four landed within 1.4% of their calibration number and one measured 9.3% lower. The confirm stage costs
up to three trials and decides on the fresh runs alone: on the Dolly p95 re-run the pick measured 601 tok/s in the
search and 616 re-measured ([option](usage.md#search-options)).

## Why measure the busiest concurrency level first?

Because most trials die there, and dying early is cheap. Trials run their heaviest level first and stop when it
cannot beat the leader, which cut 29–45% off calibration across the Llama runs and changed no pick
([evidence](benchmarks.md#measuring-the-busiest-level-first)).

## Why score at the 95th percentile rather than the mean?

Because a mean hides the requests users complain about, and a trial that looks fastest can be breaking its latency
ceiling at its fastest level. Scoring at p95 demotes such a trial to the level it actually meets: a draft-model
combination on the A100 whose p95 reached 770 ms at 8 users was scored at 4 users instead
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)). Each level's p95 rests on 32 requests, so its second-slowest
request can decide a close call — one pick scored at 32 users with a p95 of 999 ms against a 1000 ms ceiling
([evidence](benchmarks.md#the-p95-search-on-hardware-and-the-new-options)) — and a level too close to call is
measured again, a rule that never fired on the Llama runs and is reported rather than quietly dropped
([evidence](benchmarks.md#judged-at-the-95th-percentile)).

## Why plan memory instead of launching and catching the failure?

Because a failed launch costs 30 seconds and tells you nothing, while an estimate costs nothing and removes most
of the grid. The planner predicts weights, KV cache and workspace, and its error against measured peak memory is
tracked as a first-class result rather than assumed ([accuracy](benchmarks.md#memory-planner-accuracy)). The
sharper reason is that a wrong memory setting silently produces a *slow* run rather than a crash: on an RTX 3090
llama.cpp with 27 of 36 layers on the GPU measured 35 tok/s where a full offload measured 545, so quantizations
have to be compared at the memory setting they would actually be served at ([design](writeup.md#2-memory-planner)).

## Why is the explore stage off by default?

Because it did not pay. It spends about 30% more trials on random configurations of the leading engine and
precision; on the 4090 its 9 draws found nothing better than the stages had, and random search given the same 66
minutes still won by 28%. The option stays, off, with the negative result recorded
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why is answer quality checked separately from calibration?

Because the obvious way to enforce it during tuning was built, measured, and did not work. `--quality-probe`
answers the calibration prompts greedily at each precision while its engine is up and records how far they drift
from the most faithful precision that runs. Making that a *constraint* — drop a precision that drifts too far —
is what failed, and the numbers say why: identical fp8 weights answering the same prompts twice differed on 10%
of answer openings, while 4-bit and int8 checkpoints differed on 83–100%. There is no threshold between those,
and GSM8K put the same 4-bit weights about 2 points behind, so exact-match drift and answer quality disagree
outright ([measurements](benchmarks.md#qwen25-14b-where-memory-binds)).

So the probe reports and never refuses. It tells an operator that a precision rewrites most of its answers,
which is worth knowing before choosing it; deciding whether those answers are *worse* needs labels, and that
stays a separate, labelled step.

Three things about it are worth stating plainly. Its first run on hardware did nothing at all: the probe was never
built, because the guard tested prompts that a file workload only materialises later, so a 14B calibration kept a
4-bit checkpoint that had been compared against nothing (fixed, with a regression test).

Its first design was also too strict to be useful. Comparing whole 128-token greedy completions across 16 prompts
meant one changed answer was 6% — past any sane tolerance — and greedy decoding diverges as soon as a single
token differs, so nearly every cheaper precision would have been refused, with nothing to distinguish a real
change from the engine's own non-determinism. It now compares the first 32 tokens of 48 answers and measures each
precision twice, so the reference's disagreement with itself is the floor a candidate must clear.

And even working, it is not an accuracy guarantee: instruction-following answers can agree perfectly while
multi-step arithmetic degrades, which is exactly what happened on that card — near-zero drift where GSM8K lost
about 2 points ([evidence](benchmarks.md#qwen25-14b-where-memory-binds)). Labelled grading stays a separate step.

The separate, labelled check remains, because calibration measures speed and latency, and quantization can cost
accuracy that no throughput number shows. The separate check is deliberately narrow and its narrowness is the weak point: GSM8K only, where
quantization cost Qwen2.5-3B 2–5 points and 7B about one, and fp8 with activation quantization on Ada cost 7B
nothing. Other tasks, other families and llama.cpp's GGUF formats are ungraded, which is why 4-bit checkpoints
are not in `--quant auto` for larger models ([gaps](benchmarks.md#not-yet-measured)).

## Why not Bayesian optimisation?

At roughly 30 trials over a space this shape, the published results for SLO-aware Bayesian tuners do not separate
from staged or random search, and the machinery costs a dependency and reproducibility. Random search already
matches the staged search here — a smarter optimiser is worth adding only once there is a space where blind
sampling visibly degrades, which is the same missing experiment named above
([roadmap](writeup.md#9-non-goals-and-roadmap)).

## What would change these answers?

- Long-context traffic at high concurrency, where batch and cache sizing decide the result. The memory-bound
  half of this gap has now been measured twice on a 14B at 24 GB, and split: the staged search won by 4.4% in a
  104-configuration space and tied in a 788-configuration one
  ([details](benchmarks.md#qwen25-14b-where-memory-binds)). A third measurement would say which is typical.
- Traffic that drifts after calibration: profiles are cached per machine, model, objective and workload, and
  nothing re-tunes when the prompts change shape.
- A second person running a calibration on hardware and traffic the author does not control.
