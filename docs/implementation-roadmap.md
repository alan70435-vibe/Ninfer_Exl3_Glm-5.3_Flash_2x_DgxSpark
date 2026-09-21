# Implementation roadmap

This roadmap is ordered by dependency, not by how visually impressive a feature
looks. Each phase must leave an inspectable correctness boundary for the next.

## P0 — executable model/deployment contracts (current)

**Implemented**

- authoritative host `ModelSpec` for the 45-layer GLM-5.3-Flash topology;
- two-node deployment parser and validation;
- TP2 rank geometry and layer execution plan;
- KDA versus sparse-MLA persistent-state classification;
- DFlash2 plan fields;
- read-only hardware/fabric preflight;
- checkpoint manifest scanner;
- host CI and tests.

**Exit criterion:** clean host build/tests and a deployment plan generated from
the sanitized `Dgx_Spark_Best_Settings` baseline.

## P1 — real checkpoint semantic inventory and binder

1. Run `checkpoint_manifest.py` against the exact target and DFlash2
   checkpoints mounted on DGX Spark.
2. Capture all tensor names/shapes/encoded metadata without loading full weights.
3. Add `Glm53ParameterId`/logical-name schema independent of Hugging Face names.
4. Implement a strict source-name mapper for all 45 target layers and draft
   components.
5. Add shape/format/coverage checks: every required logical parameter exactly
   once, no silent extras used as substitutes.
6. Produce a deterministic mapping receipt containing checkpoint revisions and
   tensor-index hashes.

**Exit criterion:** target + draft checkpoints bind completely on CPU metadata;
no CUDA execution is needed to prove mapping completeness.

## P2 — EXL3/TR3 encoded storage and materialization

1. Document the actual 4-bpw TR3 encoding from checkpoint metadata/source
   implementation rather than inferring it from filenames.
2. Add encoded-parent/view types that preserve packed weights.
3. Implement rank-local materialization and ownership accounting.
4. Decide with evidence whether NInfer artifact-v3 can express the representation
   directly; extend format/layout names or define a new artifact version only if
   necessary.
5. Add byte-level round-trip/decoder reference tests on small tensor slices.
6. Build a memory planner for two 121-GiB unified-memory nodes including target,
   draft, graph workspace, KDA state and 1M target KV budget.

**Exit criterion:** both ranks can map the full checkpoint into deterministic
rank-local encoded views with memory-budget reporting and no model execution.

## P3 — GLM GPU operators on GB10 / SM121

Implement and validate in dependency order:

1. embedding/final norm/head and basic normalization/residual primitives;
2. EXL3 linear primitives for the exact GLM matrix geometries;
3. three dense FFN prefix layers;
4. KDA input projections, short convolution, recurrent update and output;
5. sparse-MLA Q/KV projections and target KV materialization;
6. sparse indexer query/key path, top-k 2048 selection and attention gather;
7. sparse MoE routing, top-8 dispatch, expert GEMMs and shared expert;
8. graph-safe decode variants and prefill variants.

Every public operator needs a CPU/high-precision oracle or a trusted framework
parity fixture, eager + graph replay tests, boundary-shape tests and
compute-sanitizer coverage before performance tuning is accepted.

**Exit criterion:** one rank can execute every GLM layer/operator correctly on
synthetic and checkpoint-backed fixtures; performance can still be untuned.

## P4 — TP2 / NCCL / RoCE runtime

1. one process per Spark, deterministic rank bootstrap;
2. CUDA/NCCL stream/event ownership and async error propagation;
3. TP sharding for each implemented projection/expert matrix;
4. branch-output all-reduce boundaries around mHC updates from `ExecutionPlan`;
5. sequence transaction id + committed-position checks on both ranks;
6. communication/computation overlap only after a serially equivalent path is
   proven;
7. dedicated collective microbenchmarks on the 200-Gb/s CX-7 fabric.

**Exit criterion:** logits/intermediate checkpoints from TP2 match the trusted
single-process/reference execution within the chosen quantized tolerances over
prefill and decode fixtures.

## P5 — stateful target runtime and long context

1. sequence allocator and memory accounting;
2. KDA recurrent/conv state arena with snapshot/commit/rollback;
3. paged sparse-MLA fp8 KV cache;
4. indexer state/metadata coupled transactionally to KV pages;
5. prefix/context ownership and eviction rules;
6. one-million-token admission checks based on *real* allocated bytes;
7. long-prefill chunking/fairness that preserves decode service quality.

**Exit criterion:** long-context generation survives state transitions,
cancellation and request reuse without leaks or state corruption.

## P6 — DFlash2 k=7 first-class path

1. bind and materialize the pinned DFlash2 revision;
2. TP2 proposal execution;
3. target verification block execution;
4. deterministic rank-0 acceptance + broadcast;
5. target and draft transactional state commit;
6. graph capture/warmup for k=7 shapes;
7. acceptance/throughput counters required for later k A/B.

Then add controlled k=5/k=4 variants with their own graph-shape catalogs.

**Exit criterion:** exact-output temperature-zero fixtures match target-only
execution and both ranks remain state-identical across partial acceptance,
full acceptance, rejection and cancellation cases.

## P7 — scheduler, sampler, continuation and API

- single-inflight scheduler first (`MAX_NUM_SEQS=1` baseline);
- authoritative rank-0 RNG/sampler state with deterministic replay;
- stop/token/tool/reasoning parser continuation state;
- OpenAI-compatible streaming API;
- cancellation/backpressure/timeouts;
- optional concurrency only after graph/KV/state capacity measurements.

**Exit criterion:** interrupted/resumed deterministic fixtures produce identical
continuations, and API conformance tests cover streaming/non-streaming and tool
paths.

## P8 — parity, profiling and tuning

Compare against the current vLLM baseline using the same fixed corpora and
sampling configuration. Track at minimum:

- TTFT and prompt throughput by prompt length;
- stream and aggregate output tok/s;
- TPOT and tail latency;
- DFlash proposed/accepted tokens by position;
- target/draft/KV/KDA memory;
- NCCL time and overlap;
- GPU utilization, clocks, power/energy per output token;
- API failures, NaN/Inf and exact-output/parity checks.

Only after correctness should we tune DFlash k, draft TP, graph catalog,
collective algorithms, clocks or additional concurrency.
