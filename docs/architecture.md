# Architecture contract

## 1. Product boundary

This runtime targets one production shape:

- GLM-5.3-Flash text runtime first;
- EXL3/TR3 target weights;
- exactly two NVIDIA DGX Spark GB10 nodes;
- tensor parallel size two across the dedicated CX-7 RoCE fabric;
- DFlash2 speculative decode as the first optimized decode path;
- one-million-token configured context with the model's native 1,048,576-token
  limit as a hard upper bound.

Generality is added only when it makes this target faster, safer or easier to
validate. There is no requirement that a mechanism work on arbitrary GPUs,
world sizes or model families.

## 2. Why this is a product fork, not a model-name port

Current NInfer has strong boundaries worth preserving, but its shipped model
runtime is centered on Qwen3.5/3.6/3.8 and a single-GPU product. GLM-5.3-Flash
changes all of the following simultaneously:

1. **Mixer topology:** 34 KDA/linear-attention layers and 11 sparse-MLA layers.
2. **Persistent state:** KDA recurrent/conv state is fundamentally different
   from sparse-MLA KV + index state.
3. **FFN topology:** three dense prefix layers followed by 42 routed-MoE layers.
4. **Sparse attention:** the MLA path includes an indexer contract, not just a
   dense KV cache.
5. **Speculation:** DFlash2 has its own model/state and a target-verification
   transaction.
6. **Distribution:** both target and baseline draft run TP2, so request commit,
   rollback and sampling decisions must be rank-consistent.

Treating those as loader-only differences would force state and collective
semantics into unrelated kernel code. This project instead makes them explicit
contracts first.

## 3. Layer topology

The host model contract is the source of truth for logical layer kind. Sparse
MLA layers are:

`3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43`

All remaining 34 layers are KDA. Layers 0–2 use dense FFN; layers 3–44 use
sparse MoE.

Important fixed geometry currently captured in code:

- hidden size 4096;
- four mHC hidden streams; mHC epsilon `1e-6`, 20 Sinkhorn iterations;
- mHC base/scale and KDA decay parameters are FP32 precision islands;
- 64 attention heads and 64 KV heads;
- Q LoRA rank 1536, KV LoRA rank 512;
- QK head dim 256, no-PE dim 256, RoPE dim 0, V head dim 256;
- sparse indexer: 32 heads × 128 dim, top-k 2048, k-pool 4;
- KDA: 64 heads × 128 dim, short-conv kernel 4;
- MoE: 288 routed experts, one shared expert, top-8 per token,
  expert intermediate size 2048;
- vocabulary 154,880;
- native max position 1,048,576.

## 4. Process and rank ownership

The first distributed implementation uses one process per DGX Spark and one
GB10 device per process.

**Rank 0 / head** owns external request ingress, tokenizer/frontend integration,
request scheduling, authoritative sampler/RNG/parser state, API streaming and
commit decisions.

**Rank 1 / worker** owns no external API state. It participates in the same
ordered target/draft steps and applies commit/rollback decisions broadcast by
rank 0.

Both ranks own tensor-parallel shards of model weights and persistent GPU state.
The initial contract divides logical attention/KV/KDA heads 32/32 per rank and
sparse indexer heads 16/16 per rank. MoE is tensor-parallel, not expert-parallel:
288 experts remain a logical model-wide set while each selected expert's matrix
work is sharded according to the eventual EXL3 kernel layout.

## 5. mHC stream and TP synchronization

The decoder state is `[batch, sequence, 4, hidden]`, not a single residual
stream. mHC builds Sinkhorn-constrained mixing weights, collapses the four
streams for each attention/FFN sublayer, then updates the stream bundle around
that branch. `ExecutionPlan` therefore carries the four-stream geometry as a
first-class invariant.

The plan also exposes two semantic TP collective boundaries per layer: mixer
branch output and FFN branch output. Both are currently modeled as all-reduce
boundaries. A later fused CUDA/NCCL path may overlap communication, but it may
not change the mathematical point at which both ranks feed an equivalent branch
result into the mHC update. This separation lets us tune communication without
losing a correctness oracle.

## 6. Persistent-state ledger

Every sequence will eventually own a state ledger with these classes:

### Target KDA state

Per KDA layer, per-rank recurrent state plus short-convolution history. The
state is advanced transactionally: speculative verification may compute future
states, but only accepted target tokens become committed state.

### Target sparse-MLA state

Per sparse-MLA layer, paged target KV representation plus sparse-index/indexer
state required to reproduce later token selection. The storage target is the
validated fp8 MLA path; exact byte geometry is not guessed in P0 and will be
bound to the real checkpoint/kernel contract in P2/P3.

### Draft state

DFlash2 state is separate from target KV/state even when target embeddings/head
or tokenizer resources can be shared. Proposal state is committed only through
the same accepted-token transaction.

### Host continuation state

Rank 0 must be able to serialize enough request state to resume generation
without silently changing output semantics: committed token ids, absolute
position, sampler parameters, RNG seed/counter, stop/parser state, tool/reasoning
parser state and speculation bookkeeping. This is a later implementation item,
but it is an architectural requirement rather than an optional API feature.

## 7. Step transaction

A decode transaction is intended to have four logical phases:

1. **Propose** – both ranks advance the DFlash2 draft for up to `k` proposals.
2. **Verify** – both ranks run the target over the verification block.
3. **Decide** – rank 0 performs deterministic acceptance/sampling from identical
   target results and broadcasts the accepted-token decision.
4. **Commit** – both ranks commit exactly the accepted target/draft/KDA/KV/index
   state and discard or rewind speculative suffix state.

No rank may independently decide accepted length. A communication failure inside
steps 2–4 fails the request/step rather than producing rank-divergent state.

## 8. Artifact/checkpoint strategy

Do not convert EXL3 to a generic dense format just to make loading easy; that
would erase the product's main memory/bandwidth advantage. The next loader phase
will:

1. inventory the actual EXL3/TR3 tensor names and shard metadata;
2. define logical GLM parameter names independent of source filenames;
3. map source tensors into explicit encoded-parent views;
4. preserve packed EXL3 representation when kernels can consume it directly;
5. copy/repack only when a measured kernel contract requires a different layout;
6. verify every mapping against shape, layer kind and numeric-format metadata.

NInfer's artifact/binder/materializer separation is the reference pattern. We
should extend or specialize that pattern only after the real checkpoint mapping
shows whether its current artifact-v3 formats can describe EXL3 faithfully.
Creating a new artifact version before that evidence would be premature.

## 9. CUDA graph policy

Decode should remain graph-first. Graph captures must be keyed by *actual* step
shape, including speculative block size. The current evidence-backed DFlash2
baseline is k=7, but controlled k=5/k=4 experiments remain allowed. Each arm
needs matching capture/warmup shapes; changing `k` without changing captured
shapes is an invalid benchmark.

Prefill and decode are separate scheduling phases. Long prefill must not be
allowed to destroy output-token latency by monopolizing the same execution
window; the exact fairness mechanism will be reimplemented against this
runtime's own scheduler rather than copying vLLM scheduler patches blindly.

## 10. Failure policy

The runtime must fail closed on:

- model geometry different from the pinned GLM-5.3-Flash contract;
- incomplete/ambiguous checkpoint bindings;
- rank/world-size mismatch;
- divergent sequence transaction ids or committed positions;
- missing required CUDA/NCCL capability;
- graph/kernel support gaps for a requested shape;
- NaN/Inf detected at contract checkpoints during validation builds.

An optimization may have a deliberate slower fallback only when that fallback
is tested and explicitly surfaced. Silent fallback to a semantically different
attention, quantization or speculation path is not acceptable.
