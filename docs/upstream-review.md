# Upstream review and adaptation decisions

## Reviewed snapshots

- NInfer upstream: `Neroued/ninfer` at
  `9e163eee4b8acec21ab0ac765107b6a3f287b217` (2026-09-18).
- Local deployment evidence: `alan70435/Dgx_Spark_Best_Settings` at
  `04a8eb2e6192ebd3e8f1ff243a0806a82e22833e`.
- Existing GLM DGX Spark serving recipe recorded locally at
  `0437c09c6f066ff5ee1a4b6ecd51b4b8c569a399`.

These pins are evidence/provenance references. This P0 repository does not
vendor Ninfer or the vLLM recipe.

## NInfer concepts worth carrying forward

### Explicit core/runtime/product boundaries

Upstream separates low-level device/state primitives, model execution, runtime
contracts and product surfaces. Keep this principle. In this project, model
math, two-rank deployment and execution scheduling are already separate host
contracts before GPU code exists.

### Artifact → binder → materializer

A generic byte container should not know model math, and kernels should not parse
checkpoint filenames. Keep the separation:

`checkpoint source -> logical binding -> encoded view -> materializer -> op`

The exact EXL3 representation must be learned from the real checkpoint before
choosing whether to extend upstream artifact formats.

### Graph-first decode

NInfer's explicit decode graph ownership is aligned with this product. Preserve
stable addresses and preallocated decode resources. For DFlash, graph identity
must include proposal shape; a k change is not merely a sampler flag.

### Paged state and state-machine thinking

Upstream's paged KV/cache and explicit linear-attention state are useful design
references. GLM needs both classes at once: KDA recurrent state on 34 layers and
sparse-MLA KV/index state on 11 layers.

### Operator-level correctness + benchmark surfaces

Retain the pattern of testing public ops against a reference while measuring the
same public op, not a private kernel micro-path that the model never calls.

## What cannot be copied as-is

### Single-GPU assumptions

A two-node step has distributed failure and commit semantics. CUDA success on
one process is insufficient; sequence position, accepted length and persistent
state must stay identical across ranks.

### Qwen3.5 model topology

GLM's KDA/sparse-MLA alternation, indexer and MoE dimensions require a dedicated
model program/binder. Renaming Qwen parameters is not a correct port.

### Existing tensor formats

Upstream currently documents several native formats/layouts, but the target here
is EXL3/TR3 4 bpw. We must not silently decode to another format and still call
the runtime EXL3. First prove the actual source encoding and consumer geometry.

### Hardware assumptions

Upstream performance work is heavily specialized to RTX 5090 / its exact kernel
geometries. DGX Spark GB10 is SM121 and has a very different memory/system/fabric
context. Reuse mechanisms only after compiling and benchmarking them on GB10;
never import route thresholds as performance facts.

## First implementation decision

P0 therefore does **not** start by copying hundreds of upstream CUDA files. It
establishes a narrow GLM/TP2 contract that future imported or rewritten kernels
must satisfy. That makes subsequent work reviewable: each kernel/loader/NCCL
change has an explicit target shape and state/collective boundary rather than an
implicit set of assumptions inherited from another model/hardware product.
