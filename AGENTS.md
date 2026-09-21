# Engineering contract for agents

## Product target

This repository is specialized for GLM-5.3-Flash EXL3 on exactly two NVIDIA DGX
Spark / GB10 nodes with TP=2. Do not generalize a hot path unless the target
needs it.

## Correctness rules

- `include/ninfer_glm53/model_spec.hpp` + `src/model_spec.cpp` define the logical
  model topology. Do not duplicate layer lists in kernels.
- `DeploymentContract` is the source of deployment intent; hardware preflight is
  the source of observed node facts. Never treat an env value as proof that the
  hardware matches it.
- `ExecutionPlan` contains semantic TP synchronization boundaries. Fusion may
  overlap them but may not change the equivalent mathematical result.
- KDA state, sparse-MLA KV/index state, draft state and host sampler/parser/RNG
  state are distinct ownership domains.
- A speculative step commits accepted tokens transactionally on both ranks. No
  rank-local accepted-length decision is permitted.
- Never silently fall back to a different quantization, attention backend,
  speculation method or world size.

## GPU implementation rules

Before optimizing a new public operator, provide:

1. an exact shape/format contract;
2. a reference/oracle comparison;
3. eager execution coverage;
4. CUDA graph replay coverage if used in decode;
5. boundary-shape cases;
6. compute-sanitizer coverage for newly introduced synchronization/memory code;
7. a benchmark of the public operator or model path that actually consumes it.

Performance measurements must name hardware, commit, checkpoint revision,
execution mode, input shape, sample protocol and whether caches are warm/cold.

## Checkpoint rules

`parameter_schema` is the logical catalog. `checkpoint_binding` is the coverage
check. Do not infer EXL3/TR3 tensor semantics from filenames. A binder must fail
if any required logical parameter is missing, duplicated or shape/format
incompatible. Storage shapes are not a decoded trellis; do not treat a complete
binding receipt as a serving kernel.

## Deployment baseline

`configs/dgxspark_tp2.env.example` is copied from the evidence-backed local
baseline and contains no secrets. Tuning values can change through controlled
experiments, but topology invariants (two GB10 nodes, TP2, intended RoCE path)
must remain explicit.

## Scope discipline

Do not add vLLM as a runtime dependency. It is a comparison/reference baseline.
Do not add placeholder CUDA kernels merely to advance a checklist. Host contracts
and tests are preferable to fake GPU plumbing that obscures missing semantics.
