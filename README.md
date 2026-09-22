# NInfer × GLM-5.3-Flash EXL3 × 2 DGX Spark

A purpose-built inference runtime for **GLM-5.3-Flash EXL3** on exactly
**2 × NVIDIA DGX Spark / GB10**, using tensor parallelism over the dedicated
ConnectX-7 200 Gb/s RoCE link.

This repository is intentionally *not* a generic inference framework. The goal
is to preserve the parts of NInfer that are valuable for a narrow, highly tuned
runtime—explicit execution/state contracts, ahead-of-time kernel specialization,
CUDA-graph-friendly decode, compact artifact binding—while making the product
changes required by GLM-5.3-Flash and two-node TP2.

## Status

P0 contracts and **P1 checkpoint binding** are in place. The pinned EXL3
target (150226 tensors) and DFlash2 draft (81 tensors) bind completely on CPU
metadata, including EXL3 storage shapes and the MCG multiplier. See
[`docs/checkpoint-binding.md`](docs/checkpoint-binding.md).

`ninfer-glm53-generate` runs a CPU eager text forward. Routed experts use the
EXL3 trellis with the 128-point Hadamard reconstruct. CUDA kernels, NCCL and
the serving API are not in place.

NVFP4 runtime work continues in [Ninfer_Nvfp4_Glm-5.3_Flash_2x_DgxSpark](https://github.com/alan70435/Ninfer_Nvfp4_Glm-5.3_Flash_2x_DgxSpark). This repository keeps the EXL3 binding line.

Implemented now:

- a compile-time GLM-5.3-Flash mathematical contract (45 layers);
- exact KDA vs sparse-MLA layer topology and dense vs sparse-MoE topology;
- four-stream mHC geometry and FP32 precision-island requirements;
- key MLA/indexer/MoE geometry used by later binders and kernels;
- a typed two-DGX-Spark deployment contract parsed from `.env`;
- drift validation against the evidence-backed deployment in
  `Dgx_Spark_Best_Settings`;
- a TP2 execution-plan generator with per-rank head geometry and explicit
  branch-output all-reduce boundaries around mHC updates;
- persistent-state classification for KDA versus sparse MLA/indexer layers;
- DFlash2 target/draft geometry in the plan;
- read-only DGX Spark node preflight;
- local checkpoint config/shard manifest scanner;
- logical parameter catalog for the target, next-token module, vision tower, and DFlash2 draft;
- strict checkpoint binder with safetensors header and MCG-multiplier checks;
- host build and CTest coverage plus GitHub Actions host CI.

Not implemented yet—and deliberately **not faked with placeholder kernels**:

- decoded EXL3/TR3 materialization (storage shapes are bound; the trellis is not decoded);
- CUDA/SM121 kernels for GLM KDA, sparse MLA/indexer and MoE;
- NCCL/RoCE TP2 process runtime;
- target/draft GPU execution and DFlash2 verification;
- paged target KV + KDA recurrent state allocation;
- request scheduler, RNG/sampler/parser resume state and OpenAI-compatible API;
- real checkpoint parity and DGX Spark performance validation.

See [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md) for the
order and exit criteria.

## Validated deployment baseline

The checked-in contract mirrors the sanitized 2026-09-18 deployment snapshot in
`alan70435/Dgx_Spark_Best_Settings`:

| Contract | Baseline |
|---|---|
| Target | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` |
| Target revision | `25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| Nodes / TP | 2 / 2 |
| Fabric | ConnectX-7, 200 Gb/s RoCEv2, GID index 3 |
| Quantization | EXL3 4 bpw |
| Target KV | fp8 (`fp8_ds_mla` is the later kernel/storage target) |
| Speculation | DFlash2, `k=7`, draft TP=2 |
| Context | 1,000,000 tokens (model native maximum 1,048,576) |
| In-flight | 1 sequence |
| Batched-token budget | 7,168 |
| GPU memory fraction | 0.85 |

Those values are a reproducible starting point, not eternal tuning constants.
The validator treats product invariants (for example TP=2) as errors and
controlled tuning drift (for example a future DFlash k sweep) as warnings.

## Build and inspect the plan

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/ninfer-glm53-plan configs/dgxspark_tp2.env.example
```

The plan command validates both contracts and prints all 45 layers with their
mixer, persistent-state class, FFN class and TP synchronization boundaries.
This is the host oracle that future GPU code must continue to satisfy.

Bind a real checkpoint after the build. The command reads config and
safetensors headers, checks every tensor against the logical catalog, and
writes a receipt:

```bash
./build/ninfer-glm53-bind /path/to/GLM-5.3-Flash-EXL3 -o receipt.json
./build/ninfer-glm53-bind /path/to/GLM-5.3-Flash-DFlash2 -o dflash-receipt.json
```

## DGX Spark preflight

Run independently on each node before any CUDA/NCCL bring-up:

```bash
./scripts/preflight_dgxspark.sh head configs/dgxspark_tp2.env.example
./scripts/preflight_dgxspark.sh worker configs/dgxspark_tp2.env.example
```

The script only reads system state. It checks ARM64, GB10, local CX-7 interface
and address, 200 Gb/s link, the configured RoCE GID and basic memory/toolchain
visibility. It never changes clocks, networking or drivers.

## Inventory a real checkpoint

Before writing tensor-name mapping code, capture exactly what the local
checkpoint contains:

```bash
python3 scripts/checkpoint_manifest.py /path/to/GLM-5.3-Flash-EXL3 \
  --output results/local/checkpoint-manifest.json
```

Add `--hash-shards` only when a full byte-level receipt is useful; large EXL3
shards can make that operation expensive.

## Design rule

Every optimization must preserve a separately testable mathematical/state
contract. Fast code is not accepted solely because it runs: checkpoint binding,
state ownership, collectives, graph replay and recovery semantics must be
observable and testable before performance claims are attached to them.

The upstream NInfer snapshot reviewed for this project is pinned in
[`docs/upstream-review.md`](docs/upstream-review.md). No upstream source is
vendored in this initial commit.
