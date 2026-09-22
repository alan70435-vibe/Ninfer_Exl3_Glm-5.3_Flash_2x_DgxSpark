# Build record

This file is the construction record for the EXL3 line. The paired NVFP4
runtime keeps its own copy in `Ninfer_Nvfp4_Glm-5.3_Flash_2x_DgxSpark`.
The two checkpoints stay separate. Neither binary is the Blackwell runtime yet.

## Hardware destination

The product runs on two NVIDIA DGX Spark nodes. Each GPU is a GB10, Blackwell,
compute capability 12.1 (`sm_121`), aarch64, with 121 GiB of memory shared by
the CPU and the GPU.

Observed link, 2026-09-22:

| Role | Address | CX-7 | RDMA device | RoCE v2 GID |
| --- | --- | --- | --- | --- |
| Head `gx10-5749` | 192.168.100.10/24 | `enp1s0f0np0` | `rocep1s0f0` | index 3 |
| Worker `gx10-23ec` | 192.168.100.11/24 | `enp1s0f1np1` | `rocep1s0f1` | index 4 |

Worker GID index 3 is empty. The head-facing worker port is `enp1s0f1np1`, not
`enp1s0f0np0`. `ninfer-glm53-generate` is still a host process. It prints
`gpu=NVIDIA GB10` from the contract, and it does not launch a kernel or use
the second Spark.

On 2026-09-22 a dual-node EXL3 vLLM server filled the unified memory from
about 06:25 until a hard reboot at 13:58. Do not start another full-memory
forward while the desktop is in use.

## Pins

| Piece | Pin |
| --- | --- |
| EXL3 target | MiaAI TR3 `25a44fdbf16862a46b7cc9921142c6c81350af2f` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2` `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Fixed prompt | `fixed-text-v1`, text `Hi`, token id 13041 |
| Official NVFP4 | recorded in the NVFP4 repository, not loaded here |

## What was built, in order

1. Host contracts: 45-layer GLM-5.3-Flash spec, dense layers 0–2, MoE layers
   3–44, KDA versus sparse MLA every four layers from layer 3, four-stream
   mHC, and a two-node TP2 deployment contract.
2. CPU metadata binding for the pinned EXL3 target (150226 tensors) and the
   DFlash2 draft (81 tensors), including trellis shapes and the MCG multiplier.
3. A separate NVFP4 repository, so this tree does not absorb the other quant.
4. EXL3 K4 unpack of the overlapping 16-bit trellis state, checked against
   ExLlamaV3 `unpack_trellis_kernel<4>` and a dense Sylvester H128 oracle.
   `exl3_linear_reference` is `x D_suh H128 Q H128 D_svh`.
5. DFlash2 temperature-0 commit: a partial accept keeps `target[matched]`, and
   a full accept keeps the bonus. The helper writes two local records. It does
   not run NCCL and it does not roll KDA or KV state back.
6. `ninfer-glm53-generate`: one-token-prompt CPU eager forward for greedy,
   local two-process TP2, and DFlash2 k=7. Routed experts use the trellis.
   Draft GEMVs are full-width on both local ranks. Target linears are
   row-sharded over a socket.
7. Host preflight and scanner from PR #40, merged as `2ac1d6f`: empty
   safetensors inventories fail, shard paths must stay inside the checkpoint
   root, and RoCE GID checks compare netdev, RoCE v2 type, and IPv4.

`main` after that merge is `2ac1d6f`. The host suite at that commit was
11/11 passed: the C++ contract tests plus `host_python_regressions`.

## Numeric checks that held

These were compared with the live ExLlama and vLLM sources on the EXL3
container, not with a second copy of this repository:

- one expert gate GEMV matched `exl3_gemm` and `reconstruct` (cosine 1,
  max abs about 7e-4, fp16 rounding);
- a zero-state KDA step matched `chunk_kda_with_fused_gate`;
- mHC pre matched the tilelang kernel on the layer-0 embedding of token 13041;
- layers 0–2 of the CPU forward matched an independent transcription of the
  same formulas.

## Numeric check that did not hold

The dual-Spark vLLM EXL3 server, temperature 0, seed 1, prompt token 13041,
`max_tokens=2`, returned `13041,13041` (`HiHi`) on two repeats. An unseeded
call returned `13041,20971`. The CPU eager forward of the same prompt returns
`154822,154822` (`[gMASK]`). On the full stack the logit for 13041 was about
0.45 and the logit for 154822 was about 20.9. Self-agreement with a second
CPU transcription is not that external result. CUDA kernels are not started
until this gap is closed or explicitly bounded.

## Upstream looked at on 2026-09-22

- `Neroued/ninfer` `9e163ee` (2026-09-18) routes Q4/Q5 GEMMs. It has no
  GLM-5.3 EXL3 or NVFP4 path. Branches: `master`, `dev`.
- `incoai/splash` `e8fffde` (2026-09-21) is Mac chat and Qwen Q4. Its open
  branches are KV-format and community-feedback work, not this model.
- `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` `c1b7d4c` (2026-09-22) adds
  launcher behavior: a default 14 GiB InstantTensor KV reservation and
  preservation of caller arguments on TP3/TP4. The KDA BF16 large-M prefill
  was already in `775a58b`. None of these commits replace the eager math.

## Still required on Blackwell

SM121 kernels for KDA, sparse MLA, the indexer, and MoE; an NCCL TP2 process
on the CX-7 link; paged fp8 KV and KDA state; DFlash2 on the device; then
token parity with a Blackwell server on `fixed-text-v1`. Placeholder kernels
are not part of this record.
