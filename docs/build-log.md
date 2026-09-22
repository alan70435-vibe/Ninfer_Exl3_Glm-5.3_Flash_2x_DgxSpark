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

## Launch refused, measured 2026-09-22T14:25:17+08:00

Head `gx10-5749` had an active wayland session. MemAvailable was 120881568 KiB. `glm53-exl3-head` was Exited (255). Worker `gx10-23ec` answered ping and was already running `glm53-exl3-worker` (`VLLM::Worker_TP` pid 3737500, 106226 MiB); worker MemAvailable was 6.1 GiB. `ninfer-glm53-generate` was not started for greedy, TP=2, or DFlash2. No new token ids.

The existing host build's CTest was 11/11. That suite is not a GB10 continuation.

## Short greedy on the worker, measured 2026-09-22T14:33:40+08:00

After `glm53-exl3-worker` was stopped, `ninfer-glm53-generate` ran on `gx10-23ec` (NVIDIA GB10 12.1, `GPU-d5d58447-ec10-0155-e20a-cf12158797b9`), not on the head desktop. Prompt `13041`, greedy, `new_tokens=2`, revision `25a44fdbf16862a46b7cc9921142c6c81350af2f`. Both repeats returned `token_ids=154822,154822`, `committed_length=2`, `world_size=1`. The first-token logits were `154822=20.9312` and `315=6.65862`. The recorded external continuation remains `13041,13041`. These two repeats agree with each other and do not match that continuation.

## Kernel replay on the worker, measured 2026-09-22T15:38+08:00

The same worker, still not the head desktop, ran a one-token replay inside `glm53-local:maintenance-20260918` (the image of the stopped `glm53-exl3-worker`). The replay called that image's tilelang mHC, causal convolution, fused recurrent KDA, and EXL3 `execute_exl3_linear`, with bf16 GEMMs for the dense projections. It did not start the vLLM server, TP=2, fp8 KV, or speculative decoding. Prompt token `13041`, revision `25a44fdbf16862a46b7cc9921142c6c81350af2f`. After layer 44 the lm_head argmax was `154822` at logit `21.0`, and the runner-up was `315` at `6.65625`. Given the eager forward's own q, k, v, and beta, the zero-state KDA core at layers 0, 1, 2, and 4 matched `beta * v * dot(q_scaled, k)` with max abs below `1e-8`. This agrees with the eager greedy logits above and does not match the recorded server continuation `13041,13041`.

A later pass on the same worker, still inside a 24 GiB container and without loading the vLLM engine, compared the server's 1-token kernels with that replay. Chunk KDA and fused recurrent KDA agreed at cosine `0.999998` (max abs `0.000122`). The 1-token fused mHC kernel and separate post-then-pre agreed at cosine `0.99999988` (max abs `0.000488`). Rounding each MLA latent through fp8 e4m3 before the value projection left the layer-44 argmax at `154822` with logit `21.125` and runner-up `315` at `6.65625`. The production server that recorded `13041,13041` had `SPEC_METHOD=dflash` and `TP=2`. Reloading that engine on one Spark allocated until about 1.5 GiB remained and CUDA reported out of memory; the container was stopped and worker MemAvailable returned to about 115 GiB.

A single `Glm5NextLinearAttention` for layer 0, built from that vLLM image with tensor parallel initialized and no engine load, accepted the checkpoint tensors through its own weight loaders. Every loaded KDA matrix matched the raw safetensors value with max abs `0`, including `in_proj_qkvbfg_a`, the three conv weights, and `o_proj`. Worker MemAvailable stayed near 119 GiB.

## Still required on Blackwell

SM121 kernels for KDA, sparse MLA, the indexer, and MoE; an NCCL TP2 process
on the CX-7 link; paged fp8 KV and KDA state; DFlash2 on the device; then
token parity with a Blackwell server on `fixed-text-v1`. Placeholder kernels
are not part of this record.
