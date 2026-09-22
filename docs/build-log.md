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

On 2026-09-22 the same 24 GiB replay printed per-operator norms. Layer 3's MoE norm was `261.96`, the same value previously checked against `exl3_linear_reference`. Layer 9's stream norm was `12.626`. Layer 10's FFN norm was `205.65` (routed `90.435`, shared `122.01`) and its mHC post mean abs was `0.14076`, so the added branch is about `29` and the stream norm became `33.519`. Layer 6 had a larger shared expert (`167.42`) but a post mean abs of only `0.0406`, and its stream norm stayed `13.216`. The jump is that post coefficient times the FFN, not a second application of `routed_scaling_factor`. The partial argmax became `19845` at layer 10 and `154822` at layer 29.

## Still required on Blackwell

SM121 kernels for KDA, sparse MLA, the indexer, and MoE; an NCCL TP2 process
on the CX-7 link; paged fp8 KV and KDA state; DFlash2 on the device; then
token parity with a Blackwell server on `fixed-text-v1`. Placeholder kernels
are not part of this record.

On 2026-09-22T17:34+08:00 the worker ran the image's `Glm5NextDecoderLayer` for layer 0 only, inside a 24 GiB container, with no vLLM engine. Prompt token 13041. The KDA prefill cache metadata came from `compute_causal_conv1d_metadata` on `[0, 1]` and was synchronized before the forward. The layer completed. Its contracted hidden, after the final RMSNorm and `lm_head`, argmaxed at `50085` (logit 7.53125); token `13041` had logit 1.65625. The hand-rolled replay of the same layer argmaxed at `41313`. The mHC pre inputs matched (cosine 1, max abs 0). The attention outputs did not (official norm 1.59893, replay norm 0.340715, cosine 0.005667). Calling `chunk_kda_with_fused_gate` on the replay's own q, k, and v stayed at norm 0.340656, so the split is not the zero-state forget gate. This is a one-layer partial argmax, not the 45-layer continuation `13041,13041`. Worker MemAvailable after the run was 116394840 KiB. A later capture, taken after `causal_conv1d_fn` returned and without changing the argmax, measured the prefill conv output norm at `7.38648`. The decode `causal_conv1d_update` kernel on the same activation was norm `0.797651` and matched `silu(x * weight[:, -1])` (cosine `1.00000024`, max abs `0.000676`). The prefill output matched none of the four taps (last-tap cosine `-0.0101`). Conv bias is absent. The shipped host conv follows the update kernel.

That prefill capture used cache slot 0. In this image `NULL_BLOCK_ID` is 0, and `causal_conv1d_fn` returns without writing when the cache index equals that id, so the captured tensor was not the conv result. On 2026-09-22T17:55+08:00 the same worker, still in a 24 GiB container and without an engine, compared channels directly. Slot 0 with the default null id produced an all-zero output and left the conv state at norm 0. Slot 1, and slot 0 with the null id disabled, both matched `silu(x * weight[:, -1])` at cosine `1.00000024`, max abs `0.000676`, and matched `causal_conv1d_update` on the same activation. The shipped `causal_conv_silu` already uses that last tap for the current token. It was not changed.

On 2026-09-22T17:59+08:00 the layer-0 module was run again with state index 1. Attention output norm was `0.340656` against the replay's `0.340715` (cosine `0.99999225`). Stream cosine was `0.99999774`. Both partial argmaxes were `41313`. The earlier official argmax `50085` and attention norm `1.59893` came from the null slot.

On 2026-09-22T18:07+08:00 the same container chained the image's decoder layers 0, 1, and 2, each with its own valid KDA slot, and compared them with the hand-rolled replay. Stream cosines were `0.99999774`, `0.99999213`, and `0.99999106`. Partial argmaxes matched at `41313`, `37284`, and `7755`. After layer 2 the logit of token `13041` was `4.0625` and the logit of `154822` was `0.227539`. These are one-layer partials, not the 45-layer continuation.

Constructing layer 3 on that worker selected `FLASHINFER_MLA_SPARSE_SM120` and the `fp8_ds_mla` cache layout. The image logged `Sparse MLA impl has no dense-MHA prefill path; using the top-k MQA path only.` The unpacked module is about 3758 MiB, of which the routed trellis tensors are 2304 MiB and 1152 MiB.

On 2026-09-22T18:25+08:00 the same worker ran that layer-3 module inside a 24 GiB container, with no engine. The input was the replay's layer-3 attention branch after layers 0–2 (stream norms `1.22535`, `1.30193`, `4.2973`). The top-k buffer selected token 0, the only token, and the indexer was not run. The sparse MQA output matched the dense value projection at cosine `0.99952126` (max abs `0.0166016`); after `o_proj` the cosine was `0.99969971` (max abs `0.0429688`). The official fused EXL3 MoE on that same vector selected experts `106 51 115 157 142 167 29 76` and matched the per-expert replay at cosine `0.99999470` (max abs `5.97e-5`). Layer 3 does not explain `154822` versus the recorded server continuation `13041,13041`. No shipped weight formula was changed.

On 2026-09-22T18:33+08:00 the worker compared the image's full decoder layers 4 and 10 with the replay, still in a 24 GiB container and without an engine. Each official layer was given the replay streams from the layer before it. Layer 4 matched at cosine `1.00000632` (max abs `0.0078125`, norms `42.6739` and `42.6733`). Layer 10 matched at cosine `0.99998194` (max abs `0.078125`, norms `115.457` and `115.964`). The image's short-prefill indexer writes every causal index up to the position, so a one-token prefill selects token 0. The eager greedy remains `154822,154822`. The recorded server continuation remains `13041,13041`.

On 2026-09-22T18:45+08:00 the worker chained the image's own decoder from the raw embedding of token 13041, one layer at a time, inside a 24 GiB container and without an engine. The replay of all 45 layers was computed first. Its final RMSNorm and `lm_head` argmax was `154822` at logit `21.125`, the runner-up id was `315`, and token `13041` had logit `0.447266`. The official chain matched that replay through layer 22: cosine stayed at or above `0.99994427`. Printed partial argmaxes agreed at layer 0 (`41313`), layer 5 (`15576`), layer 10 (`19845`), layer 15 (`19845`), and layer 20 (`5556`). The run stopped when host MemAvailable fell under 40 GiB because deleted layers were not reclaimed; after the container exited, MemAvailable returned to 122657380 KiB. No shipped formula was changed.

On 2026-09-22 four host defects were fixed: in-place `causal_conv_silu` stores the raw sample rather than the SiLU output, `Store` releases a mapped shard if construction throws, the TP2/DFlash fork path closes its sockets and reaps the child (including on `_exit(1)`), and DFlash verify restores the snapshot before the committed round is stepped. `cmake --build build -j` and `ctest --test-dir build --output-on-failure` passed 11/11, including text_forward, speculative, exl3_decode, exl3_linear, and host_python_regressions. Step 0 still applies SiLU to the raw activation, and Hadamard, EXL3, MoE scale, and mHC were not changed, so this does not move the first eager argmax. The in-place history fix can change later tokens, the recorded server continuation `13041,13041` was not re-measured, and no full-model forward was run.

On 2026-09-22T18:58+08:00 the worker ran RMSNorm of embedding row 13041 through `lm_head`, inside a 16 GiB container and without an engine. The argmax was `4246` at logit `6.9375`. The next four ids were `11681` (`6.844`), `5660` (`6.625`), `34249` (`6.5`), and `1425` (`6.469`). Token `13041` had logit `1.25`, token `785` (`The`) had `1.09375`, and token `154822` had `-0.5625`. The recorded server continuation is still `13041,13041`. No shipped formula was changed. Host MemAvailable after the container exited was 122900476 KiB.

On 2026-09-22T19:25+08:00 the same worker finished that official chain through layer 44. After each layer the packed CUDA parameter storage was cleared, so host MemAvailable stayed at or above 114041236 KiB and returned to 120790284 KiB after the container exited. PyTorch CUDA allocation leveled off near 567 MiB once the first MoE layer's fused workspace existed. Cosine versus the replay stayed at or above `0.99911267` through layer 43. Layer 44 returns the contracted hidden, so its cosine of `0.99109709` is against the first replay stream rather than the four-stream mean. The printed partial argmaxes still matched, including layer 29 onward at `154822`. The official final RMSNorm and `lm_head` argmax was `154822` at logit `21`, runner-up `315`, and token `13041` at `0.470703`. The replay on the same run was `154822` at `21.125`, runner-up `315`, token `13041` at `0.462891`. The recorded server continuation remains `13041,13041`. No shipped formula was changed.

On 2026-09-22T19:33+08:00 the same replay printed an argmax after every layer. None is `13041`. The sequence is layer 0 `41313`, layer 1 `37284`, layer 2 `7755`, layers 3–5 `15576`, layers 6–7 `1000`, layers 8–9 `15576`, layers 10–17 `19845`, layers 18–22 and 26–28 `5556`, layers 23–24 `33182`, layer 25 `2757`, and layers 29–44 `154822`. The final top three logits are `154822` at `21.125`, `315` at `6.65625`, and `304` at `6.3125`. Token `13041` is rank `67550` at logit `0.447266`. Token `785` (`The`) is rank `121210` at logit `-0.539062`. Masking the special ids `154820`–`154824` and `154826`–`154829` leaves `315` on top. The recorded server pair, `Hi` over `The`, is not this distribution with those ids removed. No shipped formula was changed. Host MemAvailable at the end of that container was 116956760 KiB.

On 2026-09-22T19:40+08:00 the same worker projected embedding row `13041` through the embedding table. After RMSNorm the top two ids were `13041` at logit `66` and `20971` at `48`. Token `785` was rank `8467` at logit `3.125`. That tied readout is a different peak from the server logprobs, where `Hi` leads `The` by about `3.44` nats. The serving `.env` sets `ENFORCE_EAGER=0`, `EXL3_FAT_GROUPED=1`, and `EXL3_OVERLAY_HOST` to the cooperative MoE adapter. Its runtime selects the cooperative kernel for 1 to 32 rows when the local MoE intermediate is `1024` (the TP=2 shard). The adapter's own CUDA integration note says that kernel is not bit-exact with stock and checks a relative tolerance around `1e-3`. A 1-token prefill under the fused-MoE row cap returns before the grouped-fat path. The eager TP=1 image forward remains `154822`. No shipped formula was changed.

On 2026-09-22T19:46+08:00 the worker scored layer 0 of that replay with the real `lm_head`. The embedding argmax is `4246`. The mHC pre branch stays `4246`. Mixing a zero attention branch stays `4246` at logit `6.9375`. Mixing `o_proj(v_proj(x))` before the convolution moves it to `6004`. Mixing the convolved value moves it to `60280`. Mixing the full KDA branch moves it to `25566`. The dense FFN then leaves layer 0 at `41313`, matching the earlier official layer-0 argmax. Token `13041` is rank `29209` on the embedding and rank `7878` after layer 0. Token `785` is never the runner-up at these stops. The first stream-level move away from the embedding readout is the layer-0 value projection. The recorded server continuation remains `13041,13041`. No shipped formula was changed.

On 2026-09-22T19:51+08:00 the same worker continued that replay until the `lm_head` argmax became `154822`. Layer 28 was `5556` at logit `8`. Layer 29 was the first `154822`, at logit `8`, runner-up `5556`, with token `13041` at `0.777344`. Host MemAvailable stayed above 111854724 KiB. On the layer-3 MoE input, the eight routed experts were `106, 167, 51, 115, 76, 142, 157, 28`. The per-expert replay matched stock fused MoE at cosine `0.99999964` (max abs `0.0313854`, relative L2 `0.00178229`). Each TP=2 shard has local intermediate `1024`, which is the shape the server cooperative kernel accepts. Rank 0 cooperative versus stock was cosine `1.00000036` (max abs `0.03125`, relative L2 `0.00159555`). Rank 1 was cosine `1.00000024` (max abs `0.03125`, relative L2 `0.00190456`). The sum of the stock shards matched the full tensor at cosine `0.99999827`, and the sum of the cooperative shards matched it at cosine `0.99999845`. That is bf16 noise, not a formula change. No shipped formula was changed.

On 2026-09-22T19:58+08:00 the worker split layer 29 before the MoE. The stream entering the layer argmaxes at `5556` (logit `8`). The mHC pre branch, a zero attention mix, the convolved value mix, and the full KDA mix all stay at `5556`. Mixing `o_proj` of the value projection before the convolution does argmax at `154822` (logit `7.46875`), and the convolution brings that mix back to `5556`. The full KDA branch leaves token `13041` at logit `1.02344`. The previously measured full layer, after the matched MoE, is the first `154822`. The attention, KDA, and mHC steps are not that flip. No shipped formula was changed.

On 2026-09-22T20:05+08:00 the worker took that layer-29 FFN branch, still argmax `5556` at logit `8.125`, and applied the stock EXL3 expert kernel. The shared-expert gate weight is shaped `(2048, 4096)`. The routed experts were `129, 2, 274, 244, 50, 250, 105, 46`, with scaled weights summing to `2.5`. The fused kernel matched the per-expert replay at cosine `0.99999821` (max abs `0.00805831`, relative L2 `0.00189493`). That MoE vector itself argmaxes at `28451`, logit `8.375`. Mixing it through mHC post moves the stream to `154822`, logit `8`, runner-up `5556`, token `13041` at `0.785156`. The shipped router uses the same unbiased-sigmoid weights, `2.5` scale, SwiGLU clamp, and unscaled shared expert. The stock apply is the residual being mixed, so the shipped MoE was not given a different formula. No shipped formula was changed.

On 2026-09-22T20:08+08:00 the worker compared that layer-29 mix with the shipped host rule `out[j] = post[j] * branch + sum_i comb[i, j] * residual[i]`. Against tilelang `mhc_post` the cosine was `1.00000072` and the max abs difference was `0.034955`. The post coefficients were `0.0219453, 1.92191, 0.473215, 0.0464431`. Both mixes argmax at `154822`, logit `8`, runner-up `5556`. Token `13041` was `0.777344` on the tilelang mix and `0.78125` on the host mix. The recorded server continuation remains `13041,13041`. No shipped formula was changed.

On 2026-09-22T20:34+08:00 the worker ran the image's layer-0 `Glm5NextDecoderLayer` twice inside a 24 GiB container, with no engine. A 1-token prefill at conv slot 1 argmaxed at `41313`, logit `6.84375`, runner-up `11681`, token `13041` at `2.15625`, norm `1.22515`. The same layer as a 1-token decode at slot 1 argmaxed at `41313` with the same runner-up, cosine `0.99999881` against the prefill, max abs `0.000244141`. Decode at slot 0, the null conv block, argmaxed at `5923`, logit `8.5`, runner-up `10876`, token `13041` at `-3.375`, norm `19.1847`, cosine `0.04349913` against the prefill. Slot 0 is not the `13041` distribution. Host MemAvailable stayed at or above 114944180 KiB and was 122238744 KiB after the container exited. The stopped `glm53-exl3-worker` filesystem differs from this image's `exl3.py` only by the six-line cooperative installer; its `glm53_ablit.py` and the three KV-cache Python files match the image. The worker log at the seeded `HiHi` request (container clock 09-21 19:17:30) records a top-k log-softmax compilation and no eager cooperative row-count line, so that request's MoE ran from a captured graph. No shipped formula was changed.

On 2026-09-22T20:44+08:00 the worker captured a CUDA graph around that same layer-0 module. Replaying token `13041` matched the eager hidden state at cosine `1.00000572`, max abs `0`, argmax `41313`. Copying token `785` into the graph input and replaying matched the eager forward of token `785` at max abs `0`, argmax `20469`, and that replay differed from the token-`13041` eager state at cosine `0.31527132`. The graph followed the input token. Host MemAvailable after the container exited was 122455832 KiB.

On 2026-09-22T20:48+08:00 the worker split layer-0 KDA into two head shards. The sum of the shards matched the full projection at cosine `0.99999881`, max abs `6.10352e-5`, and the mixed stream stayed at argmax `25566`. Rank 0 alone, with no allreduce, matched the full projection at cosine `0.545151`, and its mixed stream argmaxed at `23671`. Continuing that rank-0-only walk through all 45 layers, with MLA left on the full projection, ended at argmax `48130`, logit `4.96875`, runner-up `1637`, token `13041` at `1.70312`. It did not emit `13041`. Host MemAvailable stayed above 116929800 KiB.

On 2026-09-22T20:51+08:00 the worker ran layer-3 sparse attention on the replay's layer-3 branch. `cache_dtype=auto` already selected `fp8_ds_mla`. Forcing `fp8` produced the same attention vector, max abs `0`. Both matched the dense value replay at cosine `0.99969971`, max abs `0.0429688`. The mHC mix of that attention stayed at argmax `7755` for the replay, the auto cache, and fp8. No shipped formula was changed.

On 2026-09-22T21:05+08:00 the shipped two-rank path was traced in `src/text_forward.cpp`. With a rank link, `apply_rows` computes half the output rows on each rank and `gather_rows` exchanges those halves. `gemv_bf16` reads weight row `row * cols`, so the exchange is the full GEMV, not a missing reduction. The same split covers `linear_exl3` when the row bounds stay on 128-wide tiles. Layer-0 KDA shard sum already matched that full projection. The earlier paired greedy on this path is `154822,154822`, the same ids as the single-rank eager stack. The recorded server continuation remains `13041,13041`. No shipped formula was changed.

On 2026-09-22T21:03+08:00 the image's block pool was read. It builds blocks `0 .. num_gpu_blocks-1` and pops the first free block as the null placeholder, with the comment that this placeholder is block id 0. That block stays marked used, so a request's first conv slot is not 0. The earlier decode at slot 0, argmax `5923`, is not the serving slot. The serving slot is the slot-1 path whose 45-layer argmax is `154822`. No shipped formula was changed.

On 2026-09-22T21:07+08:00 the worker captured a CUDA graph around the cooperative MoE kernel on layer 3, rank 0, one token. Replaying the capture input matched the eager cooperative output at max abs `0`. Replaying a second input matched that input's eager output at max abs `0` and differed from the first eager output at cosine `0.082025`. The ctypes launch is inside the graph and follows the hidden state. Host MemAvailable after the container exited was 122701956 KiB.

On 2026-09-22T21:10+08:00 the same worker replayed all 45 layers with KDA attention replaced by zeros, leaving sparse MLA and the MoE in place. The final argmax was `565` at logit `11.375`, runner-up `1`, token `13041` at `3.5625`. That is not the recorded continuation. The decode kernel's `state_idx <= 0` early return is not this serving path: the block pool keeps block 0 as the null placeholder. No shipped formula was changed.

On 2026-09-22T22:40+08:00 the worker tokenized the checkpoint and reran layer 0. `encode("Hi")`, with and without special tokens, is `[13041]`. The chat template for user text `Hi` is 13 ids beginning `154822, 154824` and is not a 1-token prompt. The text config has `qk_rope_head_dim=0` and `mla_nope=true`. Layer 0 on token `13041` with zero conv history and zero KDA state gave the same hidden state at position 0 and position 3: cosine `1.00000024`, max abs `0`, argmax `41313` logit `6.84375` second `11681`. The shipped step copies that one embedding row into four streams and starts conv history, KDA state, and the MLA cache at zero. That packing does not produce the server pair `13041` ahead of `785`. No shipped formula was changed.

On 2026-09-22T22:34+08:00 the worker repeated that 1-token decode and applied `model.language_model.norm.weight` before `lm_head`. The final hidden shape was `(1, 4096)`. The learned norm weight had L2 `91.6188` and mean `1.42109`. With that weight, L44 argmax was `154822` logit `22.5` second `315`, token `13041` at `0.455078`. The same hidden with a unit norm weight argmaxed at `154822` logit `16.625` second `68669`, token `13041` at `-0.5`. Neither readout was `13041`, so the shipped generate entry was left unchanged. Host MemAvailable was 122853404 KiB before the container and 120648932 KiB after it exited.

On 2026-09-22T22:25+08:00 the worker walked all 45 official layers as a 1-token decode. `num_decodes=1`, `num_prefills=0`, KDA state slot 1, MLA slot 64, block size 64, and one selected index. No engine. L00 argmax was `41313` logit `6.84375` second `11681`, token `13041` at `2.15625`. L02 was `7755`, L03 was `15576`, L29 was `154822` logit `8.4375` second `5556`, and L44 was `154822` logit `22.5` second `315`, token `13041` at `0.458984`. The decode classification does not emit `13041`. Host MemAvailable was 122838440 KiB before the container and 120963124 KiB after it exited. No shipped formula was changed.

On 2026-09-22T22:16+08:00 the worker walked the same 45 official layers again and read two matrices. The embedding row of token `13041` and the `lm_head` row of that token have cosine `0.01235098`. Unit RMSNorm of that embedding row, without the learned final norm weight, scores `34249` at `4.705` through `lm_head` (`13041` at `0.9173`, `785` at `1.061`). The same row through the embedding matrix scores `13041` at `46.49` and `20971` at `33.71`, with `785` at `2.139`. After layer 44 the contracted hidden, again with unit RMSNorm and no learned norm weight, scores `154822` at `15.66` through `lm_head` (`315` at `3.903`, `13041` at `-0.5231`, `785` at `-1.155`) and `154822` at `3.58` through the embedding matrix (`13041` at `1.334`, `785` at `-0.2181`). Neither matrix puts `13041` first and `785` second. Host MemAvailable was 122861624 KiB before the container and 121848924 KiB after it exited. No shipped formula was changed.

On 2026-09-22T22:06+08:00 the worker walked all 45 official layers in a 24 GiB container with no engine. Every BF16 column-parallel output kept only its first half, and every BF16 row-parallel weight dropped its second input half. Replicated `f_a` and `g_a` stayed whole. Routed EXL3 experts stayed at `intermediate_local=2048` because the packed trellis was not cut. Prompt token `13041`. Readouts were L00 `7924` logit `6.8125` second `23671`, L02 `7755` logit `7.46875`, L03 through L20 `15576` (L20 logit `10.125`, second `20971`, token `13041` at `5.875`), L25 `97562`, L29 `89448`, L35 `11767`, L40 `19820`, and the final argmax `1115` logit `6.40625` second `76431`, token `13041` at `0.277344`. It did not emit `13041`. Host MemAvailable was 122833104 KiB before the container and 121061672 KiB after it exited. No shipped formula was changed.

On 2026-09-22T21:51+08:00 the worker ran one layer-3 sparse MLA forward inside a 24 GiB container, fp8 KV, no engine, on the embedding row of token `13041`. `kv_cache_dtype` was `fp8_ds_mla` and the configured kernel block stayed 64. A page of 64 with `seq_len=1` and one selected index produced an attention branch of norm `32.3093`. Reading that branch with the final RMSNorm and `lm_head` argmaxed at `29299`, logit `8.66572`, then `35489` at `7.56748` and `98486` at `6.88892`. Token `13041` was `2.327` and token `154822` was `0.9104`. The same query with `seq_len=8` and eight selected indices had norm `11.1409`, cosine `0.55194652` and max abs `1.75098` against the one-index branch, and its readout argmax was `72949`. It was not `13041`. A cache page whose second dimension was 3584 was rejected by the SM120 kernel: trtllm-gen accepts kernel block 32 or 64, and the paged entry refused `num_tokens=1`. The server log's 3584 is the scheduler page that this kernel splits into those 64-token blocks. Host MemAvailable was 122472176 KiB before the container and 122796236 KiB after it exited. No shipped formula was changed.

On 2026-09-22T21:44+08:00 the stopped `glm53-exl3-worker` log, 37052 bytes and 221 lines, was read on `gx10-23ec`. The process joins TP2 at `192.168.100.10:29521` as rank 1 and reports TP rank 1, EP rank 1, `experts_local=288`, `intermediate_local=1024`. KV cache is fp8. Attention is `FLASHINFER_MLA_SPARSE_SM120` with `fp8_ds_mla`, and the log says the sparse MLA implementation has no dense-MHA prefill path and uses the top-k MQA path only. The attention block is padded to 3584 tokens so the attention page matches the mamba page. Mamba cache mode is `align` because prefix caching is enabled. Breakable CUDA graph is enabled. Weight load took 109.93 seconds, the language model took 82.06 GiB, and the checkpoint size line says 163.58 GiB. Graph capture finished at container clock 09-21 19:15:32. Later inference lines include eager cooperative MoE row counts 24, 56, 120, 128, 3520, 64, 7104, 1408, 71, 62, 54, 47, and 48. The seeded `HiHi` line at 09-21 19:17:30 is only `_topk_log_softmax_kernel`. In this image, `vllm/models/glm5next/nvidia/kda.py` `forward` calls `_forward` directly; its comment says routing through `torch.ops.vllm.kda_attention` would capture the host-branching prefill under a piecewise CUDA graph. The same worker's DFlash2 file `model.safetensors` at snapshot `dc77ff1c` is 2342169800 bytes and 81 tensors, including `fc.weight`, `hidden_norm.weight`, five decoder layers, and `candidate_selector` codebooks. It has no `embed_tokens` and no `lm_head`. No shipped formula was changed, and the shipped greedy was not re-run. The recorded server continuation remains `13041,13041`. The eager stack's final argmax remains `154822`.
