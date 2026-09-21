# Checkpoint binding

P1 maps every stored tensor in the pinned GLM-5.3-Flash EXL3 target and the
DFlash2 draft onto a logical id. The mapping is host-only: it reads
`config.json`, the safetensors index, and safetensors headers. It does not
load weight payloads except the 4-byte MCG multiplier on routed experts.

`include/ninfer_glm53/parameter_schema.hpp` is the catalog. Checkpoint key
strings stay inside that catalog. Later kernels should consume logical ids,
not Hugging Face names.

## What was measured

Headers for all 120 target shards were read, and every routed-expert `.mcg`
payload was checked. The draft is one safetensors file. Both catalogs bind
with no missing names, no extra names, and no dtype or shape mismatches.

| Checkpoint | Identity | Tensors | Catalog sha256 |
|---|---|---:|---|
| `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` @ `25a44fdbf16862a46b7cc9921142c6c81350af2f` | config `4f5341e048984459471bfb9c894e6bf87e69b9c67402672af901631d1349f265`, index `2f64d21c67c90bbafeb36c4e9b2f06f54063ed439e9f7cf95962d425a1d8515d` | 150226 | `f3a76ffad53259bbde06e7bbf43a138d0e5af60344cb5a1d8305058572328a90` |
| `incoai/GLM-5.3-Flash-DFlash2` @ `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` | config `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573` | 81 | `ccad9b633dc4090c50c6d1667778402cc634eb08bd7d87ac22e2abe209626b44` |

Receipts: [`receipts/glm53-flash-exl3-tr3-4bpw.json`](receipts/glm53-flash-exl3-tr3-4bpw.json),
[`receipts/glm53-flash-dflash2.json`](receipts/glm53-flash-dflash2.json).

The target bytes are the deployment pin. `MIRROR.json` records the Brandon
upstream snapshot `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`. The
materialization receipt records BF16 source `zai-org/GLM-5.3-Flash-BF16` @
`a6c167b62691b2bac901344b65cb651a70f53e43`. Config and index hashes match that
receipt. DFlash revision matches `DFLASH_REVISION` in
`configs/dgxspark_tp2.env.example`.

## Target layout

The released namespace is `model.language_model.*` / `model.visual.*`, with
mHC tensors named `hc_attn_*` and `hc_ffn_*` and KDA `A_log` directly on
`self_attn`.

| Region | Layers | Storage |
|---|---|---|
| Embed, final norm, `lm_head` | — | BF16 |
| Decoder mHC | 0–44 | FP32 base/scale, BF16 fn |
| KDA mixer | 34 layers (0–2, then every layer that is not sparse MLA) | BF16, except FP32 `A_log` and `dt_bias` |
| Sparse MLA + indexer | 3, 7, …, 43 | BF16 |
| Dense FFN | 0–2 | BF16 |
| Routed experts | 3–44 and the next-token layer | EXL3/TR3 MCG, 4 tensors each |
| Shared expert, router | same MoE layers | BF16 router weight, FP32 score-correction bias |
| Next-token layer | source index 45 | sparse MLA + MoE + `eh_proj` / `enorm` / `hnorm` / `shared_head`; no mHC |
| Vision tower | 24 blocks plus patch, downsample, merger | BF16 |

Native tensors: 1618. Packed EXL3 tensors: 148608. FP32 tensors: 291. These
match `materialization-receipt.json`.

`o_proj` is `[4096, 8192]` on KDA layers and `[4096, 16384]` on sparse-MLA
layers, including layer 45. The 45-layer execution plan still covers decoder
layers 0–44. Layer 45 is the checkpoint's own next-token module; DFlash2 is a
separate draft and does not replace these tensors.

## EXL3 storage shape

Only routed expert `gate_proj`, `up_proj`, and `down_proj` are packed.
Sibling BF16 linears in this checkpoint are `[N, K]` (out, in). Each packed
linear keeps that orientation:

| Suffix | Dtype | Shape |
|---|---|---|
| `suh` | F16 | `[K]` |
| `svh` | F16 | `[N]` |
| `trellis` | I16 | `[K/16, N/16, 64]` |
| `mcg` | I32 | `[1]` |

For gate and up, `N = 2048` and `K = 4096`. For down, `N = 4096` and
`K = 2048`. Every `.mcg` value read from the shards is `0xCBAC1FED`, the
multiplier in `exl3-mcg-storage-abi.json`.

This does not decode the trellis. P2 owns the packed representation. The
checkpoint also still says `serving_reader_qualified: false`, so a successful
bind is not evidence of a serving kernel.

## Draft layout

DFlash2 is five sliding-attention layers, BF16, with no EXL3 tensors.

- `fc` is `[4096, 20480]`: hidden size times the five target layers
  `5, 14, 24, 33, 42`.
- Attention is 32 query heads and 8 KV heads at head dim 128.
- Each layer has attention and MLP depthwise convs: base kernel `[2, 2, 4096]`
  and kernel projection `[1024, 4096]`.
- `block_size` is 8. The deployment contract's `DFLASH_TOKENS=7` is a runtime
  proposal count, not a second copy of this field.

## Run

```bash
./build/ninfer-glm53-bind /path/to/checkpoint -o receipt.json
```

Exit 0 means the catalog covers every stored tensor and every expected tensor
was found with the pinned dtype and shape. `--names-only` checks the index
without opening shard headers; that mode is not the P1 exit check.
