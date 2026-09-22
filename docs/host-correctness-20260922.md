# Host correctness increment — 2026-09-22

Base: main `95540d87a9b200259550c9950ed83bb23beaa620`.
This increment does not merge or overwrite the separate v0.26 native runtime.
It fixes concrete host correctness boundaries before GPU bring-up.

## EXL3 and speculative output

- `exl3_abi.hpp` is the single main-line definition of the MCG multiplier and
  tile/block constants, shared by schema and decoder (#27).
- K4 unpacking reconstructs overlapping 16-bit tail-biting states. The ring
  wraps within a 256-element tile, not a packed word or 16-element span (#26).
- `exl3_decode_inner_tile` returns inner-basis Q. The old scaled-tile diagnostic
  is deprecated; scaling a 16x16 tile is not original-basis materialization.
- `exl3_linear_reference` implements x D_suh H128 Q H128 D_svh + bias, with F16
  scales/bias and FP64 accumulation. It decodes one tile at a time and does not
  allocate a full dense weight replica. Shapes, storage extents, element budget
  and finite inputs are checked (#34). This is a CPU oracle, not a fast backend.
- Greedy `commit_dflash2` now retains the target correction at every partial
  acceptance boundary, and the bonus on full acceptance. The accepted-draft
  count is separate from output count. Invalid inputs leave both local records
  unchanged (#29). Local record equality is NOT NCCL consensus; the helper has
  no EOS/budget handling, neural execution or KV/KDA state transaction.

The format/math references are the previously reviewed ExLlamaV3 snapshot
`6b84a21b6f1e5da3f291b9e1019061f0de788279` (`quant/pack.cu` and
`quant/reconstruct.cu`) and this repository's v0.26 `native/src/exl3_reference.cpp`.
Tests use raw-bit extraction, analytic half rounding and a dense Sylvester
matrix, not the implementation's packer or butterfly transform as expected data.
No upstream CUDA execution or real-checkpoint parity is claimed.

## Checkpoint scanner

`checkpoint_manifest.py` keeps legacy `valid_checkpoint_contract`, but that field
means its config/IO/inventory checks passed, NOT full model coverage or serving.
The schema is now version 3. `model_catalog_checked`, `gpu_qualified`, `deployable`
and `payload_integrity_verified` remain false. Measured shard hashes do not
prove authenticity without independently trusted expected hashes.

- Actual header-observed counts no longer fall back to index-declared counts.
  Empty and metadata-only checkpoints fail closed; even an empty observed map
  is compared against the complete index (#36).
- JSON rejects duplicate keys/nonfinite constants. Tensor shape/dtype/byte
  ranges, EOF, overlaps/holes and complete data-buffer coverage are checked.
  Unknown shard formats are explicitly unverified and rejected.
- `checkpoint_io.py` rejects absolute/parent-traversal paths and resolves
  symlinks against approved roots before payload/header reads. It checks regular
  files and, on Linux, the opened descriptor path. Header and optional hash use
  the same descriptor. Immutable snapshot directories are still required.
- Cache symlinks outside the snapshot require explicit approval, for example:

```bash
python3 scripts/checkpoint_manifest.py /cache/model/snapshots/REV \
  --trusted-root /cache/model/blobs --output results/local/manifest.json
```

This addresses the Python side of #37 only. The C++ binder is NOT routed through
this Python gate and still needs its own storage/path/config hardening (#28,
#35, #37, #39). Running this scanner is not a substitute for those changes.

## RoCE preflight

The existing shell preflight invokes `roce_gid.py` to validate the selected GID
entry's netdev, RoCE v2 type and normalized IPv4-mapped/IPv6 address together.
Missing attributes, zero/invalid addresses or index-policy mismatches fail (#38).
The helper is read-only. Alternate `--sysfs-root` values explicitly report
`observation=fixture`; no result claims GPU/network performance qualification.
The shell's env file is still trusted shell input and must be reviewed before use.

## Validation and reproduction

Focused local validation covers the changed modules, NOT a full repository
CMake/CTest build. The isolated workspace contains verified base blobs and the
changed units; absent main/native modules were not replaced by stubs.

Profiles: GCC 14.2 Release (`-O2`), GCC 14.2 Debug (`-O0 -g`), and Clang 17
ASan+UBSan (`-O1 -g -fsanitize=address,undefined`). Each profile runs:

- four C++ executables (decoder, original-basis linear, speculative, headers);
- all 65,536 MCG states and 2,560 raw state windows;
- all-zero analytic H128 plus three nonuniform multi-tile/two-row linear cases;
- 35 speculative accepted-prefix/correction cases and invalid/alias cases;
- 37 Python tests (scanner, GID fixtures, standalone/reversed public headers).

Commands and source hashes are emitted by the reproducible focused harness:

```bash
python3 scripts/validate_host_progress.py --profile gcc-release --output results/local/gcc-release
python3 scripts/validate_host_progress.py --profile gcc-debug --output results/local/gcc-debug
python3 scripts/validate_host_progress.py --profile clang-sanitize --output results/local/clang-sanitize
```

The root CMake now registers the new C++/Python tests. GitHub Actions is configured
for Release/Debug/sanitizer full builds and always-retained logs/JUnit. This does
not explain the previous Actions runner failure and is not itself evidence of a
successful hosted run (#30). Inspect the actual run attached to the new commit.

## Still open

No complete GLM forward, production 1M state layout, DFlash2 neural owner, Vision
GPU Program, SM121 kernel qualification or NCCL two-node execution was added.
Main/v0.26 consolidation and the native full-binder storage schema mismatch
remain (#25, #31, #33). The C++ config-scope/semantic gate work also remains
(#35, #39). No live Spark service, network, clocks or model precision was changed.
