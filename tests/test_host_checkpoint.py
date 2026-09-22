from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import checkpoint_io as cio
from checkpoint_manifest import EXPECTED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/checkpoint_manifest.py"
NAME = "model.language_model.norm.weight"


def config():
    # Isolates scanner IO/coverage, NOT an independent real-model ABI fixture.
    result = {"architectures": ["Glm5NextForConditionalGeneration"], "text_config": {}}
    for key, value in EXPECTED.items():
        if key.startswith("architectures."): continue
        *parts, leaf = key.split(".")
        node = result
        for part in parts: node = node.setdefault(part, {})
        node[leaf] = value
    result["text_config"]["layer_types"] = ["deepseek_sparse_attention" if i >= 3 and (i-3)%4 == 0 else "linear_attention" for i in range(45)]
    result["text_config"]["mlp_layer_types"] = ["dense" if i < 3 else "sparse" for i in range(45)]
    return result


def shard(path, header, data=b""):
    raw = header if isinstance(header, bytes) else json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def tensor():
    return {"dtype": "BF16", "shape": [1], "data_offsets": [0,2]}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.root = self.base / "checkpoint"; self.root.mkdir()
        (self.root / "config.json").write_text(json.dumps(config()))
        self.weight = self.root / "model.safetensors"
        shard(self.weight, {NAME: tensor()}, b"\x80\x3f")
        self.index({NAME: self.weight.name})

    def index(self, mapping):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.root), *map(str,args)], capture_output=True, text=True, timeout=10)

    def scan(self, valid=False, *args):
        p = self.run_cli(*args)
        self.assertEqual(p.returncode, 0 if valid else 2, p.stderr + p.stdout)
        report = json.loads(p.stdout)
        self.assertEqual(report["valid_checkpoint_contract"], valid)
        self.assertFalse(report["payload_integrity_verified"])
        return report

    def test_coherent_control_and_hash(self):
        r = self.scan(True, "--hash-shards")
        self.assertEqual(r["tensor_count"], 1)
        self.assertEqual(r["header_observed_tensor_count"], 1)
        self.assertTrue(r["storage_ranges_checked"])
        self.assertEqual(r["weight_shards"][0]["sha256"], hashlib.sha256(self.weight.read_bytes()).hexdigest())

    def test_empty_header_with_index(self):
        shard(self.weight, {})
        r = self.scan(); self.assertEqual(r["tensor_count"], 0)
        self.assertEqual(r["index_declared_tensor_count"], 1)
        self.assertEqual(r["index_missing_from_headers"], [NAME])

    def test_metadata_only(self):
        shard(self.weight, {"__metadata__": {"format": "pt"}})
        self.assertEqual(self.scan()["tensor_count"], 0)

    def test_no_index_empty(self):
        (self.root / "model.safetensors.index.json").unlink(); shard(self.weight, {})
        self.assertEqual(self.scan()["tensor_count"], 0)

    def test_unindexed_valid_control(self):
        (self.root / "model.safetensors.index.json").unlink()
        self.scan(True)

    def test_mixed_empty_and_valid(self):
        shard(self.root / "empty.safetensors", {})
        self.index({NAME: self.weight.name, "missing.weight": "empty.safetensors"})
        self.assertEqual(self.scan()["index_missing_from_headers"], ["missing.weight"])

    def test_missing_shard(self):
        self.weight.unlink(); self.assertEqual(self.scan()["missing_shards"], [self.weight.name])

    def test_duplicate_tensor_across_shards(self):
        shard(self.root / "two.safetensors", {NAME: tensor()}, b"\0\0")
        self.index({NAME: self.weight.name, "absent": "two.safetensors"}); self.scan()

    def test_wrong_owner(self):
        shard(self.root / "two.safetensors", {"other": tensor()}, b"\0\0")
        self.index({NAME: "two.safetensors", "other": self.weight.name})
        self.assertEqual(len(self.scan()["index_owner_mismatches"]), 2)

    def test_parent_and_absolute_paths(self):
        outside = self.base / "outside.safetensors"; self.weight.rename(outside)
        for path in ("../outside.safetensors", str(outside), "C:\\outside.safetensors"):
            with self.subTest(path=path):
                self.index({NAME: path}); r = self.scan(False, "--hash-shards")
                self.assertEqual(r["tensor_count"], 0)
                self.assertIsNone(r["weight_shards"][0]["sha256"])

    def test_external_symlink_rejected_and_explicit_cache_allowed(self):
        cache = self.base / "cache"; cache.mkdir(); outside = cache / "blob"
        self.weight.rename(outside); self.weight.symlink_to(outside)
        self.scan()
        r = self.scan(True, "--trusted-root", cache, "--hash-shards")
        self.assertEqual(r["weight_shards"][0]["sha256"], hashlib.sha256(outside.read_bytes()).hexdigest())

    def test_internal_symlink_allowed(self):
        blob = self.root / "blob"; self.weight.rename(blob); self.weight.symlink_to(blob)
        self.scan(True)

    def test_broken_symlink(self):
        self.weight.unlink(); self.weight.symlink_to(self.root / "missing")
        self.scan()

    def test_unsupported_bin(self):
        self.weight.rename(self.root / "model.bin"); self.index({NAME: "model.bin"})
        self.assertFalse(self.scan()["header_coverage_checked"])

    def test_invalid_weight_map_types(self):
        for value in (42, None, [], ""):
            with self.subTest(value=value):
                self.index({NAME: value}); self.scan()

    def test_truncated_payload(self):
        shard(self.weight, {NAME: tensor()}, b"\0")
        self.assertFalse(self.scan()["storage_ranges_checked"])

    def test_bad_offsets(self):
        for offsets in ([-1,1], [2,0], [0,3], [0,100], [False,2], [0]):
            with self.subTest(offsets=offsets):
                desc = tensor(); desc["data_offsets"] = offsets
                shard(self.weight, {NAME: desc}, b"\0\0"); self.scan()

    def test_overlaps_holes_and_trailing_data(self):
        for lo,hi in ((0,2),(4,6)):
            second = tensor(); second["data_offsets"] = [lo,hi]
            shard(self.weight, {NAME: tensor(), "other": second}, b"\0" * max(4,hi))
            self.scan()
        shard(self.weight, {NAME: tensor()}, b"\0\0\0\0"); self.scan()

    def test_scalar_and_empty_tensor_ranges(self):
        scalar = tensor(); scalar["shape"] = []
        empty = {"dtype": "F32", "shape": [0,8], "data_offsets": [2,2]}
        shard(self.weight, {NAME: scalar, "empty": empty}, b"\0\0")
        self.index({NAME: self.weight.name, "empty": self.weight.name}); self.scan(True)

    def test_duplicate_header_key(self):
        desc = json.dumps(tensor())
        shard(self.weight, ('{"x":'+desc+',"x":'+desc+'}').encode(), b"\0\0"); self.scan()

    def test_huge_declared_header(self):
        self.weight.write_bytes(struct.pack("<Q", 1 << 50)); self.scan()

    def test_malformed_config_is_nonzero_without_traceback(self):
        (self.root / "config.json").write_text('{"architectures":[]}')
        self.scan()
        (self.root / "config.json").write_text('{"a":1,"a":2}')
        p = self.run_cli(); self.assertEqual(p.returncode, 2); self.assertNotIn("Traceback", p.stderr)

    def test_config_symlink_outside_rejected(self):
        p = self.root / "config.json"; outside = self.base / "config.json"; p.rename(outside); p.symlink_to(outside)
        r = self.run_cli(); self.assertEqual(r.returncode, 2); self.assertIn("approved roots", r.stderr)

    def test_regular_file_only(self):
        self.weight.unlink(); self.weight.mkdir()
        self.scan()


if __name__ == "__main__": unittest.main()
