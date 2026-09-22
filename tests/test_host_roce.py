from __future__ import annotations
import ipaddress
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roce_gid import validate_gid


class GidTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sysfs = Path(self.tmp.name)
        self.base = self.sysfs / "class/infiniband/rdma_test/ports/1"
        self.write("gids", "0000:0000:0000:0000:0000:ffff:c0a8:640a")
        self.write("gid_attrs/ndevs", "cx7_test")
        self.write("gid_attrs/types", "RoCE v2")

    def write(self, path, text):
        p = self.base / path / "3"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")

    def check(self, address="192.168.100.10", index=3, nccl_index=3):
        return validate_gid(self.sysfs, "rdma_test", "cx7_test", address, index, nccl_index)

    def test_valid_ipv4_mapped(self):
        self.assertIn("ip=192.168.100.10", self.check())

    def test_native_ipv6(self):
        self.write("gids", "2001:db8::1")
        self.assertIn("ip=2001:db8::1", self.check("2001:0db8:0:0:0:0:0:1"))

    def test_wrong_netdev(self):
        self.write("gid_attrs/ndevs", "other_nic")
        with self.assertRaises(ValueError): self.check()

    def test_wrong_roce_type(self):
        for value in ("IB/RoCE v1", "IB", "unknown"):
            with self.subTest(value=value):
                self.write("gid_attrs/types", value)
                with self.assertRaises(ValueError): self.check()

    def test_wrong_ip(self):
        self.write("gids", "::ffff:192.168.100.99")
        with self.assertRaises(ValueError): self.check()

    def test_missing_attributes(self):
        for path in ("gid_attrs/ndevs", "gid_attrs/types"):
            with self.subTest(path=path):
                p = self.base / path / "3"; data = p.read_bytes(); p.unlink()
                with self.assertRaises(OSError): self.check()
                p.write_bytes(data)

    def test_zero_gid(self):
        self.write("gids", "::")
        with self.assertRaises(ValueError): self.check()

    def test_malformed_gid(self):
        self.write("gids", "not-an-address")
        with self.assertRaises(ValueError): self.check()

    def test_empty_attributes(self):
        self.write("gid_attrs/types", "")
        with self.assertRaises(ValueError): self.check()

    def test_nccl_index_mismatch(self):
        with self.assertRaises(ValueError): self.check(nccl_index=2)

    def test_path_components(self):
        with self.assertRaises(ValueError):
            validate_gid(self.sysfs, "../rdma_test", "cx7_test", "192.168.100.10", 3, 3)

    def test_cli_fixture_disclosure_and_exit(self):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/roce_gid.py"),
                   "--sysfs-root", str(self.sysfs), "--ib-device", "rdma_test", "--interface", "cx7_test",
                   "--ip", "192.168.100.10", "--gid-index", "3", "--nccl-index", "3"]
        p = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("observation=fixture hardware_qualified=false", p.stdout)
        self.write("gid_attrs/ndevs", "other")
        p = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)


if __name__ == "__main__": unittest.main()
