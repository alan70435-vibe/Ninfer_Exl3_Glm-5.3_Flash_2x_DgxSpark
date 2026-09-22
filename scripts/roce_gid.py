#!/usr/bin/env python3
"""Read-only validation of one RDMA GID entry and its netdev/type/IP binding.

A non-default sysfs root is a fixture, never a hardware qualification result.
This checks local configuration only; it does not measure link performance.
"""
from __future__ import annotations
import argparse
import ipaddress
from pathlib import Path
import re
import sys

_COMPONENT = re.compile(r"[A-Za-z0-9_.:-]+\Z")


def validate_gid(sysfs: Path, device: str, netdev: str, address: str,
                 index: int, nccl_index: int, port: int = 1) -> str:
    for value in (device, netdev):
        if not _COMPONENT.fullmatch(value) or value in (".", ".."):
            raise ValueError("invalid RDMA/netdev component")
    if not 0 <= index <= 4095 or index != nccl_index or not 1 <= port <= 255:
        raise ValueError("GID index/port invalid or inconsistent with NCCL_IB_GID_INDEX")
    expected = ipaddress.ip_address(address)
    root = sysfs / "class" / "infiniband" / device / "ports" / str(port)
    def read(relative: str) -> str:
        value = (root / relative / str(index)).read_text(encoding="ascii").strip()
        if not value:
            raise ValueError(f"empty GID attribute: {relative}")
        return value
    gid = ipaddress.IPv6Address(read("gids"))
    if gid.is_unspecified:
        raise ValueError("zero GID")
    actual_netdev = read("gid_attrs/ndevs")
    actual_type = read("gid_attrs/types")
    if actual_netdev != netdev:
        raise ValueError(f"GID netdev {actual_netdev!r} differs from {netdev!r}")
    if "".join(actual_type.lower().split()) != "rocev2":
        raise ValueError(f"expected RoCE v2, got {actual_type!r}")
    normalized = gid.ipv4_mapped or gid
    expected = getattr(expected, "ipv4_mapped", None) or expected
    if normalized != expected:
        raise ValueError(f"GID address {normalized} differs from {expected}")
    return f"{device}/port{port} GID[{index}]={gid} netdev={netdev} type=RoCEv2 ip={normalized}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--gid-index", type=int, required=True)
    parser.add_argument("--nccl-index", type=int, required=True)
    args = parser.parse_args()
    fixture = args.sysfs_root.resolve() != Path("/sys")
    print(f"observation={'fixture' if fixture else 'local-sysfs'} hardware_qualified=false")
    try:
        print(validate_gid(args.sysfs_root, args.ib_device, args.interface, args.ip, args.gid_index, args.nccl_index))
        return 0
    except (OSError, ValueError) as exc:
        print(f"GID validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
