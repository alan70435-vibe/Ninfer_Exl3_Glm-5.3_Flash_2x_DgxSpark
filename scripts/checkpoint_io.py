"""Bounded metadata IO for local, immutable checkpoint snapshots (no payload loading).

Symlinks may resolve into explicitly approved cache roots. On Linux, the opened
file descriptor's resolved path is checked again before reading. This is not a
publisher signature check; directories must not be concurrently reparented.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import struct
from typing import BinaryIO, Iterator

CONFIG_LIMIT = 16 * 1024 * 1024
INDEX_LIMIT = 64 * 1024 * 1024
HEADER_LIMIT = 256 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
               "I64": 8, "U64": 8, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1}


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def json_object(raw: bytes) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


@contextmanager
def open_checkpoint_file(root: Path, name: str, trusted_roots: tuple[Path, ...] = ()) -> Iterator[BinaryIO]:
    if (not isinstance(name, str) or not name or "\\" in name or
            any(ord(c) < 32 for c in name) or PurePosixPath(name).is_absolute() or
            PureWindowsPath(name).drive or ".." in PurePosixPath(name).parts):
        raise ValueError(f"invalid relative checkpoint path: {name!r}")
    roots = (root.resolve(strict=True),) + tuple(p.resolve(strict=True) for p in trusted_roots)
    if not all(p.is_dir() for p in roots):
        raise ValueError("checkpoint/trusted root must be a directory")
    resolved = (root / name).resolve(strict=True)
    if not _inside(resolved, roots):
        raise ValueError(f"checkpoint path escapes approved roots: {name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(resolved, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"checkpoint path is not a regular file: {name}")
        if Path("/proc/self/fd").is_dir():
            actual = Path(f"/proc/self/fd/{fd}").resolve(strict=True)
            if not _inside(actual, roots):
                raise ValueError(f"opened checkpoint file escapes approved roots: {name}")
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            yield stream
    finally:
        if fd >= 0:
            os.close(fd)


def read_json_file(root: Path, name: str, trusted_roots: tuple[Path, ...], limit: int) -> tuple[dict, str]:
    with open_checkpoint_file(root, name, trusted_roots) as stream:
        if os.fstat(stream.fileno()).st_size > limit:
            raise ValueError(f"metadata file exceeds {limit} byte limit: {name}")
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError(f"metadata file grew beyond limit: {name}")
    return json_object(raw), hashlib.sha256(raw).hexdigest()


def read_header(stream: BinaryIO) -> dict:
    size = os.fstat(stream.fileno()).st_size
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("short safetensors framing")
    length, = struct.unpack("<Q", raw)
    if not 0 < length <= HEADER_LIMIT or length > size - 8:
        raise ValueError("invalid/truncated safetensors header length")
    raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise ValueError("invalid/truncated safetensors JSON header")
    header = json_object(raw)
    payload_bytes = size - 8 - length
    ranges: list[tuple[int, int]] = []
    for name, desc in header.items():
        if name == "__metadata__":
            if not isinstance(desc, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in desc.items()):
                raise ValueError("safetensors metadata must map strings to strings")
            continue
        if not name or not isinstance(desc, dict):
            raise ValueError(f"invalid tensor description: {name}")
        shape, offsets, dtype = desc.get("shape"), desc.get("data_offsets"), desc.get("dtype")
        if not isinstance(shape, list) or len(shape) > 16 or not all(type(d) is int and 0 <= d <= 2**63 - 1 for d in shape):
            raise ValueError(f"invalid tensor shape: {name}")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported tensor dtype: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(d) is int for d in offsets):
            raise ValueError(f"invalid tensor offsets: {name}")
        lo, hi = offsets
        if not 0 <= lo <= hi <= payload_bytes or hi - lo != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"tensor byte extent/EOF mismatch: {name}")
        ranges.append((lo, hi))
    end = 0
    for lo, hi in sorted(ranges):
        if lo != end:
            raise ValueError("safetensors data has holes or overlapping tensor ranges")
        end = hi
    if end != payload_bytes:
        raise ValueError("safetensors data buffer is not fully indexed")
    return header


def hash_open_file(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(8 * 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()
