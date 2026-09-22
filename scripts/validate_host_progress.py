#!/usr/bin/env python3
"""Run the changed host modules and save commands/results/source hashes.

This focused harness is NOT the full repository CMake/CTest build or hardware
qualification. Full host regression remains in .github/workflows/host-ci.yml.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "gcc-release": ("g++", ["-O2"]),
    "gcc-debug": ("g++", ["-O0", "-g"]),
    "clang-sanitize": ("clang++", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="gcc-release")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cxx, flags = PROFILES[args.profile]
    env = dict(os.environ, CXX=cxx, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    report = {"profile": args.profile, "scope": "focused-changed-host-modules", "hardware_qualified": False,
              "full_repository_ctest": False, "commands": [], "source_sha256": {}}
    report["passed"] = False
    report["status"] = "running"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def run(label, cmd):
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        (args.output / (label + ".log")).write_text(result.stdout + result.stderr)
        report["commands"].append({"label": label, "command": cmd, "exit": result.returncode})
        return result.returncode == 0
    ok = run("compiler", [cxx, "--version"])
    with tempfile.TemporaryDirectory() as tmp:
        for suite in ("exl3_decode", "exl3_linear", "speculative", "public_headers"):
            sources = [] if suite == "public_headers" else ["src/speculative.cpp" if suite == "speculative" else "src/exl3_decode.cpp"]
            binary = str(Path(tmp) / suite)
            cmd = [cxx, "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow", *flags, "-Iinclude", *sources, f"tests/test_{suite}.cpp", "-o", binary]
            built = run("build-" + suite, cmd); ok &= built
            if built: ok &= run("run-" + suite, [binary])
        ok &= run("python", ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_host_*.py", "-v"])
        ok &= run("shell-syntax", ["bash", "-n", "scripts/preflight_dgxspark.sh"])
    paths = [*ROOT.glob("src/*.cpp"), *ROOT.glob("include/ninfer_glm53/*.hpp"),
             *ROOT.glob("tests/test_host_*.py"), *ROOT.glob("tests/test_exl3*.cpp"),
             ROOT / "tests/test_speculative.cpp", ROOT / "tests/test_public_headers.cpp", ROOT / "tests/exl3_test_oracle.hpp",
             *ROOT.glob("scripts/*.py"), ROOT / "scripts/preflight_dgxspark.sh"]
    for path in sorted(set(paths)):
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["status"] = "completed"
    report["passed"] = bool(ok)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"profile": args.profile, "passed": bool(ok), "scope": report["scope"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
