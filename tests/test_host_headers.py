from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest


class HeaderTests(unittest.TestCase):
    def test_standalone_and_reverse_include_order(self):
        root = Path(__file__).resolve().parents[1]
        headers = ["exl3_abi.hpp", "exl3_decode.hpp", "parameter_schema.hpp", "speculative.hpp"]
        groups = [[h] for h in headers] + [headers, list(reversed(headers))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "headers.cpp"
            for group in groups:
                with self.subTest(group=group):
                    path.write_text("".join(f'#include "ninfer_glm53/{h}"\n' for h in group) + 'int main() {}\n')
                    command = shlex.split(os.environ.get("CXX", "c++"))
                    p = subprocess.run(command + ["-std=c++20", "-I"+str(root/"include"), "-fsyntax-only", str(path)], capture_output=True, text=True, timeout=30)
                    self.assertEqual(p.returncode, 0, p.stderr)
