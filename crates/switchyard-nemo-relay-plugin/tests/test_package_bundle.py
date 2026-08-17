# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native Relay plugin bundle packager."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[1]
PACKAGER = CRATE_ROOT / "scripts" / "package_bundle.py"
PACKAGE_NAME = "switchyard-nemo-relay-plugin"


class PackageBundleTest(unittest.TestCase):
    """Verify materialized and archived plugin bundle contents."""

    def test_materializes_and_archives_supported_formats(self) -> None:
        for archive_suffix in (".tar.gz", ".zip"):
            with self.subTest(archive_suffix=archive_suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                library = root / "libswitchyard_nemo_relay_plugin.so"
                library.write_bytes(b"compiled plugin")
                output = root / "bundle"
                archive = root / f"{PACKAGE_NAME}-0.2.0-linux-x86_64{archive_suffix}"

                subprocess.run(
                    [
                        sys.executable,
                        str(PACKAGER),
                        "--library",
                        str(library),
                        "--output",
                        str(output),
                        "--archive",
                        str(archive),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                expected = {
                    "LICENSE",
                    "NOTICE",
                    "config.schema.json",
                    library.name,
                    "relay-plugin.toml",
                }
                self.assertEqual({path.name for path in output.iterdir()}, expected)
                manifest = (output / "relay-plugin.toml").read_text(encoding="utf-8")
                self.assertIn(f'artifact = "{library.name}"', manifest)
                self.assertIn(hashlib.sha256(library.read_bytes()).hexdigest(), manifest)
                self.assertNotIn("<platform-library-file>", manifest)
                self.assertNotIn("<artifact-sha256>", manifest)
                self.assertEqual(self.archive_members(archive), {f"{PACKAGE_NAME}/{name}" for name in expected})

    @staticmethod
    def archive_members(archive: Path) -> set[str]:
        """Return regular-file paths from a supported bundle archive."""
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as stream:
                return {member.name for member in stream.getmembers() if member.isfile()}
        with zipfile.ZipFile(archive) as stream:
            return {member.filename for member in stream.infolist() if not member.is_dir()}


if __name__ == "__main__":
    unittest.main()
