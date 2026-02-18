"""
Cognitive Core — S-008: Spec-Locking Manifest Tests
"""

import gzip
import importlib.util
import json
import os
import sys
import tempfile
import shutil
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_sl_path = os.path.join(_base, "engine", "spec_lock.py")
_spec = importlib.util.spec_from_file_location("engine.spec_lock", _sl_path)
_sl_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.spec_lock"] = _sl_mod
_spec.loader.exec_module(_sl_mod)

create_manifest = _sl_mod.create_manifest
SpecManifest = _sl_mod.SpecManifest
is_spec_lock_enabled = _sl_mod.is_spec_lock_enabled
are_snapshots_enabled = _sl_mod.are_snapshots_enabled
_hash_file = _sl_mod._hash_file
_hash_string = _sl_mod._hash_string


class TestManifestCreation(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create fake config files
        with open(os.path.join(self.tmpdir, "workflow.yaml"), "w") as f:
            f.write("name: test_workflow\nsteps:\n  - classify\n  - investigate\n")
        with open(os.path.join(self.tmpdir, "domain.yaml"), "w") as f:
            f.write("name: test_domain\ntier: auto\n")
        with open(os.path.join(self.tmpdir, "coordinator.yaml"), "w") as f:
            f.write("governance:\n  tiers: [auto, gate]\n")
        os.makedirs(os.path.join(self.tmpdir, "prompts"))
        with open(os.path.join(self.tmpdir, "prompts", "classify.txt"), "w") as f:
            f.write("Classify this input: {context}")
        with open(os.path.join(self.tmpdir, "prompts", "investigate.txt"), "w") as f:
            f.write("Investigate: {question}")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_manifest(self):
        m = create_manifest(
            "workflow.yaml", "domain.yaml", "coordinator.yaml",
            prompt_dir="prompts", project_root=self.tmpdir,
        )
        self.assertEqual(len(m.manifest_hash), 64)  # SHA-256
        self.assertEqual(len(m.file_hashes), 3)
        self.assertEqual(len(m.prompt_hashes), 2)
        self.assertGreater(m.created_at, 0)

    def test_manifest_hash_deterministic(self):
        m1 = create_manifest("workflow.yaml", "domain.yaml", project_root=self.tmpdir)
        m2 = create_manifest("workflow.yaml", "domain.yaml", project_root=self.tmpdir)
        self.assertEqual(m1.manifest_hash, m2.manifest_hash)

    def test_manifest_hash_changes_on_file_change(self):
        m1 = create_manifest("workflow.yaml", "domain.yaml", project_root=self.tmpdir)
        # Modify file
        with open(os.path.join(self.tmpdir, "domain.yaml"), "a") as f:
            f.write("\nnew_field: changed\n")
        m2 = create_manifest("workflow.yaml", "domain.yaml", project_root=self.tmpdir)
        self.assertNotEqual(m1.manifest_hash, m2.manifest_hash)

    def test_snapshots_included(self):
        m = create_manifest(
            "workflow.yaml", "domain.yaml",
            project_root=self.tmpdir, include_snapshots=True,
        )
        self.assertIsNotNone(m.file_contents)
        self.assertIn("workflow.yaml", m.file_contents)
        self.assertIn("test_workflow", m.file_contents["workflow.yaml"])

    def test_snapshots_excluded(self):
        m = create_manifest(
            "workflow.yaml", "domain.yaml",
            project_root=self.tmpdir, include_snapshots=False,
        )
        self.assertIsNone(m.file_contents)

    def test_missing_file_skipped(self):
        m = create_manifest(
            "workflow.yaml", "nonexistent.yaml",
            project_root=self.tmpdir,
        )
        self.assertEqual(len(m.file_hashes), 1)  # Only workflow.yaml


class TestManifestSerialization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "w.yaml"), "w") as f:
            f.write("name: workflow\nsteps: [a, b, c]\n")
        with open(os.path.join(self.tmpdir, "d.yaml"), "w") as f:
            f.write("name: domain\ntier: auto\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_to_dict(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        d = m.to_dict()
        self.assertEqual(d["manifest_hash"], m.manifest_hash)
        self.assertTrue(d["has_snapshots"])
        self.assertGreater(d["snapshot_size_bytes"], 0)

    def test_to_dict_without_snapshots(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir,
                           include_snapshots=False)
        d = m.to_dict()
        self.assertFalse(d["has_snapshots"])

    def test_storage_roundtrip(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        stored = m.to_storage()
        restored = SpecManifest.from_storage(stored)
        self.assertEqual(restored.manifest_hash, m.manifest_hash)
        self.assertEqual(restored.file_hashes, m.file_hashes)
        self.assertEqual(restored.file_contents, m.file_contents)

    def test_storage_roundtrip_without_snapshots(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir,
                           include_snapshots=False)
        stored = m.to_storage()
        restored = SpecManifest.from_storage(stored)
        self.assertEqual(restored.manifest_hash, m.manifest_hash)
        self.assertIsNone(restored.file_contents)

    def test_compressed_snapshot_exists(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        stored = m.to_storage()
        # Compressed data should exist and be a valid hex string
        self.assertIn("snapshots_compressed", stored)
        self.assertGreater(stored["snapshots_compressed_bytes"], 0)
        self.assertGreater(stored["snapshots_raw_bytes"], 0)
        # Verify it decompresses correctly
        compressed = bytes.fromhex(stored["snapshots_compressed"])
        raw = gzip.decompress(compressed)
        contents = json.loads(raw)
        self.assertIn("w.yaml", contents)


class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "w.yaml"), "w") as f:
            f.write("name: workflow\n")
        with open(os.path.join(self.tmpdir, "d.yaml"), "w") as f:
            f.write("name: domain\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_drift(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        report = m.verify_against_current(self.tmpdir)
        self.assertTrue(report["matches_current"])
        self.assertEqual(report["changes_detected"], 0)

    def test_detect_modification(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        # Modify a file
        with open(os.path.join(self.tmpdir, "d.yaml"), "w") as f:
            f.write("name: domain_changed\n")
        report = m.verify_against_current(self.tmpdir)
        self.assertFalse(report["matches_current"])
        self.assertEqual(report["changes_detected"], 1)
        self.assertEqual(report["changes"][0]["status"], "modified")

    def test_detect_deletion(self):
        m = create_manifest("w.yaml", "d.yaml", project_root=self.tmpdir)
        os.remove(os.path.join(self.tmpdir, "d.yaml"))
        report = m.verify_against_current(self.tmpdir)
        self.assertFalse(report["matches_current"])
        self.assertEqual(report["changes"][0]["status"], "deleted")


class TestFeatureToggle(unittest.TestCase):
    def test_enabled_by_default(self):
        # Clear any env override
        saved = os.environ.pop("CC_SPEC_LOCK_ENABLED", None)
        try:
            self.assertTrue(is_spec_lock_enabled())
        finally:
            if saved:
                os.environ["CC_SPEC_LOCK_ENABLED"] = saved

    def test_disabled_via_env(self):
        os.environ["CC_SPEC_LOCK_ENABLED"] = "false"
        try:
            self.assertFalse(is_spec_lock_enabled())
        finally:
            del os.environ["CC_SPEC_LOCK_ENABLED"]

    def test_snapshots_enabled_by_default(self):
        saved = os.environ.pop("CC_SPEC_LOCK_SNAPSHOTS", None)
        try:
            self.assertTrue(are_snapshots_enabled())
        finally:
            if saved:
                os.environ["CC_SPEC_LOCK_SNAPSHOTS"] = saved

    def test_snapshots_disabled_via_env(self):
        os.environ["CC_SPEC_LOCK_SNAPSHOTS"] = "0"
        try:
            self.assertFalse(are_snapshots_enabled())
        finally:
            del os.environ["CC_SPEC_LOCK_SNAPSHOTS"]


class TestRealProjectManifest(unittest.TestCase):
    """Test against actual project files."""

    def test_create_from_real_files(self):
        wf = "workflows/product_return.yaml"
        dom = "domains/electronics_return.yaml"
        coord = "coordinator/config.yaml"
        if not all(os.path.exists(os.path.join(_base, p)) for p in [wf, dom, coord]):
            self.skipTest("Project files not found")

        m = create_manifest(
            wf, dom, coord,
            prompt_dir="registry/prompts",
            project_root=_base,
        )
        self.assertEqual(len(m.manifest_hash), 64)
        self.assertEqual(len(m.file_hashes), 3)
        self.assertGreater(len(m.prompt_hashes), 0)

        # Verify against current (should match)
        report = m.verify_against_current(_base)
        self.assertTrue(report["matches_current"])


if __name__ == "__main__":
    unittest.main()
