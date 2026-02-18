"""
Cognitive Core — S-008: Spec-Locking Manifest Tests
"""

import os
import sys
import tempfile
import shutil
import time
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

# Direct import — engine.manifest has no heavy deps
_manifest_path = os.path.join(_base, "engine", "manifest.py")
_ns = {}
with open(_manifest_path) as _f:
    exec(compile(_f.read(), _manifest_path, "exec"), _ns)
SpecManifest = _ns["SpecManifest"]
FileSnapshot = _ns["FileSnapshot"]
is_manifest_enabled = _ns["is_manifest_enabled"]
_hash_content = _ns["_hash_content"]


class TestHashContent(unittest.TestCase):

    def test_deterministic(self):
        h1 = _hash_content("hello world")
        h2 = _hash_content("hello world")
        self.assertEqual(h1, h2)

    def test_different_content_different_hash(self):
        h1 = _hash_content("hello")
        h2 = _hash_content("world")
        self.assertNotEqual(h1, h2)

    def test_sha256_length(self):
        h = _hash_content("test")
        self.assertEqual(len(h), 64)  # SHA-256 hex = 64 chars


class TestCaptureManifest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create test files
        self.wf_path = os.path.join(self.tmpdir, "workflow.yaml")
        self.dm_path = os.path.join(self.tmpdir, "domain.yaml")
        self.coord_path = os.path.join(self.tmpdir, "coordinator.yaml")
        self.prompts_dir = os.path.join(self.tmpdir, "prompts")
        os.makedirs(self.prompts_dir)

        with open(self.wf_path, "w") as f:
            f.write("name: test_workflow\nsteps:\n  - name: classify\n")
        with open(self.dm_path, "w") as f:
            f.write("name: test_domain\ncategories: [a, b]\n")
        with open(self.coord_path, "w") as f:
            f.write("governance:\n  tiers: [auto, gate]\n")
        with open(os.path.join(self.prompts_dir, "classify.txt"), "w") as f:
            f.write("Classify this: {context}")
        with open(os.path.join(self.prompts_dir, "investigate.txt"), "w") as f:
            f.write("Investigate: {question}")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_capture_all_files(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            domain_path=self.dm_path,
            coordinator_path=self.coord_path,
            prompt_dir=self.prompts_dir,
        )
        self.assertIn("workflow", m.files)
        self.assertIn("domain", m.files)
        self.assertIn("coordinator", m.files)
        self.assertIn("prompt:classify.txt", m.files)
        self.assertIn("prompt:investigate.txt", m.files)
        self.assertEqual(len(m.files), 5)

    def test_manifest_hash_generated(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            domain_path=self.dm_path,
        )
        self.assertEqual(len(m.manifest_hash), 64)

    def test_captured_at_timestamp(self):
        before = time.time()
        m = SpecManifest.capture(workflow_path=self.wf_path)
        after = time.time()
        self.assertGreaterEqual(m.captured_at, before)
        self.assertLessEqual(m.captured_at, after)

    def test_content_stored(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            include_content=True,
        )
        content = m.get_file_content("workflow")
        self.assertIn("test_workflow", content)

    def test_content_excluded_when_disabled(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            include_content=False,
        )
        content = m.get_file_content("workflow")
        self.assertEqual(content, "")

    def test_missing_file_skipped(self):
        m = SpecManifest.capture(
            workflow_path="/nonexistent/file.yaml",
            domain_path=self.dm_path,
        )
        self.assertNotIn("workflow", m.files)
        self.assertIn("domain", m.files)

    def test_empty_prompt_dir_ok(self):
        empty_dir = os.path.join(self.tmpdir, "empty_prompts")
        os.makedirs(empty_dir)
        m = SpecManifest.capture(prompt_dir=empty_dir)
        prompt_keys = [k for k in m.files if k.startswith("prompt:")]
        self.assertEqual(len(prompt_keys), 0)

    def test_nonexistent_prompt_dir_ok(self):
        m = SpecManifest.capture(
            prompt_dir="/nonexistent/dir",
            coordinator_path="",  # No coordinator
        )
        # Only files with valid paths are captured
        prompt_keys = [k for k in m.files if k.startswith("prompt:")]
        self.assertEqual(len(prompt_keys), 0)

    def test_same_content_same_hash(self):
        m1 = SpecManifest.capture(workflow_path=self.wf_path)
        m2 = SpecManifest.capture(workflow_path=self.wf_path)
        self.assertEqual(m1.manifest_hash, m2.manifest_hash)

    def test_different_content_different_hash(self):
        m1 = SpecManifest.capture(workflow_path=self.wf_path)
        # Modify the file
        with open(self.wf_path, "a") as f:
            f.write("\n  - name: investigate\n")
        m2 = SpecManifest.capture(workflow_path=self.wf_path)
        self.assertNotEqual(m1.manifest_hash, m2.manifest_hash)


class TestVerifyManifest(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wf_path = os.path.join(self.tmpdir, "workflow.yaml")
        self.dm_path = os.path.join(self.tmpdir, "domain.yaml")
        with open(self.wf_path, "w") as f:
            f.write("name: test_workflow\n")
        with open(self.dm_path, "w") as f:
            f.write("name: test_domain\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_no_drift_when_unchanged(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            domain_path=self.dm_path,
        )
        drifts = m.verify()
        self.assertEqual(len(drifts), 0)

    def test_drift_detected_on_change(self):
        m = SpecManifest.capture(workflow_path=self.wf_path)
        # Modify file
        with open(self.wf_path, "w") as f:
            f.write("name: changed_workflow\n")
        drifts = m.verify()
        self.assertEqual(len(drifts), 1)
        self.assertEqual(drifts[0]["label"], "workflow")
        self.assertEqual(drifts[0]["status"], "changed")
        self.assertNotEqual(drifts[0]["stored_hash"], drifts[0]["current_hash"])

    def test_drift_detected_on_delete(self):
        m = SpecManifest.capture(workflow_path=self.wf_path)
        os.remove(self.wf_path)
        drifts = m.verify()
        self.assertEqual(len(drifts), 1)
        self.assertEqual(drifts[0]["status"], "missing")

    def test_multiple_drifts(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            domain_path=self.dm_path,
        )
        with open(self.wf_path, "w") as f:
            f.write("changed\n")
        os.remove(self.dm_path)
        drifts = m.verify()
        self.assertEqual(len(drifts), 2)
        statuses = {d["status"] for d in drifts}
        self.assertEqual(statuses, {"changed", "missing"})


class TestSerialization(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wf_path = os.path.join(self.tmpdir, "workflow.yaml")
        with open(self.wf_path, "w") as f:
            f.write("name: test_workflow\nsteps: [classify]\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_round_trip(self):
        m = SpecManifest.capture(workflow_path=self.wf_path)
        d = m.to_dict()
        m2 = SpecManifest.from_dict(d)
        self.assertEqual(m.manifest_hash, m2.manifest_hash)
        self.assertEqual(m.captured_at, m2.captured_at)
        self.assertEqual(len(m.files), len(m2.files))

    def test_content_survives_round_trip(self):
        m = SpecManifest.capture(workflow_path=self.wf_path, include_content=True)
        d = m.to_dict()
        m2 = SpecManifest.from_dict(d)
        self.assertEqual(
            m.get_file_content("workflow"),
            m2.get_file_content("workflow"),
        )

    def test_to_dict_json_serializable(self):
        m = SpecManifest.capture(workflow_path=self.wf_path)
        d = m.to_dict()
        # Should not raise
        import json
        s = json.dumps(d)
        self.assertIn("manifest_hash", s)

    def test_file_labels(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            coordinator_path="",
            prompt_dir="",
        )
        self.assertEqual(m.file_labels, ["workflow"])

    def test_file_count_in_dict(self):
        m = SpecManifest.capture(
            workflow_path=self.wf_path,
            coordinator_path="",
            prompt_dir="",
        )
        d = m.to_dict()
        self.assertEqual(d["file_count"], 1)


class TestManifestEnabled(unittest.TestCase):

    def test_default_enabled(self):
        old = os.environ.pop("CC_MANIFEST_ENABLED", None)
        try:
            self.assertTrue(is_manifest_enabled())
        finally:
            if old is not None:
                os.environ["CC_MANIFEST_ENABLED"] = old

    def test_explicitly_disabled(self):
        old = os.environ.get("CC_MANIFEST_ENABLED")
        os.environ["CC_MANIFEST_ENABLED"] = "false"
        try:
            self.assertFalse(is_manifest_enabled())
        finally:
            if old is not None:
                os.environ["CC_MANIFEST_ENABLED"] = old
            else:
                os.environ.pop("CC_MANIFEST_ENABLED", None)

    def test_explicitly_enabled(self):
        os.environ["CC_MANIFEST_ENABLED"] = "true"
        try:
            self.assertTrue(is_manifest_enabled())
        finally:
            os.environ.pop("CC_MANIFEST_ENABLED", None)


class TestWithRealProjectFiles(unittest.TestCase):
    """Test capture against actual project files if they exist."""

    def test_capture_real_workflow(self):
        wf = os.path.join(_base, "workflows", "product_return.yaml")
        if not os.path.exists(wf):
            self.skipTest("product_return.yaml not found")
        m = SpecManifest.capture(workflow_path=wf)
        self.assertIn("workflow", m.files)
        self.assertGreater(m.files["workflow"].size_bytes, 0)

    def test_capture_real_prompts(self):
        pd = os.path.join(_base, "registry", "prompts")
        if not os.path.isdir(pd):
            self.skipTest("prompts dir not found")
        m = SpecManifest.capture(prompt_dir=pd)
        prompt_keys = [k for k in m.files if k.startswith("prompt:")]
        self.assertGreater(len(prompt_keys), 0)

    def test_verify_no_drift_on_real_files(self):
        wf = os.path.join(_base, "workflows", "product_return.yaml")
        dm = os.path.join(_base, "domains", "electronics_return.yaml")
        if not os.path.exists(wf) or not os.path.exists(dm):
            self.skipTest("project files not found")
        m = SpecManifest.capture(workflow_path=wf, domain_path=dm)
        drifts = m.verify()
        self.assertEqual(len(drifts), 0)


if __name__ == "__main__":
    unittest.main()
