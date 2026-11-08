"""Exercise the shipped command path, including native persisted continuation."""
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cognitive_core.engine.governance import reset_governance
from cognitive_core.inquiry.native_cli import main, _store


class NativeInquiryCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "study"
        env = patch.dict(os.environ, {
            "CC_CACHE_ENABLED": "false", "CC_WEBHOOK_URL": "",
            "CC_REPLAY_ENABLED": "false", "LLM_MODEL_ID": "",
        })
        env.start()
        self.addCleanup(env.stop)
        reset_governance()
        self.addCleanup(reset_governance)

    def command(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def state(self):
        manifest = json.loads((self.directory / "run.json").read_text())
        store = _store(self.directory)
        try:
            instance = store.get_instance(manifest["instance_id"])
            state = store.get_workflow_checkpoint(instance.instance_id)
            ledger = store.get_ledger(instance_id=instance.instance_id)
            return instance, state, ledger
        finally:
            store.close()

    def test_shipped_demo_runs_native_chain_with_material_revision(self):
        code, out, err = self.command("demo", "--output", str(self.directory), "--quiet")
        self.assertEqual(code, 0, err)
        instance, state, ledger = self.state()
        self.assertEqual(instance.status.value, "completed")
        self.assertEqual(state["metadata"]["execution_profile"], "inquiry.v1")
        self.assertEqual(state["study"]["usage"], {"steps": 12, "model_calls": 24, "searches": 1, "reads": 1})
        self.assertEqual([step["primitive"] for step in state["steps"]], [
            "frame", "retrieve", "retrieve", "investigate", "verify", "challenge",
            "reflect", "investigate", "deliberate", "generate", "challenge", "govern",
        ])
        self.assertFalse(state["steps"][5]["output"]["survives"])
        self.assertTrue(state["steps"][10]["output"]["survives"])
        self.assertIn("prior approval", state["steps"][7]["output"]["finding"])
        self.assertNotEqual(state["steps"][3]["output"]["finding"], state["steps"][7]["output"]["finding"])
        self.assertEqual(state["study"]["sources"]["S1"]["status"], "read")
        self.assertEqual(len([e for e in ledger if e["action_type"] == "step_completed"]), 12)
        report = (self.directory / "report.md").read_text()
        self.assertIn("Synthetic demonstration", report)
        self.assertIn("[S1:P1]", report)
        self.assertIn("research outcome: conditional", out)

    def test_clarification_resume_uses_database_not_json_export(self):
        code, _, err = self.command("demo", "--output", str(self.directory), "--clarify", "--quiet")
        self.assertEqual(code, 0, err)
        instance, state, _ = self.state()
        self.assertEqual(instance.status.value, "suspended")
        self.assertEqual(state["study"]["usage"]["steps"], 1)
        (self.directory / "state.json").write_text('{"tampered":"inspection export only"}')
        code, _, err = self.command("resume", str(self.directory), "--answer", "c1=Yes, only the fictional policy.", "--quiet")
        self.assertEqual(code, 0, err)
        instance, state, _ = self.state()
        self.assertEqual(instance.status.value, "completed")
        self.assertEqual(state["study"]["clarifications"]["c1"]["answer"], "Yes, only the fictional policy.")
        self.assertEqual(sum(s["primitive"] == "frame" for s in state["steps"]), 1)

    def test_status_and_report_do_not_construct_model_and_output_is_not_overwritten(self):
        self.command("demo", "--output", str(self.directory), "--quiet")
        with patch("cognitive_core.engine.llm.create_llm", side_effect=AssertionError("read-only inspection called model")):
            for command in ("status", "report"):
                code, _, err = self.command(command, str(self.directory))
                self.assertEqual(code, 0, err)
        code, _, err = self.command("demo", "--output", str(self.directory), "--quiet")
        self.assertEqual(code, 2)
        self.assertIn("cannot be overwritten", err)

    def test_old_checkpoint_is_not_misread_as_native_state(self):
        self.directory.mkdir()
        (self.directory / "state.json").write_text("{}")
        code, _, err = self.command("resume", str(self.directory), "--quiet")
        self.assertEqual(code, 2)
        self.assertIn("legacy InquiryRunner", err)

    def test_failed_native_calls_export_provisional_report_and_remain_recoverable(self):
        from cognitive_core.inquiry.native_demo import FixtureModel
        original = FixtureModel.response

        def malformed(model, operation, context):
            if operation == "retrieve":
                return {"calls": "read the source", "rationale": "Deliberately malformed fixture."}
            return original(model, operation, context)

        with patch.object(FixtureModel, "response", malformed):
            code, out, _ = self.command("demo", "--output", str(self.directory), "--quiet")
            self.assertEqual(code, 1)
            self.assertIn("Coordinator status: failed", out)
            self.assertTrue((self.directory / "report.md").is_file())
            self.assertTrue(json.loads((self.directory / "run.json").read_text())["instance_id"])
            code, out, _ = self.command("resume", str(self.directory), "--quiet")
            self.assertEqual(code, 1)
            self.assertIn("failed", out)
        code, _, err = self.command("resume", str(self.directory), "--quiet")
        self.assertEqual(code, 0, err)
        instance, state, _ = self.state()
        self.assertEqual(instance.status.value, "completed")
        self.assertEqual(sum(s["primitive"] == "frame" for s in state["steps"]), 1)
        self.assertGreater(state["study"]["usage"]["model_calls"], 24)

    def test_report_export_does_not_present_superseded_artifact_as_current_answer(self):
        from cognitive_core.inquiry.native_report import render_report
        self.command("demo", "--output", str(self.directory), "--quiet")
        instance, state, _ = self.state()
        state["study"]["report"]["stale"] = True
        state["study"]["report"]["artifact"] = "SUPERSEDED CLAIM MUST NOT APPEAR"
        state["study"]["findings"][-1]["finding"] = "Current corrected finding."
        rendered = render_report(instance, state)
        self.assertNotIn("SUPERSEDED CLAIM MUST NOT APPEAR", rendered)
        self.assertIn("Current corrected finding.", rendered)
        self.assertIn("not passed or not current", rendered)


if __name__ == "__main__":
    unittest.main()
