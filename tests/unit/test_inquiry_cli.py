"""CLI contracts, checkpoint inspection, and the complete offline entry point."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cognitive_core.inquiry import cli
from cognitive_core.inquiry.models import (
    InquiryState,
    QuestionRecord,
    UserClarification,
)
from cognitive_core.inquiry.store import InquiryStore
from cognitive_core.primitives.schemas import FrameOutput, FrameQuestion


class TestInquiryCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "study"
        self.store = InquiryStore(self.directory)

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def make_state(self, status="awaiting_user"):
        question = FrameQuestion(
            id="q1", question="Where does autonomy help?", contribution="Identify useful boundaries",
        )
        frame = FrameOutput(
            confidence=0.8, reasoning="The question needs an operational scope.",
            working_question="Where should a workflow use autonomous reasoning?",
            purpose="Choose an architecture", subquestions=[question],
            answer_criteria=["Distinguish predictable steps from uncertain ones"],
        )
        return InquiryState(
            original_question="Should every step be an agent?", status=status, phase="work",
            frames=[frame], questions={"q1": QuestionRecord(specification=question, frame_version=1)},
            clarifications=[UserClarification(
                id="C1", question="Which workflow do you have in mind?",
                reason="The level of uncertainty changes the design.", question_id="q1",
            )],
        )

    def test_status_is_read_only_and_displays_clarification(self):
        state = self.make_state()
        self.store.save(state)
        original_bytes = self.store.state_path.read_bytes()
        with patch.object(cli, "_build_runner", side_effect=AssertionError("Must not create runtime")):
            code, stdout, stderr = self.invoke("status", str(self.directory))
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn(state.frame.working_question, stdout)
        self.assertIn("q1 [pending; plan; rounds 0]", stdout)
        self.assertIn("C1 (required, q1): Which workflow do you have in mind?", stdout)
        self.assertIn("steps 0/80; searches 0/12; reads 0/18", stdout)
        self.assertIn(str(self.store.state_path), stdout)
        self.assertEqual(original_bytes, self.store.state_path.read_bytes())

    def test_report_prints_saved_content_without_runtime(self):
        content = "# Inquiry report\n\nA conditional conclusion.\n"
        self.store.write_report(content)
        with patch.object(cli, "_build_runner", side_effect=AssertionError("Must not create runtime")):
            code, stdout, stderr = self.invoke("report", str(self.directory))
        self.assertEqual((code, stdout, stderr), (0, content, ""))

    def test_missing_state_reports_actionable_error(self):
        code, stdout, stderr = self.invoke("status", str(self.directory))
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("No saved inquiry", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_invalid_state_does_not_echo_payload(self):
        self.directory.mkdir()
        self.store.state_path.write_text('{"SERPER_API_KEY": "do-not-print-this"}', encoding="utf-8")
        before = self.store.state_path.read_bytes()
        code, stdout, stderr = self.invoke("status", str(self.directory))
        self.assertEqual(code, 1)
        self.assertIn("saved state is invalid", stderr)
        self.assertNotIn("do-not-print-this", stdout + stderr)
        self.assertEqual(before, self.store.state_path.read_bytes())

    def test_start_passes_question_context_and_budgets(self):
        context_path = Path(self.temp.name) / "context.md"
        context_path.write_text("# Design constraints\nHuman review is available.\n", encoding="utf-8")
        runner = MagicMock()
        runner.start.return_value = InquiryState(original_question="A question", status="budget_exhausted")
        with patch.object(cli, "_build_runner", return_value=runner) as build:
            code, stdout, stderr = self.invoke(
                "start", "  A question  ", "--output", str(self.directory),
                "--purpose", "Select a design", "--context-file", str(context_path),
                "--max-steps", "6", "--max-searches", "2", "--max-reads", "3",
                "--model", "test-model", "--provider", "test-provider", "--quiet",
            )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("budget_exhausted", stdout)
        self.assertEqual(runner.start.call_args.args, ("A question",))
        kwargs = runner.start.call_args.kwargs
        self.assertEqual(kwargs["purpose"], "Select a design")
        self.assertEqual(kwargs["context"], context_path.read_text(encoding="utf-8"))
        self.assertEqual(kwargs["limits"].max_steps, 6)
        self.assertEqual(kwargs["limits"].max_searches, 2)
        self.assertEqual(kwargs["limits"].max_reads, 3)
        self.assertEqual(build.call_args.kwargs["provider"], "test-provider")
        self.assertTrue(build.call_args.kwargs["quiet"])

    def test_existing_inquiry_cannot_be_overwritten(self):
        self.store.save(self.make_state())
        before = self.store.state_path.read_bytes()
        with patch.object(cli, "_build_runner") as build:
            code, _, stderr = self.invoke("start", "Replacement", "--output", str(self.directory))
        self.assertEqual(code, 1)
        self.assertIn("already contains an inquiry", stderr)
        build.assert_not_called()
        self.assertEqual(before, self.store.state_path.read_bytes())

    def test_missing_context_file_fails_before_runtime(self):
        with patch.object(cli, "_build_runner") as build:
            code, _, stderr = self.invoke(
                "start", "A question", "--output", str(self.directory),
                "--context-file", str(Path(self.temp.name) / "missing.md"),
            )
        self.assertEqual(code, 1)
        self.assertIn("--context-file was not found", stderr)
        build.assert_not_called()

    def test_oversized_context_file_is_rejected_before_runtime(self):
        context_path = Path(self.temp.name) / "too-large.md"
        context_path.write_text("a" * 200001, encoding="utf-8")
        with patch.object(cli, "_build_runner") as build:
            code, _, stderr = self.invoke(
                "start", "A question", "--output", str(self.directory),
                "--context-file", str(context_path),
            )
        self.assertEqual(code, 1)
        self.assertIn("200,000 characters", stderr)
        build.assert_not_called()

    def test_resume_preserves_equals_in_answers_and_passes_additional_budget(self):
        self.store.save(self.make_state())
        runner = MagicMock()
        runner.resume.return_value = self.make_state(status="budget_exhausted")
        with patch.object(cli, "_build_runner", return_value=runner):
            code, _, stderr = self.invoke(
                "resume", str(self.directory), "--answer", "C1=cost = latency",
                "--additional-steps", "10", "--additional-searches", "2", "--additional-reads", "3",
            )
        self.assertEqual((code, stderr), (0, ""))
        runner.resume.assert_called_once_with(
            answers={"C1": "cost = latency"}, additional_steps=10, additional_searches=2, additional_reads=3,
        )

    def test_unknown_clarification_fails_before_runtime(self):
        self.store.save(self.make_state())
        with patch.object(cli, "_build_runner") as build:
            code, _, stderr = self.invoke("resume", str(self.directory), "--answer", "C99=some answer")
        self.assertEqual(code, 1)
        self.assertIn("not a pending clarification", stderr)
        build.assert_not_called()

    def test_duplicate_or_malformed_answers_do_not_echo_values(self):
        self.store.save(self.make_state())
        for flags in (
            ["--answer", "do-not-print-this"],
            ["--answer", "C1="],
            ["--answer", "C1=do-not-print-this", "--answer", "C1=another answer"],
        ):
            with self.subTest(flags=flags):
                with patch.object(cli, "_build_runner") as build:
                    code, stdout, stderr = self.invoke("resume", str(self.directory), *flags)
                self.assertEqual(code, 1)
                self.assertNotIn("do-not-print-this", stdout + stderr)
                build.assert_not_called()

    def test_completed_resume_is_a_read_only_noop(self):
        self.store.save(self.make_state(status="completed"))
        with patch.object(cli, "_build_runner") as build:
            code, stdout, stderr = self.invoke("resume", str(self.directory))
        self.assertEqual((code, stderr), (0, ""))
        self.assertIn("Status: completed", stdout)
        build.assert_not_called()

    def test_cumulative_budget_limits_checked_before_runtime(self):
        self.store.save(self.make_state())
        with patch.object(cli, "_build_runner") as build:
            code, _, stderr = self.invoke("resume", str(self.directory), "--additional-steps", "1000")
        self.assertEqual(code, 1)
        self.assertIn("exceeds the per-inquiry limit", stderr)
        build.assert_not_called()

    def test_invalid_budget_is_an_argument_error(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), patch.object(cli, "_build_runner") as build:
            with self.assertRaises(SystemExit) as raised:
                cli.main(["start", "A question", "--output", str(self.directory), "--max-steps", "0"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("between 1 and 1000", stderr.getvalue())
        build.assert_not_called()

    def test_unexpected_exception_is_sanitized(self):
        with patch.object(cli, "_build_runner", side_effect=RuntimeError("secret-header-do-not-print")):
            code, stdout, stderr = self.invoke("start", "A question", "--output", str(self.directory))
        self.assertEqual(code, 1)
        self.assertIn("RuntimeError", stderr)
        self.assertNotIn("secret-header-do-not-print", stdout + stderr)
        self.assertNotIn("Traceback", stderr)

    def test_keyboard_interrupt_has_recovery_hint(self):
        with patch.object(cli, "_build_runner", side_effect=KeyboardInterrupt):
            code, _, stderr = self.invoke("start", "A question", "--output", str(self.directory))
        self.assertEqual(code, 130)
        self.assertIn("saved checkpoint", stderr)

    def test_existing_writer_has_specific_recovery_hint(self):
        runner = MagicMock()
        runner.start.side_effect = RuntimeError("This inquiry is already running in another process")
        with patch.object(cli, "_build_runner", return_value=runner):
            code, _, stderr = self.invoke("start", "A question", "--output", str(self.directory))
        self.assertEqual(code, 1)
        self.assertIn("already running in another process", stderr)
        self.assertIn("status", stderr)

    def test_failed_run_returns_failure_exit_code(self):
        runner = MagicMock()
        runner.start.return_value = InquiryState(
            original_question="A question", status="failed", error="Model provider is not configured",
        )
        with patch.object(cli, "_build_runner", return_value=runner):
            code, stdout, _ = self.invoke("start", "A question", "--output", str(self.directory))
        self.assertEqual(code, 1)
        self.assertIn("Model provider is not configured", stdout)

    def test_module_entry_point_help(self):
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "cognitive_core.inquiry", "--help"],
            cwd=repository, text=True, capture_output=True, check=False, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start,resume,status,report,demo", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_offline_demo_end_to_end(self):
        with patch("socket.create_connection", side_effect=AssertionError("Demo must stay offline")):
            code, stdout, stderr = self.invoke("demo", "--output", str(self.directory), "--quiet")
        self.assertEqual(code, 0, stderr)
        self.assertIn("OFFLINE / SYNTHETIC", stdout)
        self.assertIn("not empirical research", stdout)
        state = self.store.load()
        self.assertEqual(state.status, "completed")
        self.assertTrue(state.answer)
        self.assertGreater(state.usage.steps, 0)
        self.assertTrue(self.store.report_path.is_file())


if __name__ == "__main__":
    unittest.main()
