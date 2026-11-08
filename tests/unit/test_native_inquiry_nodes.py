"""Native profile-node contracts through the real protected-call pipeline.

Only the model factory and registered external-tool implementations are fixture
boundaries.  Parsing, correction, tool admission, profile validation, reducers,
usage accounting, and GovernancePipeline.protected_llm_call execute normally.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cognitive_core.engine.governance import get_governance, reset_governance
from cognitive_core.engine.profile_nodes import create_profile_node
from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.native_profiles import apply_output, ingest_research_result, initial_study


ROOT = Path(__file__).resolve().parents[2]
URL_A = "https://example.org/research/a"
URL_B = "https://example.org/research/b"
TARGET = {"kind": "question", "question_id": "q1"}


class BoundaryModel:
    model_name = "fixture-native-profile"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        if not self.responses:
            raise AssertionError("Unexpected extra model request")
        response = self.responses.pop(0)
        if callable(response):
            response = response()
        content = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(content=content, usage_metadata={"input_tokens": 3, "output_tokens": 5},
                               response_metadata={})


def native_state(*, repairs=2, leads=0):
    study = initial_study("What evidence distinguishes two retrieval architectures?",
                          limits={"max_output_repairs": repairs})
    study = apply_output(study, "frame", {
        "confidence": 0.7, "reasoning": "Separate discovery from evidence assessment.",
        "working_question": "Which evidence actually compares the architectures?",
        "purpose": "Create a conditional research answer.",
        "subquestions": [{"id": "q1", "question": "What do the studies compare?",
                          "contribution": "Find the relevant baseline and outcome."}],
        "answer_criteria": ["Identify the task, baseline, and measured outcome."],
    }, {"kind": "study"}, "seed-frame")
    if leads:
        study = ingest_research_result(study, "web_search", {"query": "seed sources"}, {
            "results": [{"url": url, "title": f"Source {i + 1}"}
                        for i, url in enumerate((URL_A, URL_B)[:leads])],
        }, "seed-search", TARGET)
    return {
        "input": {"question": study["original_question"], "purpose": study["purpose"]},
        "study": study, "steps": [], "current_step": "", "routing_log": [], "loop_counts": {},
        "metadata": {"execution_profile": "inquiry.v1", "instance_id": "node-test",
                     "correlation_id": "node-correlation", "domain": "reasoning_research"},
    }


def plan(*calls):
    return {"calls": list(calls), "rationale": "Read the relevant comparison before drawing a conclusion."}


def search_call():
    return {"tool": "web_search", "arguments": {"query": "matched retrieval comparison", "num": 1},
            "purpose": "Discover a directly relevant comparison."}


def read_call(url=URL_A):
    return {"tool": "read_source", "arguments": {"url": url},
            "purpose": "Inspect the actual task, baseline, and outcome."}


def accept_delta(state, delta):
    """Apply the native node return contract, without mocking node execution."""
    state["steps"].extend(delta["steps"])
    for key, value in delta.items():
        if key != "steps":
            state[key] = value


class NativeInquiryNodeTests(unittest.TestCase):
    def setUp(self):
        reset_governance()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        old_cwd = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, old_cwd)
        self.environment = patch.dict(os.environ, {
            "LLM_CONFIG_PATH": str(ROOT / "llm_config.yaml"),
            "CC_CACHE_ENABLED": "false", "CC_PII_ENABLED": "false",
            "CC_REPLAY_ENABLED": "false", "CC_WEBHOOK_URL": "", "CC_DB_BACKEND": "sqlite",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(reset_governance)

    def node(self, state, tools, invocation_id="retrieve-1"):
        target = {**TARGET, "expected_study_revision": state["study"]["revision"],
                  "frame_version": len(state["study"]["frames"])}
        return create_profile_node(
            step_name=invocation_id, primitive_name="retrieve", params={"strategy": "agentic"},
            output_profile="inquiry.retrieve.v1", target=target, model="fixture-native-profile",
            temperature=0, tool_registry=tools, invocation_id=invocation_id,
        )

    def proof_calls(self):
        return [entry for entry in get_governance().proof()["governance_proof"]["events"]
                if entry["event"] == "llm.call"]

    def test_malformed_plan_is_repaired_before_any_tool_dispatch(self):
        state = native_state()
        dispatched = []
        malformed = {"calls": [{"tool": "web_search", "arguments": {
            "query": "matched retrieval comparison", "purpose": "Wrongly nested purpose.",
        }}], "rationale": "Locate evidence."}

        def corrected():
            self.assertEqual(dispatched, [])
            return plan(search_call())

        model = BoundaryModel([malformed, corrected])
        tools = ToolRegistry()

        def search(args):
            self.assertEqual(len(model.prompts), 2)
            dispatched.append(deepcopy(args))
            return {"results": [{"url": URL_A, "title": "Comparison"}]}

        tools.register("web_search", search, description="Fixture search boundary.")
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            delta = self.node(state, tools)(state)
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(delta["study"]["usage"], {"model_calls": 2, "searches": 1, "reads": 0, "steps": 1})
        self.assertEqual(len(state["metadata"]["inquiry_rejections"]), 1)
        self.assertIn("CORRECTION REQUIRED", model.prompts[1])
        self.assertEqual(len(self.proof_calls()), 2)
        self.assertTrue(get_governance()._initialized)
        self.assertTrue(delta["steps"][0]["accepted"])

    def test_repair_exhaustion_charges_attempts_without_accepting_or_dispatching(self):
        state = native_state(repairs=1)
        before_revision = state["study"]["revision"]
        calls = []
        tools = ToolRegistry()
        tools.register("web_search", lambda args: calls.append(args) or {"results": []},
                       description="Must not be called for an invalid plan.")
        model = BoundaryModel([{"calls": "not-a-list", "rationale": "Find evidence."}, "invalid json"])
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            with self.assertRaisesRegex(ValueError, "bounded correction"):
                self.node(state, tools)(state)
        self.assertEqual(calls, [])
        self.assertEqual(state["study"]["usage"], {"model_calls": 2, "searches": 0, "reads": 0, "steps": 0})
        self.assertEqual(state["study"]["revision"], before_revision)
        self.assertEqual(state["study"]["retrievals"], [])
        self.assertEqual(state["steps"], [])
        self.assertEqual(len(state["metadata"]["inquiry_rejections"]), 2)
        self.assertEqual(len(self.proof_calls()), 2)

    def test_two_batched_reads_survive_revision_stamps_and_keep_full_snapshots(self):
        state = native_state(leads=2)
        initial_revision = state["study"]["revision"]
        bodies = {URL_A: "Source A full text. " * 300, URL_B: "Source B different evidence. " * 240}
        calls = []
        tools = ToolRegistry()

        def read(args):
            calls.append(deepcopy(args))
            return {"url": args["url"], "requested_url": args["url"], "title": "Actual source",
                    "text": bodies[args["url"]], "content_type": "text/plain", "truncated": False}

        tools.register("read_source", read, description="Fixture source reader.")
        model = BoundaryModel([plan(read_call(URL_A), read_call(URL_B))])
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            delta = self.node(state, tools)(state)
        study = delta["study"]
        self.assertEqual(calls, [{"url": URL_A}, {"url": URL_B}])
        self.assertEqual(study["revision"], initial_revision + 3)
        self.assertEqual(study["usage"], {"model_calls": 1, "searches": 0, "reads": 2, "steps": 1})
        for sid, url in (("S1", URL_A), ("S2", URL_B)):
            self.assertEqual(study["sources"][sid]["status"], "read")
            self.assertEqual("".join(p["text"] for p in study["sources"][sid]["passages"]), bodies[url])
        self.assertEqual(len(study["retrievals"][-1]["sources_queried"]), 2)
        self.assertEqual(delta["steps"][0]["target"]["expected_study_revision"], initial_revision)
        self.assertEqual(len(self.proof_calls()), 1)

    def test_duplicate_calls_reuse_recorded_tools_without_extra_external_count(self):
        for tool_name, call in (("web_search", search_call()), ("read_source", read_call())):
            with self.subTest(tool=tool_name):
                state = native_state(leads=1 if tool_name == "read_source" else 0)
                calls = []
                tools = ToolRegistry()

                def external(args):
                    calls.append(deepcopy(args))
                    if tool_name == "web_search":
                        return {"results": [{"url": URL_A}]}
                    return {"url": URL_A, "text": "A relevant comparison with explicit limitations."}

                tools.register(tool_name, external, description="A counted external boundary.")
                model = BoundaryModel([plan(call, call), plan(call)])
                with patch("cognitive_core.engine.llm.create_llm", return_value=model):
                    first = self.node(state, tools, "first-retrieval")(state)
                    accept_delta(state, first)
                    second = self.node(state, tools, "second-retrieval")(state)
                self.assertEqual(len(calls), 1)
                self.assertEqual(second["study"]["usage"]["model_calls"], 2)
                self.assertEqual(second["study"]["usage"]["steps"], 2)
                self.assertEqual(second["study"]["usage"]["searches"], int(tool_name == "web_search"))
                self.assertEqual(second["study"]["usage"]["reads"], int(tool_name == "read_source"))
                self.assertTrue(first["steps"][0]["output"]["sources_queried"][1]["cached"])
                self.assertTrue(second["steps"][0]["output"]["sources_queried"][0]["cached"])
                self.assertEqual(len(second["study"]["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
