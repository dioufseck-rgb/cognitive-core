"""Native progress and recovery contracts, with only model/web boundaries faked.

The lifecycle regression recreates a stalled study with twelve accepted
operations, three searches and no reads, then resumes its original saved
configuration through a fresh Coordinator and a real source read.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.coordinator.types import InstanceStatus
from cognitive_core.engine.agentic_devs import (
    AgenticWorkflowExecutor, InquiryExecutionError, OrchestratorStep,
)
from cognitive_core.engine.composer import load_three_layer
from cognitive_core.engine.devs_executor import StepInterrupt
from cognitive_core.engine.governance import GovernancePipeline, get_governance, reset_governance
from cognitive_core.engine.semantic_cache import SemanticCache
from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.native_demo import (
    DEMO_PURPOSE, DEMO_QUESTION, DEMO_TEXT, DEMO_URL, FixtureModel, fixture_tools,
)
from cognitive_core.inquiry.native_profiles import apply_output, initial_study, ingest_research_result
from cognitive_core.inquiry.native_report import render_report


ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "cognitive_core" / "inquiry" / "config"
STUDY = {"kind": "study"}
QUESTION = {"kind": "question", "question_id": "q1"}
REPORT = {"kind": "report"}


def workflow_config(*, legacy=False):
    configuration, _ = load_three_layer(
        PACK / "workflows" / "inquiry.yaml", PACK / "domains" / "reasoning_research.yaml",
    )
    if legacy:
        for key in ("discover_sources", "read_sources"):
            configuration["primitive_configs"].pop(key)
        configuration["primitive_configs"]["research_sources"] = {
            "primitive": "retrieve", "output_profile": "inquiry.retrieve.v1",
            "params": {"instructions": "Search or read to address the target's evidence gap."},
        }
    return configuration


def decision(key, primitive, target=STUDY, reasoning="Resolve the next material evidence gap."):
    return {"action": "invoke", "params_key": key, "primitive": primitive,
            "target": copy.deepcopy(target), "reasoning": reasoning}


def current_state(configuration):
    return {
        "input": {"question": DEMO_QUESTION, "purpose": DEMO_PURPOSE},
        "steps": [], "current_step": "", "loop_counts": {}, "routing_log": [],
        "metadata": {"execution_profile": "inquiry.v1", "instance_id": "progress-fixture",
                     "domain": "reasoning_research", "effective_config": configuration},
        "study": initial_study(DEMO_QUESTION, DEMO_PURPOSE),
    }


def response(primitive, context=None):
    return FixtureModel().response(primitive, context or {})


def accept(current, key, primitive, output, target=STUDY):
    """Build valid historical reducer results without pretending to call a model."""
    invocation = f"history-{len(current['steps'])}"
    current["study"] = apply_output(current, primitive, output, target, invocation)
    current["study"]["usage"]["steps"] += 1
    current["steps"].append({
        "step_name": invocation, "params_key": key, "primitive": primitive,
        "target": copy.deepcopy(target), "output": output, "invocation_id": invocation,
        "output_profile": current["metadata"]["effective_config"]["primitive_configs"][key]["output_profile"],
    })


def add_lead(current):
    current["study"] = ingest_research_result(
        current, "web_search", {"query": "fictional policy", "num": 1},
        {"results": [{"url": DEMO_URL, "title": "Fictional policy", "snippet": "Two requirements."}]},
        "historical-search", STUDY,
    )
    current["study"]["usage"]["searches"] += 1


class ScriptedModel:
    model_name = "progress-fixture"

    def __init__(self, script):
        self.script = copy.deepcopy(script)
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[0].content
        self.prompts.append(prompt)
        operation = prompt.split("OPERATION: ", 1)[1].splitlines()[0].strip()
        if not self.script:
            raise AssertionError(f"Unexpected model operation: {operation}")
        expected, output = self.script.pop(0)
        if operation != expected:
            raise AssertionError(f"Expected {expected}, got {operation}")
        return SimpleNamespace(content=json.dumps(output), usage_metadata={}, response_metadata={})


class NativeInquiryProgressTests(unittest.TestCase):
    @contextmanager
    def boundary(self, script, *, semantic=False):
        model = ScriptedModel(script)
        governance = GovernancePipeline()
        governance._initialized = True
        # All orchestration prompts intentionally collide in the similarity
        # layer. Repairs must bypass it while retaining protected-call hooks.
        embed = (lambda text: [1., 0.] if "OPERATION: orchestrate" in text else [0., 1.]) if semantic else None
        governance._cache = SemanticCache(enabled=True, embed_fn=embed)
        try:
            with patch("cognitive_core.engine.llm.create_llm", return_value=model), \
                 patch("cognitive_core.engine.llm._load_config", return_value={}), \
                 patch("cognitive_core.engine.governance.get_governance", return_value=governance):
                yield model, governance
        finally:
            governance.close()

    def test_availability_uses_all_history_without_resetting_exhausted_target(self):
        config = workflow_config(legacy=True)
        current = current_state(config)
        accept(current, "frame_study", "frame", response("frame"))
        for _ in range(6):
            accept(current, "research_sources", "retrieve", {
                "plan": {"calls": [], "rationale": "Legacy empty retrieval decision."},
            })
        for _ in range(5):
            accept(current, "reflect_progress", "reflect", response("reflect"))
        original = copy.deepcopy(current)
        options = OrchestratorStep(config, current, "fixture", 0)._inquiry_operation_options()
        spent = next(row for row in options["blocked"]
                     if row["params_key"] == "research_sources" and row["target"] == STUDY)
        self.assertEqual((spent["accepted_count"], spent["remaining_repeats"]), (6, 0))
        self.assertIn("repeat limit", spent["reason"])
        available = next(row for row in options["available"]
                         if row["params_key"] == "research_sources" and row["target"] == QUESTION)
        self.assertEqual((available["accepted_count"], available["remaining_repeats"]), (0, 6))
        self.assertEqual(current, original)

    def test_real_devs_repairs_repeated_exhausted_choice_with_semantic_cache_enabled(self):
        config = workflow_config(legacy=True)
        current = current_state(config)
        accept(current, "frame_study", "frame", response("frame"))
        add_lead(current)
        for _ in range(6):
            accept(current, "research_sources", "retrieve", {
                "plan": {"calls": [], "rationale": "Prior legacy decision."},
            })
        previous = copy.deepcopy(current)
        bad = decision("research_sources", "retrieve")
        purpose = "Read the two policy clauses to test whether a receipt alone is sufficient."
        good = decision("research_sources", "retrieve", QUESTION, purpose)
        plan = {"calls": [{"tool": "read_source", "arguments": {"url": DEMO_URL},
                           "purpose": purpose}], "rationale": purpose}
        script = [("orchestrate", bad), ("orchestrate", bad), ("orchestrate", good), ("retrieve", plan)]
        with self.boundary(script, semantic=True) as (model, gov):
            result = AgenticWorkflowExecutor(config, fixture_tools(), model="fixture").resume(
                current, step_callback=lambda name, output, state: StepInterrupt("inspect read", name),
            )
        self.assertFalse(model.script)
        self.assertEqual(len(model.prompts), 4)
        self.assertEqual(len(set(model.prompts[:3])), 3)
        self.assertIn("AVAILABLE OPERATION/TARGET PAIRS", model.prompts[0])
        self.assertIn("research_sources", model.prompts[1])
        self.assertIn("repeat limit", model.prompts[1])
        self.assertIn(purpose, model.prompts[-1])
        self.assertEqual(result.final_state["steps"][:-1], previous["steps"])
        self.assertEqual(result.final_state["steps"][-1]["operation_purpose"], purpose)
        self.assertEqual(result.final_state["study"]["usage"]["model_calls"], 4)
        self.assertEqual(result.final_state["study"]["usage"]["reads"], 1)
        self.assertEqual(result.final_state["study"]["usage"]["searches"], 1)
        self.assertEqual(sum(item["event"] == "cache.bypass" for item in gov._proof_ledger), 2)

    def test_read_only_operation_requires_known_uncached_url_and_budget(self):
        config = workflow_config()
        current = current_state(config)
        accept(current, "frame_study", "frame", response("frame"))
        orchestrator = OrchestratorStep(config, current, "fixture", 0)

        def read_available():
            return any(row["params_key"] == "read_sources" for row in
                       orchestrator._inquiry_operation_options()["available"])

        self.assertFalse(read_available())
        add_lead(current)
        self.assertTrue(read_available())
        current["study"]["usage"]["reads"] = current["study"]["limits"]["max_reads"]
        self.assertFalse(read_available())
        current["study"]["usage"]["reads"] = 0
        key = hashlib.sha256(json.dumps(["read_source", {"url": DEMO_URL}], sort_keys=True).encode()).hexdigest()
        current["study"]["tool_cache"][key] = {"source_ids": ["S1"]}
        self.assertFalse(read_available())

    def test_invalid_retrieval_tool_policy_fails_before_model_construction(self):
        for allowed in ([], ["execute_code"], "read_source", ["read_source", "read_source"]):
            with self.subTest(allowed=allowed):
                config = workflow_config()
                config["primitive_configs"]["read_sources"]["params"]["allowed_tools"] = allowed
                with self.assertRaises((InquiryExecutionError, ValueError)):
                    OrchestratorStep(config, current_state(config), "fixture", 0)

    def test_option_catalog_preserves_exact_report_challenge_and_govern_gate(self):
        config = workflow_config()
        current = current_state(config)
        accept(current, "frame_study", "frame", response("frame"))
        add_lead(current)
        current["study"] = ingest_research_result(
            current, "read_source", {"url": DEMO_URL},
            {"url": DEMO_URL, "text": DEMO_TEXT, "truncated": False}, "historical-read", QUESTION,
        )
        accept(current, "investigate_question", "investigate", response("investigate", {"findings": [{}]}), QUESTION)
        accept(current, "synthesize_study", "deliberate", response("deliberate"))
        accept(current, "write_report", "generate", response("generate"))
        orchestrator = OrchestratorStep(config, current, "fixture", 0)
        choices = orchestrator._inquiry_operation_options()["available"]
        self.assertEqual([(c["primitive"], c["target"]) for c in choices], [("challenge", REPORT)])
        accept(current, "challenge_answer", "challenge", response("challenge", {"target": REPORT}), REPORT)
        self.assertTrue(any(c["primitive"] == "govern" for c in
                            orchestrator._inquiry_operation_options()["available"]))
        accept(current, "investigate_question", "investigate", response("investigate", {"findings": [{}]}), QUESTION)
        self.assertFalse(any(c["primitive"] == "govern" for c in
                             orchestrator._inquiry_operation_options()["available"]))


class NativeInquiryProgressRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        old_cwd = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, old_cwd)
        self.env = patch.dict(os.environ, {
            "CC_CACHE_ENABLED": "true", "CC_PII_ENABLED": "false", "CC_REPLAY_ENABLED": "false",
            "CC_WEBHOOK_URL": "", "CC_DB_BACKEND": "sqlite",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        reset_governance()
        self.addCleanup(reset_governance)
        self.coordinators = []
        self.addCleanup(lambda: [coordinator.store.close() for coordinator in self.coordinators])
        self.searches, self.reads = [], []
        self.tools = ToolRegistry()

        def search(args):
            self.searches.append(args["query"])
            return {"results": [{"url": DEMO_URL, "title": "Fictional policy", "snippet": "Two requirements."}]}

        def read(args):
            self.reads.append(args["url"])
            return {"url": args["url"], "text": DEMO_TEXT, "truncated": False}

        self.tools.register("web_search", search, description="Offline search fixture")
        self.tools.register("read_source", read, description="Offline source fixture")

    def coordinator(self, pack):
        instance = Coordinator(config_path=pack / "coordinator.yaml", db_path=self.directory / "study.db",
                               tool_registry=self.tools, verbose=False)
        self.coordinators.append(instance)
        return instance

    def test_saved_legacy_study_recovers_from_repeat_failure_without_reset_or_yaml_replacement(self):
        legacy_pack = self.directory / "legacy-pack"
        shutil.copytree(PACK, legacy_pack)
        domain_path = legacy_pack / "domains" / "reasoning_research.yaml"
        domain = yaml.safe_load(domain_path.read_text(encoding="utf-8"))
        domain["primitive_configs"] = workflow_config(legacy=True)["primitive_configs"]
        domain_path.write_text(yaml.safe_dump(domain), encoding="utf-8")
        first = self.coordinator(legacy_pack)
        script = [("orchestrate", decision("frame_study", "frame")), ("frame", response("frame"))]
        for query in ["policy receipt", "policy prior approval", "policy exceptions"] * 2:
            script += [
                ("orchestrate", decision("research_sources", "retrieve")),
                ("retrieve", {"calls": [{"tool": "web_search", "arguments": {"query": query, "num": 1},
                                         "purpose": "Discover the policy."}], "rationale": "Discover the policy."}),
            ]
        for _ in range(5):
            script += [("orchestrate", decision("reflect_progress", "reflect")), ("reflect", response("reflect"))]
        spent = decision("research_sources", "retrieve")
        model = ScriptedModel(script + [("orchestrate", spent)] * 3)
        with patch("cognitive_core.engine.llm.create_llm", return_value=model):
            with self.assertRaisesRegex(InquiryExecutionError, "repeat limit"):
                first.start("inquiry", "reasoning_research", {"question": DEMO_QUESTION, "purpose": DEMO_PURPOSE})
        self.assertFalse(model.script)
        instance = first.store.list_instances()[0]
        self.assertEqual(instance.status, InstanceStatus.FAILED)
        saved = first.checkpoint(instance.instance_id)
        self.assertEqual(saved["study"]["usage"], {"steps": 12, "model_calls": 27, "searches": 3, "reads": 0})
        self.assertEqual(len(self.searches), 3)
        self.assertIn("1 cached", render_report(instance, saved))
        self.assertIn("repeat limit", render_report(instance, saved))
        first.store.close()
        self.coordinators.remove(first)
        reset_governance()

        # The new Coordinator loads today's default pack; native recovery must
        # nevertheless retain this study's old combined research operation.
        recovered = self.coordinator(PACK)
        purpose = "Inspect the previously discovered clauses before revising the answer."
        resume_script = [("orchestrate", spent)] * 2 + [
            ("orchestrate", decision("research_sources", "retrieve", QUESTION, purpose)),
            ("retrieve", {"calls": [{"tool": "read_source", "arguments": {"url": DEMO_URL},
                                     "purpose": purpose}], "rationale": purpose}),
            ("orchestrate", decision("investigate_question", "investigate", QUESTION)),
            ("investigate", response("investigate", {"findings": [{}]})),
            ("orchestrate", decision("synthesize_study", "deliberate")), ("deliberate", response("deliberate")),
            ("orchestrate", decision("write_report", "generate")), ("generate", response("generate")),
            ("orchestrate", decision("challenge_answer", "challenge", REPORT)),
            ("challenge", response("challenge", {"target": REPORT})),
            ("orchestrate", decision("govern_inquiry", "govern", REPORT)), ("govern", response("govern")),
        ]
        resumed_model = ScriptedModel(resume_script)
        with patch("cognitive_core.engine.llm.create_llm", return_value=resumed_model):
            recovered.recover_inquiry(instance.instance_id)
        self.assertFalse(resumed_model.script)
        final = recovered.checkpoint(instance.instance_id)
        self.assertEqual(recovered.get_instance(instance.instance_id).status, InstanceStatus.COMPLETED)
        self.assertEqual(final["steps"][:12], saved["steps"])
        self.assertEqual(final["study"]["frames"], saved["study"]["frames"])
        self.assertEqual(final["metadata"]["effective_config"], saved["metadata"]["effective_config"])
        self.assertEqual(final["study"]["usage"], {"steps": 18, "model_calls": 41, "searches": 3, "reads": 1})
        self.assertEqual(len(self.searches), 3)
        self.assertEqual(self.reads, [DEMO_URL])
        self.assertEqual(final["steps"][12]["operation_purpose"], purpose)
        self.assertEqual([s["primitive"] for s in final["steps"][-3:]], ["generate", "challenge", "govern"])
        self.assertEqual(final["study"]["final_review"]["basis"], final["study"]["governance"]["basis"])
        self.assertTrue(recovered.store.verify_ledger(instance.instance_id)["valid"])
        report = render_report(recovered.get_instance(instance.instance_id), final)
        self.assertIn("research_sources | q1 | 0 searches; 1 reads", report)
        self.assertTrue(get_governance()._cache.enabled)


if __name__ == "__main__":
    unittest.main()
