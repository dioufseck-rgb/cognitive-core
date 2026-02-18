"""
Cognitive Core — Spec Validator

Three-layer validation for declarative YAML workflows, domains, and
coordinator configuration. Catches errors that would otherwise surface
as silent LLM failures, missing data, or runtime crashes in production.

Levels:
  1. SYNTACTIC  — Schema: required fields, types, valid enums
  2. SEMANTIC   — Cross-refs: step targets exist, domain params resolve,
                  delegation targets exist, ${} refs are valid
  3. PRAGMATIC  — Runtime: merged config is executable, delegation inputs
                  can resolve, governance tiers are consistent

Usage:
    python -m engine.validate                    # validate everything
    python -m engine.validate --workflow X.yaml   # single workflow
    python -m engine.validate --strict            # treat warnings as errors

From code:
    from engine.validate import validate_all, validate_workflow, validate_domain
    errors = validate_all(project_root)
"""

from __future__ import annotations

import json
import os
import re
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Issue:
    level: str       # "error", "warning", "info"
    layer: str       # "syntactic", "semantic", "pragmatic"
    file: str        # source file
    location: str    # "step:classify_claim", "delegation:fraud_pattern"
    message: str

    def __str__(self):
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.level, "?")
        return f"  {icon} [{self.layer}] {self.file}:{self.location} — {self.message}"


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add(self, level: str, layer: str, file: str, location: str, message: str):
        self.issues.append(Issue(level, layer, file, location, message))

    def merge(self, other: "ValidationResult"):
        self.issues.extend(other.issues)

    def summary(self) -> str:
        lines = []
        if self.valid:
            lines.append(f"✓ Valid ({len(self.warnings)} warnings)")
        else:
            lines.append(f"✗ {len(self.errors)} errors, {len(self.warnings)} warnings")
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

VALID_PRIMITIVES = {
    "retrieve", "classify", "investigate", "think",
    "verify", "generate", "challenge", "act",
}

def _load_required_params() -> dict[str, list[str]]:
    """Load required params from the primitive registry if available."""
    try:
        from registry.primitives import PRIMITIVE_CONFIGS
        return {
            name: cfg["required_params"]
            for name, cfg in PRIMITIVE_CONFIGS.items()
        }
    except ImportError:
        # Fallback when registry isn't importable (no langgraph etc.)
        return {
            "classify": ["categories", "criteria"],
            "investigate": ["question", "scope"],
            "verify": ["rules"],
            "generate": ["requirements", "format", "constraints"],
            "challenge": ["perspective", "threat_model"],
            "retrieve": ["specification"],
            "think": ["instruction"],
            "act": ["actions", "authorization"],
        }

# Lazy-loaded on first use
_REQUIRED_PARAMS: dict[str, list[str]] | None = None

def get_required_params() -> dict[str, list[str]]:
    global _REQUIRED_PARAMS
    if _REQUIRED_PARAMS is None:
        _REQUIRED_PARAMS = _load_required_params()
    return _REQUIRED_PARAMS

VALID_GOVERNANCE_TIERS = {"auto", "spot_check", "gate", "hold"}

VALID_DELEGATION_MODES = {"fire_and_forget", "wait_for_result"}

VALID_OPERATORS = {
    # Currently implemented in PolicyEngine._field_matches
    "exists", "eq", "gte", "contains_any",
    # Future/planned operators
    "not_exists", "neq", "lte", "gt", "lt",
    "contains", "contains_all", "in", "not_in", "matches",
    # Aliases
    "equals", "not_equals", "greater_than", "less_than",
    "greater_equal", "less_equal",
}

VALID_TRANSITION_KEYS = {"when", "goto", "default", "agent_decide"}


# ═══════════════════════════════════════════════════════════════════
# 1. SYNTACTIC — Schema validation
# ═══════════════════════════════════════════════════════════════════

def validate_workflow_syntax(workflow: dict, filepath: str) -> ValidationResult:
    """Validate workflow YAML schema."""
    r = ValidationResult()
    fname = os.path.basename(filepath)

    # Required top-level fields
    if "name" not in workflow:
        r.add("error", "syntactic", fname, "root", "Missing required field 'name'")

    mode = workflow.get("mode", "sequential")

    # Agentic workflows use a different structure
    if mode == "agentic":
        if "goal" not in workflow and "available_primitives" not in workflow:
            r.add("error", "syntactic", fname, "root",
                   "Agentic workflow must have 'goal' or 'available_primitives'")
        return r

    if "steps" not in workflow:
        r.add("error", "syntactic", fname, "root", "Missing required field 'steps'")
        return r

    steps = workflow.get("steps", [])

    if not isinstance(steps, list):
        r.add("error", "syntactic", fname, "root", "'steps' must be a list")
        return r
    if len(steps) == 0:
        r.add("error", "syntactic", fname, "root", "'steps' is empty")
        return r

    seen_names = set()
    for i, step in enumerate(steps):
        loc = f"step[{i}]"

        if not isinstance(step, dict):
            r.add("error", "syntactic", fname, loc, f"Step must be a dict, got {type(step).__name__}")
            continue

        # Name
        name = step.get("name")
        if not name:
            r.add("error", "syntactic", fname, loc, "Missing 'name'")
        elif not isinstance(name, str):
            r.add("error", "syntactic", fname, loc, f"'name' must be string, got {type(name).__name__}")
        elif name in seen_names:
            r.add("error", "syntactic", fname, f"step:{name}", f"Duplicate step name '{name}'")
        else:
            seen_names.add(name)
            loc = f"step:{name}"

        # Primitive
        prim = step.get("primitive")
        if not prim:
            r.add("error", "syntactic", fname, loc, "Missing 'primitive'")
        elif prim not in VALID_PRIMITIVES:
            r.add("error", "syntactic", fname, loc,
                   f"Unknown primitive '{prim}'. Valid: {sorted(VALID_PRIMITIVES)}")
        else:
            # Required params for this primitive
            params = step.get("params", {})
            required = get_required_params().get(prim, [])
            for req in required:
                if req not in params:
                    r.add("error", "syntactic", fname, loc,
                           f"Missing required param '{req}' for '{prim}' primitive")

        # Transitions syntax
        for j, t in enumerate(step.get("transitions", [])):
            if not isinstance(t, dict):
                r.add("error", "syntactic", fname, f"{loc}.transitions[{j}]",
                       f"Transition must be dict, got {type(t).__name__}")
                continue
            unknown = set(t.keys()) - VALID_TRANSITION_KEYS
            if unknown:
                r.add("warning", "syntactic", fname, f"{loc}.transitions[{j}]",
                       f"Unknown transition keys: {unknown}")

            if "agent_decide" in t:
                ad = t["agent_decide"]
                if "options" not in ad:
                    r.add("error", "syntactic", fname, f"{loc}.agent_decide",
                           "agent_decide missing 'options'")
                else:
                    for k, opt in enumerate(ad["options"]):
                        if "step" not in opt:
                            r.add("error", "syntactic", fname,
                                   f"{loc}.agent_decide.options[{k}]", "Missing 'step'")
                        if "description" not in opt:
                            r.add("warning", "syntactic", fname,
                                   f"{loc}.agent_decide.options[{k}]", "Missing 'description'")

        # max_loops
        ml = step.get("max_loops")
        if ml is not None and not isinstance(ml, int):
            r.add("error", "syntactic", fname, loc,
                   f"'max_loops' must be int, got {type(ml).__name__}")

    return r


def validate_domain_syntax(domain: dict, filepath: str) -> ValidationResult:
    """Validate domain YAML schema."""
    r = ValidationResult()
    fname = os.path.basename(filepath)

    if "domain_name" not in domain:
        r.add("error", "syntactic", fname, "root", "Missing 'domain_name'")
    if "governance" not in domain:
        r.add("error", "syntactic", fname, "root", "Missing 'governance'")
    else:
        tier = domain["governance"]
        if tier not in VALID_GOVERNANCE_TIERS:
            r.add("error", "syntactic", fname, "root",
                   f"Invalid governance tier '{tier}'. Valid: {sorted(VALID_GOVERNANCE_TIERS)}")

    return r


def validate_coordinator_syntax(config: dict, filepath: str) -> ValidationResult:
    """Validate coordinator config YAML schema."""
    r = ValidationResult()
    fname = os.path.basename(filepath)

    # Governance tiers
    tiers = config.get("governance_tiers", {})
    if not tiers:
        r.add("warning", "syntactic", fname, "root", "No governance_tiers defined")
    for name, cfg in tiers.items():
        if name not in VALID_GOVERNANCE_TIERS:
            r.add("warning", "syntactic", fname, f"tier:{name}",
                   f"Non-standard tier name '{name}'")

    # Delegations
    for i, d in enumerate(config.get("delegations", [])):
        dname = d.get("name", f"delegation[{i}]")
        loc = f"delegation:{dname}"

        if "name" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'name'")
        if "conditions" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'conditions'")
        if "target_workflow" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'target_workflow'")
        if "target_domain" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'target_domain'")
        if "contract" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'contract'")
        if "inputs" not in d:
            r.add("error", "syntactic", fname, loc, "Missing 'inputs'")

        mode = d.get("mode", "fire_and_forget")
        if mode not in VALID_DELEGATION_MODES:
            r.add("error", "syntactic", fname, loc,
                   f"Invalid mode '{mode}'. Valid: {sorted(VALID_DELEGATION_MODES)}")

        # Condition operators
        for j, cond in enumerate(d.get("conditions", [])):
            op = cond.get("operator")
            if op and op not in VALID_OPERATORS:
                r.add("error", "syntactic", fname, f"{loc}.conditions[{j}]",
                       f"Unknown operator '{op}'. Valid: {sorted(VALID_OPERATORS)}")

    # Contracts
    for name, c in config.get("contracts", {}).items():
        loc = f"contract:{name}"
        if not isinstance(c, dict):
            r.add("error", "syntactic", fname, loc, f"Contract must be a dict, got {type(c).__name__}")
            continue
        if "request" not in c and "inputs" not in c:
            r.add("error", "syntactic", fname, loc, "Missing 'request' (or 'inputs') spec")
        if "response" not in c and "outputs" not in c:
            r.add("warning", "syntactic", fname, loc, "Missing 'response' (or 'outputs') spec")
        if "version" not in c:
            r.add("warning", "syntactic", fname, loc, "Missing 'version'")

    return r


# ═══════════════════════════════════════════════════════════════════
# 2. SEMANTIC — Cross-reference validation
# ═══════════════════════════════════════════════════════════════════

def validate_workflow_semantics(workflow: dict, filepath: str) -> ValidationResult:
    """Validate internal cross-references within a workflow."""
    r = ValidationResult()
    fname = os.path.basename(filepath)
    steps = workflow.get("steps", [])
    step_names = {s["name"] for s in steps if "name" in s}

    for step in steps:
        name = step.get("name", "?")
        loc = f"step:{name}"

        # Transition targets must reference valid steps
        for t in step.get("transitions", []):
            target = t.get("goto")
            if target and target != "__end__" and target not in step_names:
                r.add("error", "semantic", fname, loc,
                       f"Transition target '{target}' not found in steps: {sorted(step_names)}")

            default = t.get("default")
            if default and default != "__end__" and default not in step_names:
                r.add("error", "semantic", fname, loc,
                       f"Default target '{default}' not found in steps: {sorted(step_names)}")

            for opt in t.get("agent_decide", {}).get("options", []):
                s = opt.get("step", "")
                if s and s != "__end__" and s not in step_names:
                    r.add("error", "semantic", fname, loc,
                           f"Agent option target '{s}' not found in steps: {sorted(step_names)}")

        # loop_fallback must be a valid step
        fallback = step.get("loop_fallback")
        if fallback and fallback != "__end__" and fallback not in step_names:
            r.add("error", "semantic", fname, loc,
                   f"loop_fallback '{fallback}' not found in steps: {sorted(step_names)}")

        # ${step_name.field} references in params should reference prior steps
        params = step.get("params", {})
        for pk, pv in params.items():
            if not isinstance(pv, str):
                continue
            # Find ${step_name.field} refs (not ${domain.*} or ${input.*})
            refs = re.findall(r'\$\{([^}]+)\}', pv)
            # Known runtime prefixes that aren't step names
            known_prefixes = {"domain.", "input.", "previous.", "_last_",
                              "_loop_count", "_delegations"}
            for ref in refs:
                if any(ref.startswith(p) for p in known_prefixes):
                    continue
                ref_step = ref.split(".")[0]
                if ref_step not in step_names:
                    r.add("warning", "semantic", fname, f"{loc}.params.{pk}",
                           f"Reference '${{" + ref + "}}' — step '{ref_step}' not found. "
                           f"Will resolve to empty at runtime.")

    # Unreachable steps: check if any path leads to this step.
    # In sequential workflows, step[i] implicitly flows to step[i+1]
    # when it has no transitions or only a default transition.
    reachable = {steps[0]["name"]} if steps else set()

    for i, step in enumerate(steps):
        sname = step.get("name", "")
        transitions = step.get("transitions", [])

        # Explicit transition targets
        for t in transitions:
            if t.get("goto"):
                reachable.add(t["goto"])
            if t.get("default"):
                reachable.add(t["default"])
            for opt in t.get("agent_decide", {}).get("options", []):
                if opt.get("step"):
                    reachable.add(opt["step"])

        if step.get("loop_fallback"):
            reachable.add(step["loop_fallback"])

        # Implicit sequential flow: step[i] flows to step[i+1] when:
        #   - No transitions at all (bare sequential step)
        #   - Has conditional transitions (when/agent_decide) — the implicit
        #     next is the fallback if no condition matches
        # Does NOT flow implicitly when:
        #   - Has only {default: X} — that's an explicit redirect
        has_conditionals = any(
            "when" in t or "agent_decide" in t for t in transitions
        )
        has_explicit_default = any("default" in t for t in transitions)
        no_transitions = len(transitions) == 0

        if i + 1 < len(steps):
            next_name = steps[i + 1].get("name", "")
            if next_name:
                if no_transitions or has_conditionals:
                    reachable.add(next_name)

    unreachable = step_names - reachable - {"__end__"}
    for u in unreachable:
        r.add("warning", "semantic", fname, f"step:{u}",
               "Step is unreachable — no transition or agent route leads to it")

    return r


def validate_domain_against_workflow(
    domain: dict, domain_path: str,
    workflow: dict, workflow_path: str,
) -> ValidationResult:
    """Validate that domain provides everything the workflow references."""
    r = ValidationResult()
    dfname = os.path.basename(domain_path)
    wfname = os.path.basename(workflow_path)

    steps = workflow.get("steps", [])
    for step in steps:
        name = step.get("name", "?")
        params = step.get("params", {})
        for pk, pv in params.items():
            if not isinstance(pv, str):
                continue
            domain_refs = re.findall(r'\$\{domain\.([^}]+)\}', pv)
            for ref in domain_refs:
                parts = ref.split(".")
                obj = domain
                resolved = True
                for p in parts:
                    if isinstance(obj, dict):
                        obj = obj.get(p)
                        if obj is None:
                            resolved = False
                            break
                    else:
                        resolved = False
                        break
                if not resolved:
                    r.add("error", "semantic", dfname, f"step:{name}.params.{pk}",
                           f"Domain reference '${{domain.{ref}}}' does not resolve. "
                           f"Workflow '{wfname}' expects this but domain '{dfname}' "
                           f"doesn't define it. The LLM will receive '[domain.{ref} not found]'.")

    return r


# ═══════════════════════════════════════════════════════════════════
# 3. PRAGMATIC — Runtime conformance
# ═══════════════════════════════════════════════════════════════════

def validate_delegation_targets(
    config: dict, config_path: str,
    workflows: dict[str, dict],
    domains: dict[str, dict],
) -> ValidationResult:
    """Validate that delegation targets reference real workflows/domains."""
    r = ValidationResult()
    fname = os.path.basename(config_path)
    wf_names = set(workflows.keys())
    domain_names = set(domains.keys())

    for d in config.get("delegations", []):
        dname = d.get("name", "?")
        loc = f"delegation:{dname}"

        tw = d.get("target_workflow", "")
        if tw and tw not in wf_names:
            r.add("error", "pragmatic", fname, loc,
                   f"target_workflow '{tw}' not found. Available: {sorted(wf_names)}")

        td = d.get("target_domain", "")
        if td and td not in domain_names:
            r.add("error", "pragmatic", fname, loc,
                   f"target_domain '{td}' not found. Available: {sorted(domain_names)}")

        # Check delegation inputs for tool data when target has retrieve step
        target_wf = workflows.get(tw, {})
        has_retrieve = any(
            s.get("primitive") == "retrieve"
            for s in target_wf.get("steps", [])
        )
        inputs = d.get("inputs", {})
        has_tool_data = any(
            k.startswith("get_") for k in inputs
        )
        if has_retrieve and not has_tool_data:
            r.add("warning", "pragmatic", fname, loc,
                   f"Target workflow '{tw}' has a retrieve step but delegation "
                   f"inputs contain no get_* tool data. Handler's retrieve will "
                   f"find no tools and the LLM may hang.")

        # Check input ${source.*} refs are structurally valid
        for field_name, ref in inputs.items():
            if isinstance(ref, str) and ref.startswith("${source."):
                path = ref[len("${source."):-1] if ref.endswith("}") else ref
                valid_prefixes = ("input.", "last_", "any_", "final_output")
                if not any(path.startswith(p) for p in valid_prefixes):
                    r.add("warning", "pragmatic", fname, f"{loc}.inputs.{field_name}",
                           f"Source ref '{ref}' uses unusual path '{path}'. "
                           f"Expected prefixes: {valid_prefixes}")

    return r


def validate_domain_governance_consistency(
    domains: dict[str, dict],
    coord_config: dict,
    coord_path: str,
) -> ValidationResult:
    """Validate that every domain's governance tier exists in coordinator config."""
    r = ValidationResult()
    fname = os.path.basename(coord_path)
    coord_tiers = set(coord_config.get("governance_tiers", {}).keys())

    for dname, dcfg in domains.items():
        tier = dcfg.get("governance", "")
        if tier and tier not in coord_tiers:
            r.add("error", "pragmatic", fname, f"domain:{dname}",
                   f"Domain declares governance='{tier}' but coordinator "
                   f"has no tier config for it. Defined tiers: {sorted(coord_tiers)}")

    return r


def validate_workflow_domain_pairs(
    workflows: dict[str, dict],
    domains: dict[str, dict],
) -> ValidationResult:
    """Validate that every domain references a workflow that exists."""
    r = ValidationResult()
    wf_names = set(workflows.keys())

    for dname, dcfg in domains.items():
        wf = dcfg.get("workflow", "")
        if wf and wf not in wf_names:
            r.add("error", "pragmatic", dname + ".yaml", "root",
                   f"Domain references workflow '{wf}' but it doesn't exist. "
                   f"Available: {sorted(wf_names)}")

    return r


# ═══════════════════════════════════════════════════════════════════
# Orchestrator: validate everything
# ═══════════════════════════════════════════════════════════════════

def validate_all(project_root: str | Path) -> ValidationResult:
    """Run all validation layers across the entire project."""
    root = Path(project_root)
    result = ValidationResult()

    # Load everything
    workflows: dict[str, dict] = {}
    domains: dict[str, dict] = {}
    wf_paths: dict[str, str] = {}
    domain_paths: dict[str, str] = {}

    wf_dir = root / "workflows"
    dom_dir = root / "domains"
    coord_path = root / "coordinator" / "config.yaml"

    for f in sorted(wf_dir.glob("*.yaml")) if wf_dir.exists() else []:
        with open(f) as fh:
            wf = yaml.safe_load(fh)
        name = wf.get("name", f.stem)
        workflows[name] = wf
        wf_paths[name] = str(f)

    for f in sorted(dom_dir.glob("*.yaml")) if dom_dir.exists() else []:
        with open(f) as fh:
            dom = yaml.safe_load(fh)
        name = dom.get("domain_name", f.stem)
        domains[name] = dom
        domain_paths[name] = str(f)

    coord_config = {}
    if coord_path.exists():
        with open(coord_path) as fh:
            coord_config = yaml.safe_load(fh)

    # ── Layer 1: Syntactic ──
    for name, wf in workflows.items():
        result.merge(validate_workflow_syntax(wf, wf_paths[name]))
    for name, dom in domains.items():
        result.merge(validate_domain_syntax(dom, domain_paths[name]))
    if coord_config:
        result.merge(validate_coordinator_syntax(coord_config, str(coord_path)))

    # ── Layer 2: Semantic ──
    for name, wf in workflows.items():
        result.merge(validate_workflow_semantics(wf, wf_paths[name]))

    # Cross-reference: domain provides what workflow needs
    for dname, dom in domains.items():
        wf_name = dom.get("workflow", "")
        if wf_name in workflows:
            result.merge(validate_domain_against_workflow(
                dom, domain_paths[dname],
                workflows[wf_name], wf_paths[wf_name],
            ))

    # ── Layer 3: Pragmatic ──
    if coord_config:
        result.merge(validate_delegation_targets(
            coord_config, str(coord_path), workflows, domains,
        ))
        result.merge(validate_domain_governance_consistency(
            domains, coord_config, str(coord_path),
        ))
    result.merge(validate_workflow_domain_pairs(workflows, domains))

    return result


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Core Spec Validator")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--workflow", help="Validate single workflow file")
    parser.add_argument("--domain", help="Validate single domain file")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        # Try relative to script location
        root = Path(__file__).resolve().parent.parent
    if not (root / "workflows").exists():
        root = Path(__file__).resolve().parent.parent

    result = validate_all(root)

    if args.strict:
        # Promote warnings to errors
        for issue in result.issues:
            if issue.level == "warning":
                issue.level = "error"

    print(result.summary())

    if not result.valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
