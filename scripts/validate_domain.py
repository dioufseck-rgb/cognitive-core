#!/usr/bin/env python3
"""
Validate a domain YAML against its contract spec.

Usage:
    python scripts/validate_domain.py domains/electronics_return.yaml
    python scripts/validate_domain.py domains/scaffold_domain.yaml --verbose
    python scripts/validate_domain.py --all
"""

import argparse
import importlib.util
import os
import sys
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Direct import to avoid engine/__init__.py dependency chain
dc_path = os.path.join(BASE, "engine", "domain_contract.py")
dc_spec = importlib.util.spec_from_file_location("engine.domain_contract", dc_path)
dc_mod = importlib.util.module_from_spec(dc_spec)
dc_mod.__package__ = "engine"
sys.modules["engine.domain_contract"] = dc_mod
dc_spec.loader.exec_module(dc_mod)
DomainContract = dc_mod.DomainContract


def validate_domain(path: str, verbose: bool = False) -> bool:
    """Validate a single domain file. Returns True if valid."""
    name = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    try:
        contract = DomainContract.from_yaml(path)
    except Exception as e:
        print(f"  ❌ PARSE ERROR: {e}")
        return False

    if not contract.has_contracts:
        print(f"  ℹ️  Legacy domain (no schema-first contracts)")
        print(f"     Upgrade to v2 format for auto-validation.")
        return True

    print(contract.summary())

    issues = []

    # Check tool contracts
    for tool_name, tc in contract.tool_contracts.items():
        if not tc.required_keys:
            issues.append(f"Tool '{tool_name}' has no required_keys")

    # Check artifact contracts
    for art_name, ac in contract.artifact_contracts.items():
        if not ac.required_keys:
            issues.append(f"Artifact '{art_name}' has no required_keys")
        # Check that artifact is referenced by a step
        referenced = any(
            s.artifact_schema == art_name
            for s in contract.step_specs.values()
        )
        if not referenced:
            issues.append(
                f"Artifact '{art_name}' not referenced by any step's artifact_schema"
            )

    # Check step specs
    for step_name, spec in contract.step_specs.items():
        if spec.primitive == "generate" and not spec.artifact_schema:
            issues.append(f"Generate step '{step_name}' has no artifact_schema")
        if spec.confidence_floor is not None:
            if spec.confidence_floor < 0 or spec.confidence_floor > 1:
                issues.append(
                    f"Step '{step_name}' confidence_floor {spec.confidence_floor} "
                    f"out of range [0, 1]"
                )

    # Check coherence tables reference valid steps
    for step_name, spec in contract.step_specs.items():
        if spec.coherence_with:
            if spec.coherence_with not in contract.step_specs:
                issues.append(
                    f"Step '{step_name}' coherence_with references "
                    f"unknown step '{spec.coherence_with}'"
                )
            if not spec.coherence_table:
                issues.append(
                    f"Step '{step_name}' has coherence_with but no coherence_table"
                )

    # Check enums referenced by output_schema
    for step_name, spec in contract.step_specs.items():
        if spec.output_schema:
            for field, type_ref in spec.output_schema.items():
                if isinstance(type_ref, str) and type_ref.startswith("enum:"):
                    enum_name = type_ref[5:]
                    if enum_name not in contract.enums:
                        issues.append(
                            f"Step '{step_name}' output_schema references "
                            f"unknown enum '{enum_name}'"
                        )

    # Check consistency rules reference real step paths
    for rule in contract.consistency_rules:
        if not rule.rule.strip():
            issues.append("Empty consistency rule")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for i in issues:
            print(f"    ⚠  {i}")
    else:
        print(f"\n  ✅ Domain contract is valid")

    if verbose:
        # Show what invariants would be auto-generated
        print(f"\n  Auto-generated invariants:")
        if contract.tool_contracts:
            tools = ", ".join(contract.tool_contracts.keys())
            print(f"    I0: Retrieve completeness for [{tools}]")
        for step_name, spec in contract.step_specs.items():
            if spec.artifact_schema:
                ac = contract.artifact_contracts.get(spec.artifact_schema)
                if ac:
                    print(f"    I3: {step_name} artifact must have {ac.required_keys}")
                    if ac.redactions:
                        print(f"        Redactions: {ac.redactions}")
            if spec.confidence_floor is not None:
                print(f"    I4: {step_name} confidence >= {spec.confidence_floor}")
        for rule in contract.consistency_rules:
            desc = rule.description or rule.rule[:60]
            print(f"    I7: [{rule.severity}] {desc}")
        for step_name, spec in contract.step_specs.items():
            if spec.coherence_table:
                print(f"    I7: {step_name} coherence with {spec.coherence_with}: "
                      f"{spec.coherence_table}")

    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Validate domain contracts")
    parser.add_argument("path", nargs="?", help="Path to domain YAML")
    parser.add_argument("--all", action="store_true", help="Validate all domains")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show auto-generated invariants")
    args = parser.parse_args()

    if args.all:
        domains_dir = os.path.join(BASE, "domains")
        files = sorted(
            os.path.join(domains_dir, f)
            for f in os.listdir(domains_dir)
            if f.endswith(".yaml")
        )
    elif args.path:
        files = [args.path]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for f in files:
        results[f] = validate_domain(f, verbose=args.verbose)

    v2_count = 0
    v1_count = 0
    for f in files:
        try:
            contract = DomainContract.from_yaml(f)
            if contract.has_contracts:
                v2_count += 1
            else:
                v1_count += 1
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  Summary: {len(files)} domains, {v2_count} v2 (schema-first), "
          f"{v1_count} v1 (legacy)")
    valid = sum(1 for v in results.values() if v)
    print(f"  Valid: {valid}/{len(files)}")
    print(f"{'='*60}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
