"""Local CLI for the Coordinator-owned inquiry execution profile.

This is an adapter, not a research controller. All operation selection,
execution, persistence, clarification, and governance belong to native Core.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import re
import sys
from typing import Sequence


PACK_CONFIG = Path(__file__).parent / "config" / "coordinator.yaml"


def _integer(minimum: int, maximum: int):
    def parse(value):
        result = int(value)
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return result
    return parse


def _runtime_options(parser):
    parser.add_argument("--model", default="default", help="Native provider model ID or configured alias")
    parser.add_argument("--provider", help="Native LLM provider (for example openai or anthropic)")
    parser.add_argument("--quiet", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(prog="cognitive-inquiry", description="Run a native Cognitive Core inquiry.")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Start an inquiry through the native Coordinator")
    start.add_argument("question")
    start.add_argument("--purpose", default="")
    start.add_argument("--context-file", type=Path)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--config", type=Path, default=PACK_CONFIG, help="Coordinator YAML; workflow/domain paths resolve relative to it")
    start.add_argument("--domain", default="reasoning_research")
    start.add_argument("--max-steps", type=_integer(8, 1000))
    start.add_argument("--max-model-calls", type=_integer(16, 3000))
    start.add_argument("--max-searches", type=_integer(0, 200))
    start.add_argument("--max-reads", type=_integer(0, 300))
    _runtime_options(start)
    resume = commands.add_parser("resume", help="Answer clarification work orders or recover a failed local run")
    resume.add_argument("directory", type=Path)
    resume.add_argument("--answer", action="append", default=[], metavar="ID=ANSWER")
    resume.add_argument("--recover-running", action="store_true", help="Assert the previous local process has stopped, then recover its checkpoint")
    _runtime_options(resume)
    for name in ("status", "report"):
        sub = commands.add_parser(name, help=f"Read saved {name} without model or search calls")
        sub.add_argument("directory", type=Path)
    demo = commands.add_parser("demo", help="Run actual native machinery with synthetic model/web boundaries")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--clarify", action="store_true", help="Pause on a native clarification work order; exercise resume separately")
    demo.add_argument("--quiet", action="store_true")
    return parser


def _display(value):
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(value))


def _write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(directory: Path) -> dict:
    path = directory / "run.json"
    if not path.is_file():
        if (directory / "state.json").is_file():
            raise ValueError("This is a legacy InquiryRunner directory. Start a new native inquiry; automatic checkpoint migration is not supported.")
        raise ValueError("No native inquiry run.json was found in this directory.")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("execution_profile") != "inquiry.v1":
        raise ValueError("Unsupported inquiry execution profile.")
    if not (directory / "coordinator.db").is_file():
        raise ValueError("The native Coordinator database is missing.")
    return data


def _store(directory: Path):
    from cognitive_core.coordinator.store import CoordinatorStore
    from cognitive_core.engine.db import SQLiteBackend
    # The first native implementation has one local SQLite authority. Ambient
    # deployment variables must not silently select a remote backend or queue.
    path = directory / "coordinator.db"
    return CoordinatorStore(path, db_backend=SQLiteBackend(str(path)))


def _instance_id(store, manifest):
    identifier = manifest.get("instance_id")
    if identifier:
        return identifier
    # start() is synchronous. If its process was interrupted before writing the
    # convenience manifest, the newly created directory still has one instance.
    instances = store.list_instances()
    if len(instances) != 1:
        raise ValueError("The run has no unambiguous native workflow instance.")
    return instances[0].instance_id


def _coordinator(store, manifest, quiet=False):
    from cognitive_core.coordinator.runtime import Coordinator
    from cognitive_core.coordinator.tasks import SQLiteTaskQueue
    tools = None
    if manifest.get("synthetic_demo"):
        from cognitive_core.inquiry.native_demo import fixture_tools
        tools = fixture_tools()
    return Coordinator(config_path=manifest["config_path"], store=store,
                       task_queue=SQLiteTaskQueue(store.db), tool_registry=tools, verbose=not quiet)


@contextmanager
def _provider(name):
    previous = os.environ.get("LLM_PROVIDER")
    if name:
        os.environ["LLM_PROVIDER"] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = previous


def _boundaries(manifest):
    if manifest.get("synthetic_demo"):
        from cognitive_core.inquiry.native_demo import fixture_model
        return fixture_model(clarification=manifest.get("demo_clarification", False))
    return nullcontext()


def _saved_state(store, manifest):
    identifier = _instance_id(store, manifest)
    instance = store.get_instance(identifier)
    if instance is None:
        raise ValueError("The manifest refers to a missing Coordinator instance.")
    state = store.get_workflow_checkpoint(identifier) or {}
    return instance, state


def _export(directory, instance, state, manifest):
    from cognitive_core.inquiry.native_report import render_report
    manifest["instance_id"] = instance.instance_id
    _write_json(directory / "run.json", manifest)
    # Inspection exports only; neither is ever read by resume().
    _write_json(directory / "state.json", state)
    (directory / "report.md").write_text(render_report(instance, state), encoding="utf-8")


def _finish_run(directory, store, manifest):
    instance, state = _saved_state(store, manifest)
    if manifest.get("synthetic_demo"):
        # Presentation label only. Resume always uses the canonical database.
        state.setdefault("metadata", {})["synthetic_demo"] = True
    _export(directory, instance, state, manifest)
    _status(instance, state, directory)
    return 1 if instance.status.value == "failed" else 0


def _export_native_failure(directory, store, manifest):
    """Expose a durably failed run even when the native API raises to its caller."""
    try:
        instance, _ = _saved_state(store, manifest)
    except ValueError:
        return False
    if instance.status.value != "failed":
        return False
    _finish_run(directory, store, manifest)
    return True


def _status(instance, state, directory):
    study = state.get("study", {})
    print(f"Inquiry: {instance.instance_id}")
    print(f"Coordinator status: {instance.status.value}; research outcome: {study.get('outcome', 'unavailable')}")
    usage = study.get("usage", {})
    print(f"Accepted operations: {usage.get('steps', 0)}; model attempts: {usage.get('model_calls', 0)}; "
          f"searches: {usage.get('searches', 0)}; reads: {usage.get('reads', 0)}")
    if instance.error:
        print(f"Run error: {_display(instance.error)}")
    for identifier, item in study.get("clarifications", {}).items():
        if item.get("active", True) and item.get("answer") is None:
            print(f"Clarification {identifier}: {_display(item['question'])}")
            print(f"  {_display(item['reason'])}")
    if instance.status.value == "suspended" and not any(
        c.get("active", True) and c.get("blocking", True) and c.get("answer") is None
        for c in study.get("clarifications", {}).values()
    ):
        print("A native governance or other work-order review is pending; clarification answers cannot approve it.")
    print(f"Report: {directory / 'report.md'}")


def _start(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("run.json", "coordinator.db", "state.json")):
        raise ValueError("Choose a new output directory; an existing run cannot be overwritten.")
    demo = args.command == "demo"
    manifest = {"execution_profile": "inquiry.v1", "instance_id": None,
                "config_path": str((PACK_CONFIG if demo else args.config).resolve()),
                "domain": "reasoning_research" if demo else args.domain,
                "provider": None if demo else args.provider,
                "synthetic_demo": demo, "demo_clarification": getattr(args, "clarify", False)}
    if demo:
        from cognitive_core.inquiry.native_demo import DEMO_QUESTION, DEMO_PURPOSE
        case = {"question": DEMO_QUESTION, "purpose": DEMO_PURPOSE,
                "context": "This is a synthetic integration demonstration, not live research."}
    else:
        context = args.context_file.read_text(encoding="utf-8-sig") if args.context_file else ""
        if len(context) > 32000:
            raise ValueError("Keep initial context within 32,000 characters for the minimal local inquiry.")
        case = {"question": args.question, "purpose": args.purpose, "context": context,
                "limits": {name: getattr(args, name) for name in
                           ("max_steps", "max_model_calls", "max_searches", "max_reads")
                           if getattr(args, name) is not None}}
    _write_json(directory / "run.json", manifest)
    store = _store(directory)
    try:
        coordinator = _coordinator(store, manifest, args.quiet)
        try:
            with _provider(manifest["provider"]), _boundaries(manifest):
                identifier = coordinator.start("inquiry", manifest["domain"], case,
                                               model="inquiry-synthetic-fixture" if demo else args.model)
        except Exception:
            if _export_native_failure(directory, store, manifest):
                return 1
            raise
        manifest["instance_id"] = identifier
        return _finish_run(directory, store, manifest)
    finally:
        store.close()


def _resume(args):
    directory = args.directory.resolve()
    manifest = _manifest(directory)
    answers = {}
    for value in args.answer:
        identifier, separator, answer = value.partition("=")
        if not separator or not identifier.strip() or not answer.strip() or identifier.strip() in answers:
            raise ValueError("Use one nonempty --answer ID=ANSWER for each clarification ID.")
        answers[identifier.strip()] = answer.strip()
    store = _store(directory)
    try:
        instance, state = _saved_state(store, manifest)
        if set(answers) - set(state.get("study", {}).get("clarifications", {})):
            raise ValueError("An answer refers to an unknown clarification ID; inspect status first.")
        coordinator = _coordinator(store, manifest, args.quiet)
        try:
            with _provider(args.provider or manifest.get("provider")), _boundaries(manifest):
                if answers:
                    for identifier, answer in answers.items():
                        coordinator.fulfill_resource(instance.instance_id, identifier, answer, model=args.model)
                elif instance.status.value in {"failed", "running"}:
                    coordinator.recover_inquiry(instance.instance_id, allow_running=args.recover_running, model=args.model)
                elif instance.status.value != "completed":
                    raise ValueError("Answer the pending clarification with --answer, or resolve the native governance review. A generic resume cannot clear a human wait.")
        except Exception:
            if _export_native_failure(directory, store, manifest):
                return 1
            raise
        return _finish_run(directory, store, manifest)
    finally:
        store.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"start", "demo"}:
            return _start(args)
        if args.command == "resume":
            return _resume(args)
        directory = args.directory.resolve()
        manifest = _manifest(directory)
        store = _store(directory)
        try:
            instance, state = _saved_state(store, manifest)
            if args.command == "status":
                _status(instance, state, directory)
            else:
                from cognitive_core.inquiry.native_report import render_report
                if manifest.get("synthetic_demo"):
                    state.setdefault("metadata", {})["synthetic_demo"] = True
                print(_display(render_report(instance, state)))
        finally:
            store.close()
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Error: {_display(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
