"""Command-line access to saved, research-assisted inquiries.

Heavy runtime imports live inside the commands that need them. Inspecting a
checkpoint or reading its report never requires credentials or an LLM SDK.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from cognitive_core.inquiry.models import InquiryEvent, InquiryState
    from cognitive_core.inquiry.store import InquiryStore


class CLIError(Exception):
    """A user-correctable error with a deliberately safe display message."""


def _bounded_integer(minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer") from None
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return number

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognitive-inquiry",
        description="Frame, research, challenge, and synthesize a question in a saved inquiry.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Start a new inquiry")
    start.add_argument("question", help="The initial question to explore")
    start.add_argument("--purpose", default="", help="What the answer should help you understand or decide")
    start.add_argument("--context-file", type=Path, help="UTF-8 text or Markdown containing supplied context")
    start.add_argument("--output", type=Path, required=True, help="Directory for state.json and report.md")
    start.add_argument("--max-steps", type=_bounded_integer(1, 1000), default=80)
    start.add_argument("--max-searches", type=_bounded_integer(0, 200), default=12)
    start.add_argument("--max-reads", type=_bounded_integer(0, 300), default=18)
    _add_runtime_options(start)

    resume = commands.add_parser("resume", help="Continue a checkpoint, optionally answering clarifications")
    resume.add_argument("directory", type=Path)
    resume.add_argument(
        "--answer", action="append", default=[], metavar="ID=ANSWER",
        help="Answer a displayed clarification, e.g. C1=Architecture design; may be repeated",
    )
    resume.add_argument("--additional-steps", type=_bounded_integer(0, 1000), default=0)
    resume.add_argument("--additional-searches", type=_bounded_integer(0, 200), default=0)
    resume.add_argument("--additional-reads", type=_bounded_integer(0, 300), default=0)
    _add_runtime_options(resume)

    status = commands.add_parser("status", help="Inspect a saved inquiry without making API calls")
    status.add_argument("directory", type=Path)

    report = commands.add_parser("report", help="Print the saved Markdown report without making API calls")
    report.add_argument("directory", type=Path)

    demo = commands.add_parser("demo", help="Run an offline demonstration with synthetic fixtures")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--quiet", action="store_true", help="Suppress step progress")
    return parser


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="default", help="Model alias or provider model ID")
    parser.add_argument("--provider", help="LLM provider from Cognitive Core's provider factory")
    parser.add_argument("--quiet", action="store_true", help="Suppress step progress")


def _parse_answers(values: Sequence[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for value in values:
        identifier, separator, answer = value.partition("=")
        identifier, answer = identifier.strip(), answer.strip()
        if not separator or not identifier or not answer or any(c.isspace() for c in identifier):
            raise CLIError("Each --answer must contain a clarification ID and a nonempty answer: C1=answer")
        if identifier in answers:
            raise CLIError("A clarification ID was supplied more than once; provide one answer per ID")
        answers[identifier] = answer
    return answers


def _display(value: object) -> str:
    """Keep source/model text from emitting terminal control sequences."""
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", str(value))


def _progress(event: InquiryEvent) -> None:
    target = f" ({event.question_id})" if event.question_id else ""
    summary = " ".join(_display(event.summary).split())[:180]
    print(f"[{event.step:03d}] {_display(event.kind)}{_display(target)}: {summary}", file=sys.stderr, flush=True)


def _new_store(directory: Path) -> InquiryStore:
    from cognitive_core.inquiry.store import InquiryStore

    return InquiryStore(directory)


def _load_state(store: InquiryStore) -> InquiryState:
    if not store.state_path.is_file():
        raise CLIError("No saved inquiry was found in that directory; use start or choose an existing inquiry")
    try:
        return store.load()
    except (ValueError, TypeError):
        raise CLIError("The saved state is invalid or uses an unsupported schema; the file has not been changed") from None


def _build_runner(store: InquiryStore, *, model: str, provider: str | None, quiet: bool):
    from cognitive_core.engine.tools import ToolRegistry
    from cognitive_core.inquiry.reasoning import CoreReasoner
    from cognitive_core.inquiry.research import SerperResearchProvider
    from cognitive_core.inquiry.runner import InquiryRunner

    tools = ToolRegistry()
    SerperResearchProvider().register_tools(tools)
    return InquiryRunner(
        reasoner=CoreReasoner(model=model, provider=provider),
        tools=tools,
        store=store,
        on_event=None if quiet else _progress,
    )


def _print_status(state: InquiryState, store: InquiryStore) -> None:
    print(f"Inquiry: {_display(state.id)}")
    print(f"Status: {state.status}")
    print(f"Stage: {state.phase}")
    print(f"Question: {_display(state.original_question)}")
    if state.frame:
        print(f"Working question: {_display(state.frame.working_question)}")
        print(f"Purpose: {_display(state.frame.purpose)}")
        print(f"Frame version: {len(state.frames)}")
    if state.questions:
        print("\nSubquestions:")
        for identifier, question in state.questions.items():
            print(
                f"  {_display(identifier)} [{question.status}; {question.phase}; "
                f"rounds {question.rounds}]: {_display(question.specification.question)}"
            )
    pending = [c for c in state.clarifications if c.active and c.answer is None]
    if pending:
        print("\nPending clarifications:")
        for clarification in pending:
            requirement = "required" if clarification.blocking else "optional"
            target = f", {_display(clarification.question_id)}" if clarification.question_id else ""
            print(f"  {_display(clarification.id)} ({requirement}{target}): {_display(clarification.question)}")
            print(f"    Why: {_display(clarification.reason)}")
        print('  Resume with --answer "ID=your answer" for each clarification you can resolve.')
    print(
        f"\nBudget used: steps {state.usage.steps}/{state.limits.max_steps}; "
        f"searches {state.usage.searches}/{state.limits.max_searches}; "
        f"reads {state.usage.reads}/{state.limits.max_reads}"
    )
    read_count = sum(source.status == "read" for source in state.sources.values())
    print(f"Sources: {len(state.sources)} discovered; {read_count} read")
    if state.stop_reason:
        print(f"Stopped: {_display(state.stop_reason)}")
    if state.error:
        # The runner's contract stores a sanitized error, not a raw HTTP/model response.
        print(f"Error: {_display(state.error)}")
    print(f"State: {store.state_path}")
    print(f"Report: {store.report_path}")
    if state.status == "budget_exhausted":
        print("Continue with resume and --additional-steps; add search/read budget if needed.")


def _execute(args: argparse.Namespace) -> int:
    if args.command == "start":
        from cognitive_core.inquiry.models import InquiryLimits

        if not args.question.strip():
            raise CLIError("The initial question cannot be empty")
        if len(args.question.strip()) > 20000:
            raise CLIError("The initial question is too long; move supporting material into --context-file")
        context = ""
        if args.context_file is not None:
            try:
                with args.context_file.open(encoding="utf-8") as handle:
                    context = handle.read(200001)
            except FileNotFoundError:
                raise CLIError("The --context-file was not found") from None
            except UnicodeError:
                raise CLIError("The --context-file must be UTF-8 text or Markdown") from None
            if len(context) > 200000:
                raise CLIError("The context file exceeds the supported limit of 200,000 characters")
        store = _new_store(args.output)
        if store.state_path.exists():
            raise CLIError("That directory already contains an inquiry; use resume or choose another --output")
        runner = _build_runner(store, model=args.model, provider=args.provider, quiet=args.quiet)
        state = runner.start(
            args.question.strip(), purpose=args.purpose, context=context,
            limits=InquiryLimits(
                max_steps=args.max_steps, max_searches=args.max_searches, max_reads=args.max_reads,
            ),
        )
    elif args.command == "resume":
        answers = _parse_answers(args.answer)
        store = _new_store(args.directory)
        current = _load_state(store)
        pending_ids = {c.id for c in current.clarifications if c.active and c.answer is None}
        if any(identifier not in pending_ids for identifier in answers):
            raise CLIError("An answer ID is not a pending clarification; use status to see the current IDs")
        if current.status == "completed":
            if answers or args.additional_steps or args.additional_searches or args.additional_reads:
                raise CLIError("This inquiry is complete; start a new inquiry to explore a revised question")
            _print_status(current, store)
            return 0
        if (
            current.limits.max_steps + args.additional_steps > 1000
            or current.limits.max_searches + args.additional_searches > 200
            or current.limits.max_reads + args.additional_reads > 300
        ):
            raise CLIError("The requested budget exceeds the per-inquiry limit (1,000 steps, 200 searches, 300 reads)")
        runner = _build_runner(store, model=args.model, provider=args.provider, quiet=args.quiet)
        state = runner.resume(
            answers=answers, additional_steps=args.additional_steps,
            additional_searches=args.additional_searches, additional_reads=args.additional_reads,
        )
    elif args.command == "status":
        store = _new_store(args.directory)
        state = _load_state(store)
    elif args.command == "report":
        store = _new_store(args.directory)
        if not store.report_path.is_file():
            raise CLIError("No saved report was found in that directory; run or resume the inquiry first")
        sys.stdout.write(store.report_path.read_text(encoding="utf-8"))
        return 0
    elif args.command == "demo":
        from cognitive_core.inquiry.demo import run_demo

        store = _new_store(args.output)
        if store.state_path.exists():
            raise CLIError("That directory already contains an inquiry; choose another --output for the demo")
        print("OFFLINE / SYNTHETIC DEMONSTRATION — scripted model responses and source fixtures.")
        print("This demonstrates the inquiry mechanics; its findings are not empirical research.\n")
        state = run_demo(args.output, on_event=None if args.quiet else _progress)
    else:
        raise CLIError("Unknown command")
    _print_status(state, store)
    return 1 if state.status == "failed" else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _execute(args)
    except CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted. Use status, then resume, to continue from the saved checkpoint.", file=sys.stderr)
        return 130
    except ImportError:
        print(
            'Error: A required dependency is missing. Install ".[inquiry]" and the SDK for your chosen '
            "LLM provider; see docs/inquiry.md. Offline commands do not require an LLM SDK.",
            file=sys.stderr,
        )
    except (PermissionError, IsADirectoryError, NotADirectoryError):
        print("Error: The inquiry or context path is inaccessible. Check the path and its permissions.", file=sys.stderr)
    except FileNotFoundError:
        print("Error: A required file was not found. Check the inquiry directory and LLM_CONFIG_PATH.", file=sys.stderr)
    except Exception as exc:
        # Provider errors may contain headers, keys, or request bodies. Display
        # neither exception text nor traceback at this boundary.
        if isinstance(exc, RuntimeError) and str(exc) == "This inquiry is already running in another process":
            print("Error: This inquiry is already running in another process. Use status to inspect it.", file=sys.stderr)
            return 1
        print(
            f"Error: Inquiry command failed ({type(exc).__name__}). "
            "Use status to inspect the checkpoint and check your runtime configuration.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
