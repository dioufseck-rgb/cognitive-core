"""Render complete or interrupted inquiry results without inventing an answer."""

from __future__ import annotations

import re

from cognitive_core.inquiry.models import InquiryState


def _plain(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;").replace("[", "\\[").replace("]", "\\]")


def render_report(state: InquiryState) -> str:
    lines = ["# Inquiry", "", state.original_question, "", f"**Status:** {state.status}", ""]
    if state.frame:
        lines += ["## Working question", "", state.frame.working_question, "", state.frame.purpose, ""]
    if state.stop_reason:
        lines += [state.stop_reason, ""]
    if state.error:
        lines += ["**Run error:** " + state.error, ""]
    if state.answer:
        def link(match):
            pid = match.group(1)
            source = state.sources.get(pid.split(":")[0])
            return f"[{pid}](<{source.url}>)" if source else match.group(0)

        answer = re.sub(r"\[(S[1-9][0-9]*:P[1-9][0-9]*)\]", link, state.answer)
        lines += ["## Answer", "", answer, ""]
    else:
        lines += ["## Findings so far", "", "This is an unfinished inquiry; the findings below are provisional.", ""]
    lines += ["## Subquestions", ""]
    for q in state.questions.values():
        lines += [f"### {q.specification.id}: {q.specification.question}", "", f"Status: **{q.status}**", ""]
        if q.findings:
            lines += [q.findings[-1].finding, ""]
        if q.unresolved_reason:
            lines += ["Qualification: " + q.unresolved_reason, ""]
        if q.findings and q.findings[-1].unresolved_issues:
            lines += ["Open issues:", ""] + [f"- {v}" for v in q.findings[-1].unresolved_issues] + [""]
    assumptions = list(state.frame.assumptions if state.frame else [])
    for q in state.questions.values():
        if q.findings:
            assumptions.extend(q.findings[-1].assumptions)
    if assumptions:
        lines += ["## Assumptions", ""] + [f"- {v}" for v in dict.fromkeys(assumptions)] + [""]
    if state.synthesis and state.synthesis.limitations:
        lines += ["## Limitations", ""] + [f"- {v}" for v in state.synthesis.limitations] + [""]
    if state.synthesis and state.synthesis.open_questions:
        lines += ["## Remaining questions", ""] + [f"- {v}" for v in state.synthesis.open_questions] + [""]
    pending = [c for c in state.clarifications if c.active and c.answer is None]
    if pending:
        lines += ["## Clarifications", ""]
        for c in pending:
            lines += [f"- **{c.id}:** {c.question} — {c.reason}" + ("" if c.blocking else " (optional)")]
        lines += [""]
    if state.sources:
        lines += ["## Sources", "", "Only read passages can support source citations. Search snippets are discovery leads.", ""]
        for s in state.sources.values():
            lines += [f"- **{s.id}** [{_plain(s.title or s.url)}](<{s.url}>) — {s.status}"
                      + ("; extracted text truncated" if s.truncated else "")
                      + (f"; retrieved {s.retrieved_at}" if s.retrieved_at else "")]
        lines += [""]
    lines += ["## Run details", "",
              f"Steps: {state.usage.steps}/{state.limits.max_steps}; "
              f"searches: {state.usage.searches}/{state.limits.max_searches}; "
              f"source reads: {state.usage.reads}/{state.limits.max_reads}.", "",
              "The checkpoint contains the frame history, structured findings, source passages, objections, and events.", ""]
    return "\n".join(lines)
