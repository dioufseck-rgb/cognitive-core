"""Render complete or interrupted inquiry results without inventing an answer."""

from __future__ import annotations

from collections import Counter
import re

from cognitive_core.inquiry.models import FindingOutput, InquiryState, QuestionRecord


def _plain(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;").replace("[", "\\[").replace("]", "\\]")


def _claim_lines(state: InquiryState, finding: FindingOutput) -> list[str]:
    """Expose the model's classifications without upgrading them to evidence."""
    labels = {
        "source": "Source claim",
        "inference": "Inference",
        "assumption": "Assumption",
        "user": "User-supplied claim",
    }
    items = []
    cited_claims = 0
    for claim in finding.claims:
        links = []
        unavailable = []
        for reference in claim.evidence:
            source = state.sources.get(reference.passage_id.split(":")[0])
            passage = next((p for p in source.passages if p.id == reference.passage_id), None) if source else None
            if (
                source is not None and source.status == "read" and passage is not None
                and " ".join(reference.quote.split()) in " ".join(passage.text.split())
            ):
                links.append(f"[{reference.passage_id}](<{source.url}>)")
            else:
                unavailable.append(reference.passage_id)
        cited_claims += bool(links)
        item = f"- **{labels[claim.basis]}:** {_plain(claim.text)}"
        if links:
            item += " Read passages: " + ", ".join(dict.fromkeys(links)) + "."
        if unavailable:
            item += " Unavailable evidence references: " + ", ".join(dict.fromkeys(unavailable)) + "."
        if claim.qualification:
            item += " Qualification: " + _plain(claim.qualification)
        items.append(item)
    claim_word = "claim" if len(finding.claims) == 1 else "claims"
    coverage = (
        f"Read passages are cited for {cited_claims} of {len(finding.claims)} {claim_word}."
        if cited_claims else "No claim in the latest finding cites a read passage."
    )
    return ["**Evidence references:** " + coverage, "", "Claims (model classifications):", "", *items, ""]


def _review_lines(question: QuestionRecord) -> list[str]:
    lines = []
    if question.findings and len(question.challenges) < len(question.findings):
        lines += ["**Review status:** The latest finding has not yet been challenged.", ""]
    if not question.challenges:
        return lines
    challenge = question.challenges[-1]
    earlier = " of an earlier finding" if len(question.challenges) < len(question.findings) else ""
    verdict = "survives" if challenge.survives else "does not survive"
    lines += [f"**Latest critique{earlier} (model assessment):** {verdict}. "
              + _plain(challenge.overall_assessment), ""]
    if challenge.vulnerabilities:
        lines += ["Objections in that critique:", ""]
        for objection in challenge.vulnerabilities:
            lines += [f"- **{_plain(objection.severity)}:** {_plain(objection.description)} "
                      + "Suggested remedy: " + _plain(objection.recommendation)]
        lines += [""]
    # A newer critique can be checkpointed before its reflection. Do not attach
    # an older response or decision to that new critique.
    if question.reflections and len(question.reflections) >= len(question.challenges):
        reflection = question.reflections[-1]
        decision = reflection.next_action
        if decision == "accept":
            decision += f" ({reflection.resolution})"
        lines += ["**Last review decision (model):** " + decision + ".", ""]
        if reflection.objection_response:
            lines += ["Response to critique: " + _plain(reflection.objection_response), ""]
        if reflection.next_question:
            lines += ["Requested follow-up: " + _plain(reflection.next_question), ""]
    return lines


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
    source_counts = {status: sum(s.status == status for s in state.sources.values())
                     for status in ("read", "discovered", "failed")}
    lines += [f"**Research coverage:** Read sources: {source_counts['read']}; "
              f"unread leads: {source_counts['discovered']}; failed sources: {source_counts['failed']}.", ""]
    lines += ["## Subquestions", "",
              "Claim classifications and critique verdicts below are supplied by the model. "
              "A passage citation records evidence that was read; it does not by itself establish the claim.", ""]
    for q in state.questions.values():
        lines += [f"### {q.specification.id}: {q.specification.question}", "", f"Status: **{q.status}**", ""]
        lines += [f"Progress: findings drafted {len(q.findings)}; critiques {len(q.challenges)}; "
                  f"review decisions {len(q.reflections)}.", ""]
        if q.findings:
            lines += [q.findings[-1].finding, ""]
            lines += _claim_lines(state, q.findings[-1])
        else:
            lines += ["No finding has been drafted.", ""]
        if q.unresolved_reason:
            lines += ["Qualification: " + q.unresolved_reason, ""]
        if q.findings and q.findings[-1].unresolved_issues:
            lines += ["Open issues:", ""] + [f"- {v}" for v in q.findings[-1].unresolved_issues] + [""]
        lines += _review_lines(q)
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
              f"source reads: {state.usage.reads}/{state.limits.max_reads}.", ""]
    operation_counts = Counter(event.kind for event in state.events)
    operation_labels = {
        "frame": "frame", "retrieve": "retrieval planning", "investigate": "investigate",
        "challenge": "challenge", "reflect": "reflect", "deliberate": "deliberate", "generate": "generate",
    }
    completed_operations = [f"{label}: {operation_counts[kind]}"
                            for kind, label in operation_labels.items() if operation_counts[kind]]
    if completed_operations:
        lines += ["Completed model operations: " + "; ".join(completed_operations) + ".", ""]
    lines += ["The checkpoint contains the frame history, structured findings, source passages, objections, and events.", ""]
    return "\n".join(lines)
