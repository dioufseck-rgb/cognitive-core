"""Read-only presentation of a native Coordinator inquiry checkpoint."""
from __future__ import annotations


def render_report(instance, state: dict) -> str:
    study = state.get("study", {})
    status = getattr(instance.status, "value", instance.status)
    report = study.get("report") or {}
    current_report = bool(report and not report.get("stale"))
    review = study.get("final_review") or {}
    reviewed = False
    if current_report and review.get("survives") and review.get("valid", False):
        from cognitive_core.inquiry.native_profiles import review_basis
        reviewed = (review.get("report_revision") == report.get("revision")
                    and review.get("basis") == review_basis(study))
    governed = status == "completed" and bool(study.get("governance")) and reviewed
    lines = ["# Inquiry", "", study.get("original_question", state.get("input", {}).get("question", "No Study was initialized.")), "",
             f"**Coordinator status:** {status}",
             f"**Research outcome:** {study.get('outcome', 'unavailable')}", ""]
    if state.get("metadata", {}).get("synthetic_demo"):
        lines += ["**Synthetic demonstration:** the model and source responses are fixtures; no live research was performed.", ""]
    if not governed:
        lines += ["This inquiry has not completed native governance. Any findings or report below remain provisional.", ""]
    if instance.error:
        lines += [f"**Run error:** {instance.error}", ""]
    if study.get("outcome_reason") and not current_report:
        lines += [study["outcome_reason"], ""]
    if current_report:
        lines += [report["artifact"], ""]
    else:
        if report:
            lines += ["The previous report is superseded by subsequent study changes. Current findings follow; a new synthesis and report review are required.", ""]
        frame = study.get("frame")
        if frame:
            lines += ["## Working question", "", frame["working_question"], "", frame["purpose"], ""]
        lines += ["## Findings so far", ""]
        for identifier, question in study.get("questions", {}).items():
            lines += [f"### {identifier}: {question.get('question', '')}", "",
                      f"Status: {question.get('status', 'pending')}", ""]
            findings = [f for f in study.get("findings", [])
                        if f.get("question_id") == identifier and not f.get("stale")]
            if findings:
                lines += [findings[-1]["finding"], ""]
            else:
                lines += ["No current finding has been accepted.", ""]
    pending = {k: v for k, v in study.get("clarifications", {}).items()
               if v.get("active", True) and v.get("answer") is None}
    if pending:
        lines += ["## Clarifications", ""]
        for identifier, item in pending.items():
            lines += [f"- **{identifier}**: {item['question']} — {item['reason']}"]
        lines.append("")
    lines += ["## Sources", "", "Search results are leads. Only stored, read passages can support citations.", ""]
    for identifier, source in study.get("sources", {}).items():
        title = source.get("title", identifier).replace("[", "(").replace("]", ")")
        passages = ", ".join(p["id"] for p in source.get("passages", []))
        detail = f"; passages: {passages}" if passages else ""
        if source.get("truncated"):
            detail += "; extracted text truncated"
        lines += [f"- **{identifier}** [{title}](<{source['url']}>) — {source['status']}{detail}"]
    if not study.get("sources"):
        lines += ["No sources have been recorded."]
    usage, limits = study.get("usage", {}), study.get("limits", {})
    lines += ["", "## Run details", "",
              f"Instance: `{instance.instance_id}`; Study revision: {study.get('revision', 0)}.", "",
              f"Accepted operations: {usage.get('steps', 0)}/{limits.get('max_steps', '—')}; "
              f"model attempts: {usage.get('model_calls', 0)}/{limits.get('max_model_calls', '—')}; "
              f"searches: {usage.get('searches', 0)}/{limits.get('max_searches', '—')}; "
              f"source reads: {usage.get('reads', 0)}/{limits.get('max_reads', '—')}.", "",
              f"Final report review: {'passed' if reviewed else 'not passed or not current'}; "
              f"governance tier: {instance.governance_tier}.", "",
              "Claim classifications, support assessments, and critique verdicts are model judgments. "
              "Passage validity does not by itself establish substantive support.", "",
              "The native Coordinator database is authoritative. It contains the Study, exact read passages, "
              "accepted operations, resource work orders, and audit events. JSON and Markdown files are inspection exports.", ""]
    return "\n".join(lines)
