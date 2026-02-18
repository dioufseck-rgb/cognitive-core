"""
Cognitive Core — Eval Report Generator (PDF)

Produces a visual PDF dashboard from eval results.

Usage:
    python -m evals.report --pack evals/packs/product_return.yaml --demo
    python -m evals.report --pack evals/packs/product_return.yaml --dry-run
    python -m evals.report --pack evals/packs/product_return.yaml --model default
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# ── Import runner — works both as script and as python -m ──
try:
    from evals.runner import (
        EvalResult, CaseScore, AcceptanceCriteria, CaseExpectation,
        load_eval_pack, run_eval_pack,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    _runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
    _spec = importlib.util.spec_from_file_location("evals.runner", _runner_path)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["evals.runner"] = _mod
    _spec.loader.exec_module(_mod)
    EvalResult = _mod.EvalResult
    CaseScore = _mod.CaseScore
    AcceptanceCriteria = _mod.AcceptanceCriteria
    CaseExpectation = _mod.CaseExpectation
    load_eval_pack = _mod.load_eval_pack
    run_eval_pack = _mod.run_eval_pack


# ── reportlab imports ──
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Wedge
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF

# ═══════════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════════

NAVY = HexColor("#1B2A4A")
BLUE = HexColor("#2E5984")
ACCENT = HexColor("#3A7BBF")
LIGHT_BLUE = HexColor("#E8F0F8")
GREEN = HexColor("#2E7D32")
GREEN_LIGHT = HexColor("#E8F5E9")
RED = HexColor("#C62828")
RED_LIGHT = HexColor("#FFEBEE")
AMBER = HexColor("#F57F17")
AMBER_LIGHT = HexColor("#FFF8E1")
WARM_GRAY = HexColor("#F5F5F0")
MID_GRAY = HexColor("#888888")
LIGHT_GRAY = HexColor("#E0E0E0")
BG_WHITE = HexColor("#FFFFFF")


# ═══════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════

def draw_gauge(value, threshold, label, width=130, height=100):
    """Draw a semicircular gauge with value vs threshold."""
    d = Drawing(width, height)
    cx, cy = width / 2, 35
    r = 38

    # Background arc (gray)
    d.add(Wedge(cx, cy, r, 0, 180, fillColor=LIGHT_GRAY, strokeColor=None))

    # Value arc
    angle = min(value / 100.0, 1.0) * 180
    color = GREEN if value >= threshold else RED
    d.add(Wedge(cx, cy, r, 0, angle, fillColor=color, strokeColor=None))

    # Inner circle (white center)
    d.add(Circle(cx, cy, r * 0.6, fillColor=BG_WHITE, strokeColor=None))

    # Value text
    d.add(String(cx, cy - 5, f"{value:.0f}%",
                 fontSize=14, fontName="Helvetica-Bold",
                 fillColor=color, textAnchor="middle"))

    # Threshold line
    thresh_angle = (threshold / 100.0) * math.pi
    tx = cx - r * 0.8 * math.cos(thresh_angle)
    ty = cy + r * 0.8 * math.sin(thresh_angle)
    tx2 = cx - r * 1.1 * math.cos(thresh_angle)
    ty2 = cy + r * 1.1 * math.sin(thresh_angle)
    d.add(Line(tx, ty, tx2, ty2, strokeColor=NAVY, strokeWidth=1.5))

    # Label
    d.add(String(cx, height - 12, label,
                 fontSize=8, fontName="Helvetica-Bold",
                 fillColor=NAVY, textAnchor="middle"))

    # Pass/fail badge
    badge_color = GREEN if value >= threshold else RED
    badge_text = "PASS" if value >= threshold else "FAIL"
    d.add(String(cx, cy - 18, badge_text,
                 fontSize=7, fontName="Helvetica-Bold",
                 fillColor=badge_color, textAnchor="middle"))

    return d


def draw_category_bar(normal_pass, normal_total, edge_pass, edge_total, adv_pass, adv_total, width=460, height=50):
    """Stacked horizontal bar showing pass/fail by category."""
    d = Drawing(width, height)
    bar_y = 15
    bar_h = 20
    total = normal_total + edge_total + adv_total
    if total == 0:
        return d

    x = 10
    for label, passed, count, color, light in [
        ("Normal", normal_pass, normal_total, GREEN, GREEN_LIGHT),
        ("Edge", edge_pass, edge_total, AMBER, AMBER_LIGHT),
        ("Adversarial", adv_pass, adv_total, RED, RED_LIGHT),
    ]:
        if count == 0:
            continue
        w = (count / total) * (width - 20)
        # Background
        d.add(Rect(x, bar_y, w, bar_h, fillColor=light, strokeColor=LIGHT_GRAY, strokeWidth=0.5))
        # Fill for passed
        pw = (passed / count) * w if count > 0 else 0
        d.add(Rect(x, bar_y, pw, bar_h, fillColor=color, strokeColor=None))
        # Label
        d.add(String(x + w / 2, bar_y + 6, f"{label}: {passed}/{count}",
                     fontSize=8, fontName="Helvetica-Bold",
                     fillColor=white if passed == count else NAVY,
                     textAnchor="middle"))
        x += w

    return d


# ═══════════════════════════════════════════════════════════════════
# PDF builder
# ═══════════════════════════════════════════════════════════════════

def build_report(result: EvalResult, output_path: str):
    """Build a visual PDF report from eval results."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("CoverTitle", fontName="Helvetica-Bold", fontSize=28,
                              textColor=NAVY, alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle("CoverSub", fontName="Helvetica", fontSize=14,
                              textColor=BLUE, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=16,
                              textColor=NAVY, spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle("SubTitle", fontName="Helvetica-Bold", fontSize=11,
                              textColor=BLUE, spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5,
                              textColor=black, spaceAfter=6, leading=13))
    styles.add(ParagraphStyle("Small", fontName="Helvetica", fontSize=8,
                              textColor=MID_GRAY, spaceAfter=4))
    styles.add(ParagraphStyle("VerdictPass", fontName="Helvetica-Bold", fontSize=18,
                              textColor=GREEN, alignment=TA_CENTER, spaceBefore=10))
    styles.add(ParagraphStyle("VerdictFail", fontName="Helvetica-Bold", fontSize=18,
                              textColor=RED, alignment=TA_CENTER, spaceBefore=10))

    story = []

    # ── Cover ──
    story.append(Spacer(1, 60))
    story.append(Paragraph("COGNITIVE CORE", styles["CoverTitle"]))
    story.append(Paragraph("Eval Report", styles["CoverSub"]))
    story.append(Spacer(1, 20))

    # Verdict banner
    if result.all_gates_pass:
        story.append(Paragraph("\u2713  ALL QUALITY GATES PASSED", styles["VerdictPass"]))
    else:
        failed_gates = [n for n, g in result.gate_results().items() if not g["passed"]]
        story.append(Paragraph(f"\u2717  GATES FAILED: {', '.join(failed_gates)}", styles["VerdictFail"]))

    story.append(Spacer(1, 20))

    # Summary stats table
    summary_data = [
        ["Pack", result.pack_name],
        ["Workflow / Domain", f"{result.workflow} / {result.domain}"],
        ["Cases", f"{result.total} ({result.passed} passed, {result.failed} failed)"],
        ["Errors", str(len(result.error_cases))],
        ["Duration", f"{result.elapsed_seconds:.1f}s"],
    ]
    summary_table = Table(summary_data, colWidths=[120, 340])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("TEXTCOLOR", (1, 0), (1, -1), black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LIGHT_GRAY),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)

    story.append(PageBreak())

    # ── Quality Gates ──
    story.append(Paragraph("Quality Gates", styles["SectionTitle"]))

    gates = result.gate_results()
    if gates:
        # Draw gauges in a row
        gauge_drawings = []
        for name, g in gates.items():
            short_name = name.replace("_", " ").title()
            if len(short_name) > 16:
                short_name = short_name[:15] + "."
            gauge = draw_gauge(g["actual"], g["threshold"], short_name)
            gauge_drawings.append(gauge)

        # Lay out gauges in a table (max 4 per row)
        rows = []
        for i in range(0, len(gauge_drawings), 4):
            rows.append(gauge_drawings[i:i + 4])
        # Pad last row
        if rows and len(rows[-1]) < 4:
            rows[-1].extend([""] * (4 - len(rows[-1])))
        if rows:
            gauge_table = Table(rows, colWidths=[130] * 4)
            gauge_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(gauge_table)

        story.append(Spacer(1, 10))

        # Gate detail table
        gate_data = [["Gate", "Actual", "Threshold", "Status"]]
        for name, g in gates.items():
            status = "\u2713 PASS" if g["passed"] else "\u2717 FAIL"
            gate_data.append([
                name.replace("_", " ").title(),
                f"{g['actual']}%",
                f"{g['threshold']}%",
                status,
            ])

        gt = Table(gate_data, colWidths=[180, 80, 80, 80])
        gt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_WHITE, WARM_GRAY]),
            ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))

        # Color the status column
        for i, (name, g) in enumerate(gates.items(), start=1):
            color = GREEN if g["passed"] else RED
            gt.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), color)]))

        story.append(gt)

    # ── Category Breakdown ──
    story.append(Spacer(1, 16))
    story.append(Paragraph("Category Breakdown", styles["SectionTitle"]))

    normal = [c for c in result.cases if c.category == "normal"]
    edge = [c for c in result.cases if c.category == "edge"]
    adv = [c for c in result.cases if c.category == "adversarial"]

    bar = draw_category_bar(
        sum(1 for c in normal if c.passed), len(normal),
        sum(1 for c in edge if c.passed), len(edge),
        sum(1 for c in adv if c.passed), len(adv),
    )
    story.append(bar)
    story.append(Spacer(1, 8))

    cat_data = [["Category", "Total", "Passed", "Failed", "Rate"]]
    for cat_name, cases in [("Normal", normal), ("Edge", edge), ("Adversarial", adv)]:
        p = sum(1 for c in cases if c.passed)
        f = len(cases) - p
        rate = f"{p / len(cases) * 100:.0f}%" if cases else "N/A"
        cat_data.append([cat_name, str(len(cases)), str(p), str(f), rate])
    cat_data.append(["Total", str(result.total), str(result.passed), str(result.failed),
                     f"{result.passed / result.total * 100:.0f}%" if result.total else "N/A"])

    ct = Table(cat_data, colWidths=[120, 60, 60, 60, 60])
    ct.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [BG_WHITE, WARM_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(ct)

    story.append(PageBreak())

    # ── Per-Case Results ──
    story.append(Paragraph("Case Results", styles["SectionTitle"]))

    case_data = [["Case ID", "Category", "Checks", "Status", "Detail"]]
    for cs in result.cases:
        status = "\u2713" if cs.passed else "\u2717"
        checks = f"{cs.pass_count}/{cs.check_count}"
        # Find first failed check for detail
        failed = [c for c in cs.checks if not c["passed"]]
        detail = failed[0]["detail"][:55] + "..." if failed else ("ERROR: " + cs.error[:45] if cs.error else "All passed")
        case_data.append([cs.case_id, cs.category, checks, status, detail])

    ct2 = Table(case_data, colWidths=[110, 70, 50, 40, 200])
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BG_WHITE, WARM_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (2, 0), (3, -1), "CENTER"),
    ]

    # Color status column per row
    for i, cs in enumerate(result.cases, start=1):
        color = GREEN if cs.passed else RED
        style_cmds.append(("TEXTCOLOR", (3, i), (3, i), color))
        style_cmds.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))

    ct2.setStyle(TableStyle(style_cmds))
    story.append(ct2)

    # ── Failed Check Details ──
    failed_cases = [cs for cs in result.cases if not cs.passed]
    if failed_cases:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Failed Check Details", styles["SectionTitle"]))

        for cs in failed_cases:
            story.append(Paragraph(
                f"<b>{cs.case_id}</b> [{cs.category}] — {cs.description}",
                styles["SubTitle"],
            ))
            failed_checks = [c for c in cs.checks if not c["passed"]]
            for check in failed_checks:
                story.append(Paragraph(
                    f"\u2717 <b>{check['name']}</b>: {check.get('detail', '')}",
                    styles["Body"],
                ))
            if cs.error:
                story.append(Paragraph(f"ERROR: {cs.error}", styles["Body"]))
            story.append(Spacer(1, 4))

    # ── Footer ──
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Generated by Cognitive Core Eval Framework — {time.strftime('%Y-%m-%d %H:%M')}",
        styles["Small"],
    ))

    doc.build(story)
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Demo: generate synthetic scores for report preview
# ═══════════════════════════════════════════════════════════════════

def generate_demo_result(pack_path: str, project_root: str = ".") -> EvalResult:
    """Generate a realistic synthetic EvalResult for report demo."""
    pack, expectations, criteria = load_eval_pack(pack_path, project_root)
    random.seed(42)  # deterministic

    scores = []
    for exp in expectations:
        checks = []

        # Schema valid (95% pass rate)
        checks.append({"name": "schema_valid", "passed": random.random() < 0.95,
                        "detail": "0 parse failures" if random.random() < 0.95 else "1 parse failure"})

        # Classification
        if exp.expected_classification:
            correct = random.random() < (0.92 if exp.category == "normal" else 0.70 if exp.category == "edge" else 0.55)
            actual = exp.expected_classification if correct else "buyers_remorse"
            checks.append({"name": "classification", "passed": correct,
                           "expected": exp.expected_classification, "actual": actual,
                           "detail": f"expected '{exp.expected_classification}', got '{actual}'"})

        # Investigation
        if exp.expected_finding_contains:
            found = random.random() < (0.88 if exp.category == "normal" else 0.65)
            checks.append({"name": "investigation_finding", "passed": found,
                           "detail": f"all {len(exp.expected_finding_contains)} keywords found" if found
                           else f"missing: {exp.expected_finding_contains[-1]}"})

        # Confidence
        if exp.min_confidence is not None or exp.max_confidence is not None:
            conf = random.uniform(0.4, 0.95)
            in_range = True
            if exp.min_confidence and conf < exp.min_confidence:
                in_range = False
            if exp.max_confidence and conf > exp.max_confidence:
                in_range = False
            checks.append({"name": "confidence_range", "passed": in_range,
                           "detail": f"confidence {conf:.2f}"})

        # Fail-closed
        if exp.should_escalate:
            escalated = random.random() < 0.85
            checks.append({"name": "fail_closed", "passed": escalated,
                           "detail": "escalated" if escalated else "did not escalate — CRITICAL"})

        # Generate compliance
        for phrase in exp.must_contain:
            found = random.random() < 0.90
            checks.append({"name": "must_contain", "passed": found,
                           "expected": f"contains '{phrase}'",
                           "detail": f"looking for '{phrase}'"})
        for phrase in exp.must_not_contain:
            absent = random.random() < 0.88
            checks.append({"name": "must_not_contain", "passed": absent,
                           "expected": f"does NOT contain '{phrase}'",
                           "detail": f"checking '{phrase}' not in output"})

        all_passed = all(c["passed"] for c in checks)
        scores.append(CaseScore(
            case_id=exp.case_id,
            category=exp.category,
            description=exp.description,
            passed=all_passed,
            checks=checks,
            elapsed_seconds=random.uniform(3.0, 18.0),
        ))

    return EvalResult(
        pack_name=pack.get("name", "demo"),
        workflow=pack["workflow"],
        domain=pack["domain"],
        cases=scores,
        criteria=criteria,
        elapsed_seconds=sum(s.elapsed_seconds for s in scores),
    )


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cognitive Core Eval Report")
    parser.add_argument("--pack", required=True, help="Path to eval pack YAML")
    parser.add_argument("--output", "-o", help="Output PDF path")
    parser.add_argument("--demo", action="store_true", help="Use synthetic scores")
    parser.add_argument("--dry-run", action="store_true", help="Score pre-recorded outputs")
    parser.add_argument("--model", default="default", help="Model alias for live run")
    parser.add_argument("--root", default=".", help="Project root")
    args = parser.parse_args()

    pack_name = Path(args.pack).stem
    output = args.output or f"{pack_name}_eval_report.pdf"

    if args.demo:
        result = generate_demo_result(args.pack, args.root)
    else:
        result = run_eval_pack(args.pack, args.root, args.model, args.dry_run)

    build_report(result, output)
    print(f"Report: {output}")
    print(result.summary())


if __name__ == "__main__":
    main()
