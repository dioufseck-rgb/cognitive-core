"""
Cognitive Core — HTML Trace Page (Sprint 1.3)

Serves a self-contained HTML trace page at GET /instances/{id}/trace

Three modes:
  Watch mode  — live SSE rendering as the workflow executes
  Input mode  — renders when workflow suspends at a HITL gate
  Result mode — complete audit trace after workflow completes

Single self-contained HTML file, no build step, no framework.
Vanilla JS, served directly from the FastAPI server.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

import cognitive_core.api.deps as deps

router = APIRouter()


_TRACE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognitive Core — Workflow Trace</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222535;
    --border: #2d3148;
    --text: #e2e8f0;
    --muted: #64748b;
    --dim: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #f59e0b;
    --blue: #3b82f6;
    --purple: #a855f7;
    --cyan: #06b6d4;
    --orange: #f97316;
    --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
    --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.6;
    padding: 0;
    min-height: 100vh;
  }

  /* ── Header ── */
  #header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  #header-left { display: flex; align-items: center; gap: 12px; }

  .logo {
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .logo span { color: var(--cyan); }

  #instance-id {
    font-size: 12px;
    color: var(--muted);
  }

  #status-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .badge-running   { background: rgba(59,130,246,0.15); color: var(--blue); border: 1px solid rgba(59,130,246,0.3); }
  .badge-suspended { background: rgba(245,158,11,0.15); color: var(--yellow); border: 1px solid rgba(245,158,11,0.3); }
  .badge-completed { background: rgba(34,197,94,0.15);  color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
  .badge-failed    { background: rgba(239,68,68,0.15);   color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }
  .badge-created   { background: rgba(148,163,184,0.15); color: var(--dim);   border: 1px solid rgba(148,163,184,0.3); }

  #header-right { display: flex; align-items: center; gap: 12px; }

  .btn {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 5px;
    border: none;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-outline {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--dim);
  }
  .btn-primary {
    background: var(--cyan);
    color: #000;
  }

  /* ── Main layout ── */
  #main {
    max-width: 900px;
    margin: 0 auto;
    padding: 32px 24px 80px;
  }

  /* ── Case header ── */
  #case-header {
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  #case-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
  }

  #case-meta {
    font-size: 11px;
    color: var(--muted);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }

  #case-meta span { display: flex; align-items: center; gap: 5px; }

  #elapsed-counter {
    font-variant-numeric: tabular-nums;
  }

  /* ── Steps ── */
  #steps { display: flex; flex-direction: column; gap: 2px; }

  .step {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--surface);
  }

  .step-header {
    display: grid;
    grid-template-columns: 16px 180px 90px 1fr 80px;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    cursor: pointer;
    user-select: none;
  }
  .step-header:hover { background: var(--surface2); }

  .step-icon { font-size: 10px; color: var(--muted); }
  .step-icon.running  { color: var(--blue); animation: spin 1s linear infinite; }
  .step-icon.done     { color: var(--green); }
  .step-icon.failed   { color: var(--red); }
  .step-icon.paused   { color: var(--yellow); }
  .step-icon.human    { color: var(--purple); }

  @keyframes spin { to { transform: rotate(360deg); } }

  .step-name {
    font-weight: 600;
    color: var(--text);
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .step-primitive {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .step-summary {
    font-size: 11px;
    color: var(--dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .step-elapsed {
    font-size: 11px;
    color: var(--muted);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .step-body {
    display: none;
    padding: 12px 16px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface2);
  }

  .step.expanded .step-body { display: block; }

  .step-output {
    font-size: 11px;
    color: var(--dim);
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Confidence bar */
  .confidence-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 8px;
    margin-bottom: 4px;
  }

  .confidence-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .confidence-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--green);
    transition: width 0.4s ease;
  }

  .confidence-label {
    font-size: 10px;
    color: var(--muted);
    min-width: 60px;
    text-align: right;
  }

  /* Verify checks */
  .verify-checks { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
  .verify-check  { font-size: 11px; display: flex; align-items: flex-start; gap: 8px; }
  .check-pass    { color: var(--green); }
  .check-fail    { color: var(--red); }

  /* Governance tier badge */
  .tier-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-right: 6px;
  }
  .tier-auto       { background: rgba(34,197,94,0.15); color: var(--green); }
  .tier-spot_check { background: rgba(59,130,246,0.15); color: var(--blue); }
  .tier-gate       { background: rgba(245,158,11,0.2); color: var(--yellow); }
  .tier-hold       { background: rgba(239,68,68,0.15); color: var(--red); }

  /* ── HITL panel ── */
  #hitl-panel {
    display: none;
    margin: 20px 0;
    border: 1px solid var(--yellow);
    border-radius: 8px;
    background: rgba(245,158,11,0.05);
    overflow: hidden;
  }

  #hitl-panel.visible { display: block; }

  .hitl-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    border-bottom: 1px solid rgba(245,158,11,0.2);
    background: rgba(245,158,11,0.08);
  }

  .hitl-header-icon { color: var(--yellow); font-size: 14px; }

  .hitl-header-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--yellow);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .hitl-body { padding: 18px; }

  .hitl-brief {
    font-size: 12px;
    color: var(--text);
    margin-bottom: 14px;
    line-height: 1.7;
  }

  .hitl-row {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .hitl-label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    min-width: 90px;
    padding-top: 2px;
  }

  .hitl-value {
    font-size: 12px;
    color: var(--dim);
    flex: 1;
  }

  .hitl-input {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    padding: 8px 12px;
    width: 100%;
    resize: none;
    outline: none;
    transition: border-color 0.15s;
  }
  .hitl-input:focus { border-color: var(--cyan); }

  .hitl-reviewer-input {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    padding: 6px 10px;
    width: 200px;
    outline: none;
  }
  .hitl-reviewer-input:focus { border-color: var(--cyan); }

  .decision-buttons {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
  }

  .decision-btn {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    padding: 8px 18px;
    border-radius: 5px;
    border: 1px solid;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    transition: all 0.15s;
  }
  .decision-btn:hover { opacity: 0.85; transform: translateY(-1px); }
  .decision-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  .btn-approve          { background: rgba(34,197,94,0.15);  border-color: var(--green);  color: var(--green);  }
  .btn-approve_modified { background: rgba(6,182,212,0.15);  border-color: var(--cyan);   color: var(--cyan);   }
  .btn-deny             { background: rgba(239,68,68,0.15);   border-color: var(--red);    color: var(--red);    }
  .btn-refer            { background: rgba(168,85,247,0.15); border-color: var(--purple); color: var(--purple); }
  .btn-escalate         { background: rgba(245,158,11,0.15); border-color: var(--yellow); color: var(--yellow); }
  .btn-release          { background: rgba(34,197,94,0.15);  border-color: var(--green);  color: var(--green);  }

  /* ── Human decision step ── */
  .human-decision-step {
    border: 1px solid var(--purple);
    border-radius: 6px;
    background: rgba(168,85,247,0.05);
    padding: 12px 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .human-decision-icon { color: var(--purple); font-size: 14px; padding-top: 1px; }

  .human-decision-body { flex: 1; }

  .human-decision-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
  }

  .human-decision-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--purple);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .human-decision-reviewer {
    font-size: 10px;
    color: var(--muted);
  }

  .human-decision-text {
    font-size: 12px;
    color: var(--text);
    line-height: 1.6;
  }

  /* ── Result panel ── */
  #result-panel {
    display: none;
    margin-top: 20px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    overflow: hidden;
  }

  #result-panel.visible { display: block; }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }

  .result-header-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .result-icon-complete { color: var(--green); }
  .result-icon-failed   { color: var(--red); }

  .result-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .result-elapsed {
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .result-body { padding: 16px 18px; }

  .result-disposition {
    font-size: 13px;
    color: var(--text);
    margin-bottom: 12px;
  }

  .result-disposition strong { color: var(--cyan); }

  /* ── Verified badge ── */
  #verify-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .verified   { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
  .tampered   { background: rgba(239,68,68,0.15); color: var(--red);   border: 1px solid rgba(239,68,68,0.3); }
  .unverified { background: rgba(148,163,184,0.1); color: var(--muted); border: 1px solid var(--border); }

  /* ── Spinner ── */
  #loading {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 40px 0;
    color: var(--muted);
    font-size: 12px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--cyan);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
  }

  /* ── Error state ── */
  #error-msg {
    display: none;
    padding: 16px;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 6px;
    color: var(--red);
    font-size: 12px;
    margin-top: 20px;
  }

  /* ── Footer ── */
  #footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  #footer-meta {
    font-size: 10px;
    color: var(--muted);
    display: flex;
    gap: 20px;
  }

  #footer-actions { display: flex; gap: 10px; }

  /* ── Submitting overlay ── */
  .submitting-overlay {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--cyan);
  }
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <div id="header-left">
    <div class="logo">Cognitive <span>Core</span></div>
    <div id="instance-id">—</div>
    <span id="status-badge" class="badge-created">Created</span>
    <span id="verify-badge" class="unverified" title="Ledger integrity check">LEDGER</span>
  </div>
  <div id="header-right">
    <span id="header-submitting" style="display:none" class="submitting-overlay">
      <span class="spinner"></span> Processing…
    </span>
    <button class="btn btn-outline" id="btn-refresh" onclick="location.reload()">↺ Refresh</button>
    <button class="btn btn-outline" id="btn-export" onclick="exportTrace()" disabled>Download Trace ↓</button>
  </div>
</div>

<!-- Main content -->
<div id="main">
  <div id="case-header">
    <div id="case-title">Loading…</div>
    <div id="case-meta">
      <span>⏱ Elapsed: <span id="elapsed-counter">—</span></span>
      <span>↳ Steps: <span id="step-count">0</span></span>
      <span>⬡ Domain: <span id="domain-label">—</span></span>
      <span>⬢ Workflow: <span id="workflow-label">—</span></span>
    </div>
  </div>

  <div id="loading"><span class="spinner"></span> Connecting to workflow stream…</div>
  <div id="error-msg"></div>

  <!-- HITL panel (hidden until suspension) -->
  <div id="hitl-panel">
    <div class="hitl-header">
      <span class="hitl-header-icon">⏸</span>
      <span class="hitl-header-title">Human Review Required</span>
    </div>
    <div class="hitl-body">
      <div id="hitl-brief" class="hitl-brief"></div>

      <div class="hitl-row">
        <span class="hitl-label">Tier</span>
        <span id="hitl-tier" class="hitl-value">—</span>
      </div>
      <div class="hitl-row">
        <span class="hitl-label">Reason</span>
        <span id="hitl-reason" class="hitl-value">—</span>
      </div>
      <div class="hitl-row">
        <span class="hitl-label">Reviewer ID</span>
        <input id="hitl-reviewer" class="hitl-reviewer-input" type="text" placeholder="your_id" value="reviewer">
      </div>
      <div class="hitl-row" style="flex-direction:column; gap:6px;">
        <span class="hitl-label">Rationale</span>
        <textarea id="hitl-rationale" class="hitl-input" rows="2" placeholder="Optional note…"></textarea>
      </div>

      <div class="decision-buttons" id="decision-buttons"></div>
    </div>
  </div>

  <!-- Step list -->
  <div id="steps"></div>

  <!-- Result panel -->
  <div id="result-panel">
    <div class="result-header">
      <div class="result-header-left">
        <span id="result-icon" class="result-icon-complete">✓</span>
        <span id="result-label" class="result-label">Complete</span>
        <span id="result-elapsed" class="result-elapsed"></span>
      </div>
    </div>
    <div class="result-body">
      <div id="result-disposition" class="result-disposition"></div>
    </div>
  </div>
</div>

<!-- Footer -->
<div id="footer">
  <div id="footer-meta">
    <span>Cognitive Core v0.1.0-technical-preview</span>
    <span id="footer-instance">—</span>
    <span id="footer-ledger-count">0 ledger entries</span>
  </div>
  <div id="footer-actions">
    <span id="footer-submitting" style="display:none" class="submitting-overlay">
      <span class="spinner"></span> Submitting decision…
    </span>
  </div>
</div>

<script>
// ─── State ────────────────────────────────────────────────────────
const instanceId = document.getElementById('instance-id').textContent = getInstanceId();
let workorder = null;
let instanceStatus = 'created';
let startTime = null;
let elapsedTimer = null;
let ledgerCount = 0;
let traceData = { instance: null, steps: [], workorder: null, events: [] };
let stepMap = {};  // step_name → DOM element index

function getInstanceId() {
  const parts = window.location.pathname.split('/');
  const idx = parts.indexOf('instances');
  return idx >= 0 ? parts[idx + 1] : 'unknown';
}

// ─── DOM helpers ─────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function setStatus(status) {
  instanceStatus = status;
  const badge = $('status-badge');
  badge.textContent = status.toUpperCase();
  badge.className = 'badge-' + status;
}

function fmtElapsed(sec) {
  if (sec === null || sec === undefined) return '—';
  if (sec < 60) return sec.toFixed(1) + 's';
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}m ${s}s`;
}

function fmtMs(ms) {
  if (!ms) return '';
  if (ms < 1000) return ms + 'ms';
  return (ms / 1000).toFixed(1) + 's';
}

function startElapsedCounter(since) {
  startTime = since || Date.now() / 1000;
  if (elapsedTimer) clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    if (instanceStatus === 'running') {
      $('elapsed-counter').textContent = fmtElapsed(Date.now() / 1000 - startTime);
    }
  }, 200);
}

function stopElapsedCounter() {
  if (elapsedTimer) clearInterval(elapsedTimer);
}

// ─── Step rendering ───────────────────────────────────────────────
const PRIMITIVE_COLORS = {
  retrieve: '#06b6d4', classify: '#3b82f6', investigate: '#a855f7',
  deliberate: '#f97316', verify: '#22c55e', govern: '#f59e0b',
  generate: '#64748b', orchestrate: '#e2e8f0'
};

function primitiveColor(p) {
  return PRIMITIVE_COLORS[p] || '#94a3b8';
}

function summaryFor(event) {
  const out = event.output || {};
  if (out.category) {
    const conf = out.confidence ? ` (${(out.confidence * 100).toFixed(0)}%)` : '';
    return out.category + conf;
  }
  if (out.recommendation) return out.recommendation;
  if (out.disposition)    return out.disposition;
  if (out.tier)           return 'Tier: ' + out.tier;
  if (out.status)         return out.status;
  if (typeof out === 'string') return out.substring(0, 80);
  const keys = Object.keys(out);
  if (keys.length > 0) {
    const v = out[keys[0]];
    return typeof v === 'string' ? v.substring(0, 80) : JSON.stringify(v).substring(0, 80);
  }
  return '';
}

function buildOutputHTML(event) {
  const out = event.output || {};
  const prim = event.primitive || '';
  let html = '';

  if (prim === 'classify' && out.category) {
    const conf = out.confidence || 0;
    const pct = (conf * 100).toFixed(0);
    const filled = Math.round(conf * 20);
    const bars = '█'.repeat(filled) + '░'.repeat(20 - filled);
    html += `<div style="color:var(--text);font-weight:600;margin-bottom:6px;">${out.category}</div>`;
    html += `<div class="confidence-row"><span style="font-size:10px;color:var(--muted);min-width:80px;">Confidence</span>
      <div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:${pct}%;background:${conf > 0.7 ? 'var(--green)' : conf > 0.5 ? 'var(--yellow)' : 'var(--red)'};"></div></div>
      <span class="confidence-label">${pct}%</span></div>`;
    if (out.reasoning) html += `<div style="margin-top:8px;font-size:11px;color:var(--dim);">${out.reasoning}</div>`;
    return html;
  }

  if (prim === 'verify') {
    const checks = out.checks || out.results || [];
    if (Array.isArray(checks) && checks.length > 0) {
      html += '<div class="verify-checks">';
      checks.forEach(c => {
        const pass = c.status === 'pass' || c.pass === true || c.result === 'pass' || c.conforms === true;
        const label = c.rule || c.name || c.label || JSON.stringify(c).substring(0, 60);
        const note  = c.note || c.reason || '';
        html += `<div class="verify-check"><span class="${pass ? 'check-pass' : 'check-fail'}">${pass ? '✓' : '✗'}</span>
          <span style="color:var(--dim);">${label}${note ? ' — ' + note : ''}</span></div>`;
      });
      html += '</div>';
    }
    const overall = out.disposition || out.overall || '';
    if (overall) html += `<div style="margin-top:8px;font-size:11px;color:var(--dim);">${overall}</div>`;
    if (!html) html = renderGenericOutput(out);
    return html;
  }

  if (prim === 'govern') {
    const tier = out.tier || out.governance_tier || '';
    if (tier) {
      const tierKey = tier.toLowerCase().replace('-', '_');
      html += `<span class="tier-badge tier-${tierKey}">${tier}</span>`;
    }
    if (out.rationale || out.reason) {
      html += `<span style="font-size:11px;color:var(--dim);">${out.rationale || out.reason}</span>`;
    }
    return html || renderGenericOutput(out);
  }

  if (prim === 'retrieve') {
    const sources = out.sources || out.tools_called || [];
    if (Array.isArray(sources) && sources.length > 0) {
      html += `<div style="font-size:11px;color:var(--muted);">Sources: ${sources.join(', ')}</div>`;
    }
    if (out.summary) html += `<div style="margin-top:6px;font-size:11px;color:var(--dim);">${out.summary}</div>`;
    if (!html) html = renderGenericOutput(out);
    return html;
  }

  return renderGenericOutput(out);
}

function renderGenericOutput(out) {
  if (!out || Object.keys(out).length === 0) return '';
  const text = typeof out === 'string' ? out : JSON.stringify(out, null, 2);
  return `<pre class="step-output">${escHtml(text.substring(0, 800))}${text.length > 800 ? '\n…' : ''}</pre>`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addOrUpdateStep(event) {
  const name = event.step_name || event.action_type;
  if (!name) return;

  const stepsEl = $('steps');
  let el = stepMap[name];

  if (!el) {
    el = document.createElement('div');
    el.className = 'step';
    el.dataset.stepName = name;
    el.innerHTML = `
      <div class="step-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span class="step-icon">▶</span>
        <span class="step-name">${escHtml(name)}</span>
        <span class="step-primitive" style="color:${primitiveColor(event.primitive)}">${escHtml(event.primitive || '')}</span>
        <span class="step-summary"></span>
        <span class="step-elapsed"></span>
      </div>
      <div class="step-body"></div>`;
    stepsEl.appendChild(el);
    stepMap[name] = el;
  }

  const iconEl    = el.querySelector('.step-icon');
  const summaryEl = el.querySelector('.step-summary');
  const elapsedEl = el.querySelector('.step-elapsed');
  const bodyEl    = el.querySelector('.step-body');
  const primEl    = el.querySelector('.step-primitive');

  if (event.primitive) {
    primEl.textContent = event.primitive;
    primEl.style.color = primitiveColor(event.primitive);
  }

  const evType = event.event || event.action_type || '';

  if (evType === 'step_started' || evType === 'workflow_started') {
    iconEl.className = 'step-icon running';
    iconEl.textContent = '◌';
  } else if (evType === 'step_completed') {
    iconEl.className = 'step-icon done';
    iconEl.textContent = '▶';
    if (event.elapsed_ms) elapsedEl.textContent = fmtMs(event.elapsed_ms);
    summaryEl.textContent = summaryFor(event);
    bodyEl.innerHTML = buildOutputHTML(event);
    ledgerCount++;
    $('footer-ledger-count').textContent = ledgerCount + ' ledger entries';
    $('step-count').textContent = Object.keys(stepMap).length;
  } else if (evType === 'step_failed') {
    iconEl.className = 'step-icon failed';
    iconEl.textContent = '✗';
  } else if (evType === 'hitl_requested' || evType === 'governance_decision') {
    iconEl.className = 'step-icon paused';
    iconEl.textContent = '⏸';
    const tier = event.governance_tier || event.tier || '';
    if (tier) {
      const tierKey = tier.toLowerCase().replace('-','_');
      summaryEl.innerHTML = `<span class="tier-badge tier-${tierKey}">${tier}</span>`;
    }
  }

  traceData.events.push(event);
}

function addHumanDecisionStep(event) {
  const stepsEl = $('steps');
  const el = document.createElement('div');
  el.className = 'human-decision-step';
  const reviewer = event.reviewer_id || 'reviewer';
  const ts = event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : '';
  const decision = event.decision || '';
  const rationale = event.rationale || '';
  el.innerHTML = `
    <span class="human-decision-icon">◉</span>
    <div class="human-decision-body">
      <div class="human-decision-header">
        <span class="human-decision-label">Human Decision</span>
        <span class="human-decision-reviewer">reviewer: ${escHtml(reviewer)} · ${ts}</span>
      </div>
      <div class="human-decision-text">
        <strong style="color:var(--cyan)">${escHtml(decision)}</strong>
        ${rationale ? ' — ' + escHtml(rationale) : ''}
      </div>
    </div>`;
  stepsEl.appendChild(el);
}

// ─── HITL panel ───────────────────────────────────────────────────
async function loadWorkorder() {
  try {
    const r = await fetch(`/api/instances/${instanceId}/workorder`);
    if (!r.ok) return;
    workorder = await r.json();
    traceData.workorder = workorder;
    renderHITLPanel(workorder);
  } catch(e) {
    console.warn('Could not load workorder', e);
  }
}

function renderHITLPanel(wo) {
  $('hitl-panel').classList.add('visible');
  $('hitl-brief').textContent = wo.brief || '';
  const tier = wo.governance_tier || '';
  const tierKey = tier.toLowerCase().replace('-','_');
  $('hitl-tier').innerHTML = tier
    ? `<span class="tier-badge tier-${tierKey}">${tier}</span>`
    : '—';
  $('hitl-reason').textContent = wo.escalation_reason || '—';

  const btns = $('decision-buttons');
  btns.innerHTML = '';
  (wo.decision_options || ['approve','deny']).forEach(opt => {
    const b = document.createElement('button');
    b.className = `decision-btn btn-${opt.toLowerCase().replace(' ','_')}`;
    b.textContent = opt.replace('_', ' ').toUpperCase();
    b.onclick = () => submitDecision(opt);
    btns.appendChild(b);
  });
}

async function submitDecision(decision) {
  const reviewer = $('hitl-reviewer').value.trim() || 'reviewer';
  const rationale = $('hitl-rationale').value.trim();
  if (!reviewer) { alert('Please enter a reviewer ID'); return; }

  // Disable all buttons while processing
  $('decision-buttons').querySelectorAll('.decision-btn').forEach(b => b.disabled = true);
  $('header-submitting').style.display = 'flex';
  $('footer-submitting').style.display = 'flex';

  try {
    const r = await fetch(`/api/instances/${instanceId}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, rationale, reviewer_id: reviewer }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || r.statusText);
    }

    // Hide HITL panel, add human decision step
    $('hitl-panel').classList.remove('visible');
    addHumanDecisionStep({ decision, rationale, reviewer_id: reviewer, timestamp: Date.now()/1000 });
    setStatus('running');
    startElapsedCounter();

  } catch(e) {
    alert('Decision failed: ' + e.message);
    $('decision-buttons').querySelectorAll('.decision-btn').forEach(b => b.disabled = false);
  } finally {
    $('header-submitting').style.display = 'none';
    $('footer-submitting').style.display = 'none';
  }
}

// ─── SSE stream ───────────────────────────────────────────────────
function connectStream() {
  const es = new EventSource(`/api/instances/${instanceId}/stream`);

  const eventTypes = [
    'step_started','step_completed','step_failed',
    'governance_decision','hitl_requested','hitl_resolved',
    'workflow_completed','workflow_failed'
  ];

  eventTypes.forEach(type => {
    es.addEventListener(type, e => {
      $('loading').style.display = 'none';
      let event;
      try { event = JSON.parse(e.data); } catch { return; }
      handleEvent(event);
    });
  });

  es.onerror = () => {
    // SSE reconnects automatically; only show error if never connected
    if ($('loading').style.display !== 'none') {
      $('loading').style.display = 'none';
      loadInstanceDirect();
    }
  };

  return es;
}

function handleEvent(event) {
  const evType = event.event || '';
  traceData.events.push(event);

  if (evType === 'step_started') {
    setStatus('running');
    if (!startTime) startElapsedCounter(event.timestamp);
    addOrUpdateStep(event);
  } else if (evType === 'step_completed') {
    setStatus('running');
    addOrUpdateStep(event);
  } else if (evType === 'step_failed') {
    addOrUpdateStep(event);
  } else if (evType === 'governance_decision' || evType === 'hitl_requested') {
    addOrUpdateStep(event);
    setStatus('suspended');
    stopElapsedCounter();
    loadWorkorder();
  } else if (evType === 'hitl_resolved') {
    $('hitl-panel').classList.remove('visible');
    if (event.decision) {
      addHumanDecisionStep(event);
    }
    setStatus('running');
    startElapsedCounter();
  } else if (evType === 'workflow_completed') {
    setStatus('completed');
    stopElapsedCounter();
    $('elapsed-counter').textContent = fmtElapsed(event.elapsed_seconds);
    showResult(event, 'completed');
    $('btn-export').disabled = false;
    checkLedgerIntegrity();
  } else if (evType === 'workflow_failed') {
    setStatus('failed');
    stopElapsedCounter();
    showResult(event, 'failed');
    $('btn-export').disabled = false;
    checkLedgerIntegrity();
  }

  $('footer-instance').textContent = instanceId;
}

// ─── Direct load (fallback / completed instances) ─────────────────
async function loadInstanceDirect() {
  try {
    const r = await fetch(`/api/instances/${instanceId}/chain`);
    if (!r.ok) {
      showError(`Instance not found: ${instanceId}`);
      return;
    }
    const chain = await r.json();
    const inst = chain.find(i => i.instance_id === instanceId) || chain[0];
    if (!inst) { showError('No instance data found'); return; }

    traceData.instance = inst;

    $('case-title').textContent = buildCaseTitle(inst);
    $('domain-label').textContent = inst.domain || '—';
    $('workflow-label').textContent = inst.workflow_type || '—';

    setStatus(inst.status);
    if (inst.elapsed_seconds) {
      $('elapsed-counter').textContent = fmtElapsed(inst.elapsed_seconds);
    }

    // Render steps from result summary
    if (inst.result) {
      const steps = inst.result.steps || inst.result.step_results || [];
      steps.forEach(s => addOrUpdateStep({
        event: 'step_completed',
        step_name: s.name || s.step_name,
        primitive: s.primitive,
        output: s.output || s.result,
        elapsed_ms: s.elapsed_ms,
      }));
    }

    if (inst.status === 'suspended') {
      setStatus('suspended');
      loadWorkorder();
    } else if (inst.status === 'completed') {
      showResult({ result: inst.result, elapsed_seconds: inst.elapsed_seconds }, 'completed');
      $('btn-export').disabled = false;
      checkLedgerIntegrity();
    } else if (inst.status === 'failed') {
      showResult({ error: inst.error }, 'failed');
      $('btn-export').disabled = false;
    }
  } catch(e) {
    showError('Failed to load instance: ' + e.message);
  }
}

// ─── Instance metadata ────────────────────────────────────────────
async function loadMeta() {
  try {
    const r = await fetch(`/api/instances/${instanceId}/chain`);
    if (!r.ok) return;
    const chain = await r.json();
    const inst = chain.find(i => i.instance_id === instanceId) || chain[0];
    if (!inst) return;
    traceData.instance = inst;
    $('case-title').textContent = buildCaseTitle(inst);
    $('domain-label').textContent = inst.domain || '—';
    $('workflow-label').textContent = inst.workflow_type || '—';
    $('footer-instance').textContent = instanceId;
    if (inst.created_at) startElapsedCounter(inst.created_at);
    if (inst.elapsed_seconds && inst.status !== 'running') {
      $('elapsed-counter').textContent = fmtElapsed(inst.elapsed_seconds);
    }
  } catch(e) { /* non-fatal */ }
}

function buildCaseTitle(inst) {
  const meta = inst.case_meta || {};
  const name = meta.applicant_name || meta.member_name || meta.customer_name || meta.name || '';
  const amount = meta.loan_amount || meta.amount || meta.claim_amount || '';
  const type = inst.workflow_type ? inst.workflow_type.replace(/_/g, ' ') : '';
  if (name && amount) return `Case: ${type} — ${name}, ${amount}`;
  if (name) return `Case: ${type} — ${name}`;
  return `Case: ${type || inst.instance_id}`;
}

// ─── Result panel ─────────────────────────────────────────────────
function showResult(event, status) {
  const panel = $('result-panel');
  panel.classList.add('visible');

  const iconEl   = $('result-icon');
  const labelEl  = $('result-label');
  const dispEl   = $('result-disposition');
  const elapsed  = $('result-elapsed');

  if (status === 'completed') {
    iconEl.textContent = '✓ COMPLETE';
    iconEl.className = 'result-icon-complete';
    labelEl.style.color = 'var(--green)';
  } else {
    iconEl.textContent = '✗ FAILED';
    iconEl.className = 'result-icon-failed';
    labelEl.style.color = 'var(--red)';
  }

  if (event.elapsed_seconds) {
    elapsed.textContent = `total elapsed ${fmtElapsed(event.elapsed_seconds)}`;
  }

  const result = event.result || {};
  const disposition = result.disposition || result.decision || result.outcome || result.recommendation || '';
  const summary = result.summary || result.brief || result.rationale || '';
  let html = '';
  if (disposition) html += `<div><strong>${escHtml(disposition)}</strong></div>`;
  if (summary) html += `<div style="margin-top:6px;font-size:11px;color:var(--dim);">${escHtml(summary)}</div>`;
  if (event.error) html += `<div style="color:var(--red);font-size:11px;">${escHtml(event.error)}</div>`;
  if (!html) html = '<div style="color:var(--muted);font-size:11px;">Workflow complete — see steps above for details.</div>';
  dispEl.innerHTML = html;

  if (status === 'completed') {
    labelEl.textContent = 'COMPLETE';
  } else {
    labelEl.textContent = 'FAILED';
  }
}

// ─── Ledger integrity check ───────────────────────────────────────
async function checkLedgerIntegrity() {
  try {
    const r = await fetch(`/api/instances/${instanceId}/verify`);
    if (!r.ok) return;
    const data = await r.json();
    const badge = $('verify-badge');
    if (data.valid === true) {
      badge.textContent = '✓ VERIFIED';
      badge.className = 'verified';
      badge.title = `Ledger hash chain verified — ${data.entry_count || ''} entries`;
    } else {
      badge.textContent = '✗ TAMPERED';
      badge.className = 'tampered';
      badge.title = `Hash chain invalid at entry ${data.first_invalid_entry}`;
    }
  } catch(e) { /* verify endpoint may not exist yet */ }
}

// ─── Export ───────────────────────────────────────────────────────
function exportTrace() {
  const snapshot = {
    exported_at: new Date().toISOString(),
    instance_id: instanceId,
    instance: traceData.instance,
    workorder: traceData.workorder,
    events: traceData.events,
  };
  const snapshotJson = JSON.stringify(snapshot, null, 2);
  const currentHTML = document.documentElement.outerHTML;
  const exported = currentHTML.replace(
    '</body>',
    `<script id="trace-snapshot" type="application/json">${escHtml(snapshotJson)}<\/script></body>`
  );

  const blob = new Blob([exported], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trace-${instanceId}-${Date.now()}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ─── Error ────────────────────────────────────────────────────────
function showError(msg) {
  const el = $('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
  $('loading').style.display = 'none';
}

// ─── Boot ─────────────────────────────────────────────────────────
(async () => {
  $('footer-instance').textContent = instanceId;
  await loadMeta();
  connectStream();
  setStatus('running');
})();
</script>
</body>
</html>"""


@router.get("/instances/{instance_id}/trace", response_class=Response)
async def get_trace_page(instance_id: str):
    """
    Serve the self-contained HTML trace page for a workflow instance.

    Three modes (handled client-side via SSE + REST):
      Watch mode  — live rendering as the workflow executes
      Input mode  — HITL review form when workflow is suspended
      Result mode — complete audit trace after workflow completes
    """
    coord = deps.get_coordinator()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    return Response(content=_TRACE_HTML, media_type="text/html")
