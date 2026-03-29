/**
 * Sentinel FCU — Fraud Operations Center
 * Kanban board + detail panel + completed cases panel
 */

// ── State ──────────────────────────────────────────────────────────
let selectedInstanceId = null;   // root instance_id of selected case
let backlogData = [];             // last backlog response
let chainPollingTimer = null;
let backlogPollingTimer = null;
let activeTab = 'detail';        // 'detail' | 'completed'
let activeDetailTab = 'member';  // 'member' | 'workflow' | 'decisions' | 'meta'
let pendingConfirmAction = null; // { type, taskId|instanceId, ... }
let availableCases = [];

// ── Init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadCasesForModal();
  await loadBacklog();
  backlogPollingTimer = setInterval(loadBacklog, 5000);
});

// ── Backlog ─────────────────────────────────────────────────────────
async function loadBacklog() {
  try {
    const res = await fetch('/api/backlog');
    if (!res.ok) return;
    backlogData = await res.json();
    renderKanban(backlogData);
    updateHeaderBadges(backlogData);
    document.getElementById('last-refresh').textContent =
      new Date().toLocaleTimeString();
  } catch (e) {
    console.warn('Backlog fetch failed:', e);
  }
}

// ── Kanban ──────────────────────────────────────────────────────────
const STAGES = ['triage', 'investigation', 'needs_review'];

function renderKanban(cases) {
  const active = cases.filter(c => !['completed', 'failed'].includes(c.stage));
  const completed = cases.filter(c => c.stage === 'completed');

  // Update completed count badge
  document.getElementById('completed-count-badge').textContent = completed.length;

  // Group active by stage
  const byStage = { triage: [], investigation: [], needs_review: [] };
  for (const c of active) {
    if (byStage[c.stage]) byStage[c.stage].push(c);
  }

  for (const stage of STAGES) {
    const col = document.getElementById(`col-${stage}`);
    const count = document.getElementById(`count-${stage}`);
    const items = byStage[stage] || [];
    count.textContent = items.length;
    if (items.length === 0) {
      col.innerHTML = `<div class="text-center text-slate-400 text-xs py-4">No cases</div>`;
    } else {
      col.innerHTML = items.map(c => renderCaseCard(c)).join('');
    }
  }

  // Re-render completed list if on that tab
  if (activeTab === 'completed') renderCompletedList(completed);

  // Highlight selected card
  if (selectedInstanceId) {
    document.querySelectorAll('.case-card').forEach(el => {
      el.classList.toggle('selected', el.dataset.id === selectedInstanceId);
    });
  }
}

function renderCaseCard(inst) {
  const meta = inst.case_meta || {};
  const isGate = inst.stage === 'needs_review';
  const isRunning = ['triage','investigation'].includes(inst.stage) && inst.status === 'running';
  const isSelected = inst.instance_id === selectedInstanceId;

  const fraudLabel = formatFraudType(meta.fraud_type || inst.domain);
  const memberName = meta.member_name || '—';
  const amount = meta.amount_at_risk != null ? '$' + fmtAmount(meta.amount_at_risk) : '';
  const priority = meta.priority || '';
  const sla = meta.sla_deadline ? fmtSla(meta.sla_deadline) : '';
  const riskScore = meta.risk_score != null ? Math.round(meta.risk_score * 100) : null;
  const caseId = meta.case_id || inst.correlation_id.slice(-10).toUpperCase();
  const age = formatTimeAgo(inst.created_at);

  const priorityColors = {
    critical: 'bg-red-50 text-red-700 border-red-200',
    high:     'bg-orange-50 text-orange-700 border-orange-200',
    medium:   'bg-yellow-50 text-yellow-700 border-yellow-200',
    low:      'bg-slate-50 text-slate-600 border-slate-200',
  };
  const priorityClass = priorityColors[priority] || 'bg-slate-50 text-slate-500 border-slate-200';

  return `
  <div class="case-card ${isGate ? 'gate-pending' : ''} ${isSelected ? 'selected' : ''}"
       data-id="${inst.instance_id}" onclick="selectCase('${inst.instance_id}')">

    <div class="flex items-start justify-between gap-2 mb-2">
      <span class="font-mono text-xs font-bold text-slate-700">${escHtml(caseId)}</span>
      <div class="flex items-center gap-1.5 shrink-0">
        ${riskScore != null ? `<span class="text-xs font-medium ${riskScore >= 80 ? 'text-red-600' : riskScore >= 60 ? 'text-amber-600' : 'text-green-600'}">${riskScore}%</span>` : ''}
        ${priority ? `<span class="text-xs border px-1.5 py-0.5 rounded ${priorityClass}">${priority}</span>` : ''}
      </div>
    </div>

    <div class="text-sm font-medium text-slate-800 truncate mb-0.5">${escHtml(memberName)}</div>
    <div class="text-xs text-slate-500 truncate mb-2">${escHtml(fraudLabel)}</div>

    <div class="flex items-center justify-between text-xs text-slate-400">
      <span>${amount ? `<span class="font-medium text-slate-600">${amount}</span> at risk` : ''}</span>
      <div class="flex items-center gap-2">
        ${isGate ? `<span class="flex items-center gap-1 text-amber-600 font-medium"><span class="pulse-dot w-1.5 h-1.5 rounded-full bg-amber-500"></span>Gate</span>` : isRunning ? `<span class="flex items-center gap-1 text-sky-500 font-medium"><span class="spinner"></span>Running</span>` : `<span>${age}</span>`}
        ${sla ? `<span title="SLA deadline">${sla}</span>` : ''}
      </div>
    </div>
  </div>`;
}

function renderCompletedList(completed) {
  const el = document.getElementById('completed-list');
  if (!completed || completed.length === 0) {
    el.innerHTML = `<div class="p-6 text-center text-slate-400 text-sm">No completed cases yet.</div>`;
    return;
  }
  el.innerHTML = completed.map(inst => {
    const meta = inst.case_meta || {};
    const caseId = meta.case_id || inst.correlation_id.slice(-10).toUpperCase();
    const member = meta.member_name || '—';
    const fraud = formatFraudType(meta.fraud_type || inst.domain);
    const amount = meta.amount_at_risk != null ? '$' + fmtAmount(meta.amount_at_risk) : '';
    const isSelected = inst.instance_id === selectedInstanceId;
    return `
    <div class="px-4 py-3 cursor-pointer hover:bg-slate-50 transition-colors ${isSelected ? 'bg-blue-50' : ''}"
         onclick="selectCase('${inst.instance_id}'); switchTab('detail')">
      <div class="flex items-center justify-between gap-2">
        <span class="font-mono text-xs font-bold text-slate-700">${escHtml(caseId)}</span>
        <span class="text-xs text-green-700 bg-green-50 border border-green-200 px-1.5 py-0.5 rounded">✓ done</span>
      </div>
      <div class="text-sm font-medium text-slate-700 truncate mt-0.5">${escHtml(member)}</div>
      <div class="flex items-center justify-between mt-0.5">
        <span class="text-xs text-slate-500">${escHtml(fraud)}</span>
        <span class="text-xs text-slate-400">${amount}</span>
      </div>
    </div>`;
  }).join('');
}

// ── Header badges ────────────────────────────────────────────────────
function updateHeaderBadges(cases) {
  const gates = cases.filter(c => c.has_pending_gate).length;
  const running = cases.filter(c => ['triage','investigation'].includes(c.stage)).length;

  const gb = document.getElementById('gate-badge');
  gb.classList.toggle('hidden', gates === 0);
  gb.classList.toggle('flex', gates > 0);
  document.getElementById('gate-badge-text').textContent = `${gates} pending gate${gates !== 1 ? 's' : ''}`;

  const rb = document.getElementById('running-badge');
  rb.classList.toggle('hidden', running === 0);
  rb.classList.toggle('flex', running > 0);
  document.getElementById('running-badge-text').textContent = `${running} running`;
}

// ── Tab switching ────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-detail').classList.toggle('hidden', tab !== 'detail');
  document.getElementById('tab-completed').classList.toggle('hidden', tab !== 'completed');
  document.getElementById('tab-detail-btn').classList.toggle('active', tab === 'detail');
  document.getElementById('tab-completed-btn').classList.toggle('active', tab === 'completed');

  if (tab === 'completed') {
    const completed = backlogData.filter(c => c.stage === 'completed');
    renderCompletedList(completed);
  }
  // also refresh completed count badge
  document.getElementById('completed-count-badge').textContent =
    backlogData.filter(c => c.stage === 'completed').length;
}

// ── Case selection ───────────────────────────────────────────────────
async function selectCase(instanceId) {
  selectedInstanceId = instanceId;
  activeDetailTab = 'member';  // reset inner tab on case change
  switchTab('detail');

  // Highlight card
  document.querySelectorAll('.case-card').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === instanceId);
  });

  // Stop old polling
  if (chainPollingTimer) { clearInterval(chainPollingTimer); chainPollingTimer = null; }

  // Load chain immediately
  const chain = await loadChain(instanceId);

  // Always poll: the root may already be complete while child delegations are
  // still running or suspended.  Stop only when every instance is terminal.
  chainPollingTimer = setInterval(async () => {
    if (selectedInstanceId !== instanceId) { clearInterval(chainPollingTimer); chainPollingTimer = null; return; }
    const c = await loadChain(instanceId);
    if (!c || c.every(i => isTerminal(i.status))) {
      clearInterval(chainPollingTimer);
      chainPollingTimer = null;
      await loadBacklog();
    }
  }, 2000);
}

async function loadChain(instanceId) {
  try {
    const res = await fetch(`/api/instances/${instanceId}/chain`);
    if (!res.ok) return null;
    const chain = await res.json();
    if (selectedInstanceId === instanceId) renderDetail(chain);
    return chain;
  } catch (e) {
    console.warn('Chain fetch failed:', e);
    return null;
  }
}

// ── Detail Panel ─────────────────────────────────────────────────────
function renderDetail(chain) {
  const panel = document.getElementById('tab-detail');
  if (!chain || chain.length === 0) {
    panel.innerHTML = `<div class="p-8 text-slate-400 text-sm">No data found.</div>`;
    return;
  }

  const root = chain[0];
  const meta = root.case_meta || {};
  // Prefer case_input from root (for completed), fallback to first chain member with it
  const caseInput = root.case_input
    || chain.find(i => i.case_input && Object.keys(i.case_input).length > 0)?.case_input
    || {};

  const suspended = chain.find(i => i.status === 'suspended' && i.pending_task);
  const anySuspended = chain.find(i => i.status === 'suspended');
  const isLive = chain.some(i => ['running','created'].includes(i.status));
  const caseId = meta.case_id || root.correlation_id.slice(-10).toUpperCase();

  panel.innerHTML = `
  <div class="p-4 space-y-4">

    <!-- Case header -->
    <div class="bg-white rounded-xl border border-slate-200 p-4">
      <div class="flex items-start justify-between gap-3 mb-3">
        <div>
          <div class="flex items-center gap-2 mb-0.5">
            <span class="font-mono font-bold text-navy text-base">${escHtml(caseId)}</span>
            ${meta.priority ? `<span class="text-xs border px-1.5 py-0.5 rounded ${priorityClass(meta.priority)}">${meta.priority}</span>` : ''}
          </div>
          <div class="text-sm text-slate-500">${escHtml(formatFraudType(meta.fraud_type || root.domain))}</div>
        </div>
        <div class="text-right text-xs text-slate-400 shrink-0">
          <div>Started ${formatTimeAgo(root.created_at)}</div>
          ${isLive ? `<div class="text-sky-500 flex items-center gap-1 justify-end"><span class="pulse-dot w-1.5 h-1.5 rounded-full bg-sky-400"></span>Processing</div>` : ''}
        </div>
      </div>

      <!-- Stage flow -->
      <div class="flex flex-wrap items-center gap-1.5 text-xs">
        ${chain.map(inst => {
          const si = statusInfo(inst.status);
          const isRunning = ['running','created'].includes(inst.status);
          return `<span class="flex items-center gap-1 ${si.text} ${si.bg} border ${si.border} px-2 py-0.5 rounded-full">
            ${isRunning ? `<span class="pulse-dot w-1.5 h-1.5 rounded-full bg-current"></span>` : si.icon}
            ${escHtml(workflowLabel(inst.workflow_type, inst.domain))}
          </span>`;
        }).join('<span class="text-slate-300">→</span>')}
      </div>
    </div>

    <!-- Triage summary -->
    ${renderTriageSummary(chain)}

    <!-- Gate banner -->
    ${suspended ? renderGateBanner(suspended) : ''}

    <!-- Content tabs -->
    <div class="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div class="flex border-b border-slate-100 px-2 bg-slate-50">
        <button class="tab-btn active" id="dt-member-btn" onclick="switchDetailTab('member')">Member &amp; Case</button>
        <button class="tab-btn" id="dt-workflow-btn" onclick="switchDetailTab('workflow')">Workflow</button>
        <button class="tab-btn" id="dt-decisions-btn" onclick="switchDetailTab('decisions')">Decisions</button>
        <button class="tab-btn" id="dt-meta-btn" onclick="switchDetailTab('meta')">Metadata</button>
      </div>

      <div id="dt-member" class="p-4">
        ${renderMemberTab(caseInput, meta)}
      </div>
      <div id="dt-workflow" class="p-4 hidden">
        ${chain.map(inst => renderInstanceWorkflow(inst)).join('')}
      </div>
      <div id="dt-decisions" class="p-4 hidden">
        ${renderDecisionsTab(chain)}
      </div>
      <div id="dt-meta" class="p-4 hidden">
        ${renderMetaTab(chain, root)}
      </div>
    </div>

    <!-- Action buttons -->
    ${renderActionButtons(chain, suspended)}

  </div>`;

  // Restore whichever inner tab was active before the re-render
  if (activeDetailTab !== 'member') {
    switchDetailTab(activeDetailTab);
  }
}

function switchDetailTab(tab) {
  activeDetailTab = tab;
  ['member','workflow','decisions','meta'].forEach(t => {
    document.getElementById(`dt-${t}`)?.classList.toggle('hidden', t !== tab);
    document.getElementById(`dt-${t}-btn`)?.classList.toggle('active', t === tab);
  });
}

// ── Triage Summary ───────────────────────────────────────────────────
function renderTriageSummary(chain) {
  const triage = chain.find(i => i.workflow_type === 'fraud_triage');
  if (!triage) return '';
  const steps = triage.result?.steps || [];
  const isRunning = ['running','created'].includes(triage.status);
  if (isRunning && steps.length === 0) {
    return `
    <div class="bg-sky-50 border border-sky-200 rounded-xl p-3 flex items-center gap-2 text-sm text-sky-700">
      <span class="pulse-dot w-2 h-2 rounded-full bg-sky-400 shrink-0"></span>
      Triage running — classifying fraud type and assessing risk…
    </div>`;
  }
  if (steps.length === 0) return '';

  // Pull key classify steps
  const classifyFraud = steps.find(s => s.step_name === 'classify_fraud_type' || (s.primitive === 'classify' && s.step_name?.includes('fraud')));
  const classifyCustomer = steps.find(s => s.step_name?.includes('customer') || s.step_name?.includes('member'));
  const priority = steps.find(s => s.step_name?.includes('priority') || s.step_name?.includes('assess'));

  // Routing target = first specialist in chain
  const specialist = chain.find(i => i !== triage);
  const routedTo = specialist ? workflowLabel(specialist.workflow_type, specialist.domain) : null;

  const rows = [];
  if (classifyFraud?.category) rows.push(['Fraud Type', classifyFraud.category, classifyFraud.confidence]);
  if (classifyCustomer?.category) rows.push(['Customer', classifyCustomer.category, classifyCustomer.confidence]);
  if (priority?.category) rows.push(['Priority', priority.category, priority.confidence]);

  if (rows.length === 0 && !routedTo) return '';

  return `
  <div class="bg-white border border-slate-200 rounded-xl p-4">
    <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Triage Result</div>
    <div class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm mb-3">
      ${rows.map(([label, val, conf]) => `
        <div class="text-slate-500 text-xs">${label}</div>
        <div class="flex items-center gap-1.5">
          <span class="font-medium text-slate-800">${escHtml(formatFraudType(val))}</span>
          ${conf != null ? `<span class="text-xs text-slate-400">${Math.round(conf*100)}%</span>` : ''}
        </div>`).join('')}
    </div>
    ${routedTo ? `
    <div class="flex items-center gap-1.5 text-xs text-slate-500 pt-2 border-t border-slate-100">
      <span>Routed to</span>
      <span class="font-medium text-slate-700">→ ${escHtml(routedTo)}</span>
      ${specialist?.status === 'suspended' ? '<span class="text-amber-600 font-medium">· awaiting approval</span>' : ''}
      ${specialist?.status === 'running' ? '<span class="pulse-dot w-1.5 h-1.5 rounded-full bg-sky-400 inline-block"></span>' : ''}
    </div>` : ''}
  </div>`;
}

// ── Gate Banner ──────────────────────────────────────────────────────
function renderGateBanner(inst) {
  const task = inst.pending_task;
  if (!task) return '';
  return `
  <div class="gate-banner bg-amber-50 border-2 border-amber-400 rounded-xl p-4">
    <div class="flex items-center gap-2 mb-1">
      <span class="text-amber-500 text-base">⚠</span>
      <span class="font-semibold text-amber-800 text-sm">Awaiting Analyst Approval</span>
    </div>
    <div class="text-xs text-amber-700">
      ${escHtml(workflowLabel(inst.workflow_type, inst.domain))} ·
      Queue: <code class="bg-amber-100 px-1 rounded">${escHtml(task.queue || 'fraud_analyst_review')}</code>
    </div>
  </div>`;
}

// ── Member & Case Tab ────────────────────────────────────────────────
function renderMemberTab(input, meta) {
  const member = input.get_member || {};
  const account = input.get_account || {};
  const alert = input.get_alert || {};
  const card = input.get_card || {};
  const activity = input.get_triggering_activity || {};
  const regE = input.get_reg_e_info || {};
  const priorAlerts = input.get_prior_alerts || {};

  if (!member.full_name && !meta.member_name) {
    return `<p class="text-sm text-slate-400 italic">Case data will appear after the first retrieve step completes.</p>`;
  }

  const name = member.full_name || meta.member_name || '—';
  const sections = [];

  // Member section
  sections.push(`
  <div class="mb-4">
    <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Member</div>
    <table class="data-table">
      ${row('Name', name)}
      ${row('Member ID', member.member_id)}
      ${row('DOB', member.date_of_birth)}
      ${row('SSN', member.ssn_masked)}
      ${row('Address', member.address)}
      ${row('Phone', member.phone)}
      ${row('Email', member.email)}
      ${row('Member Since', member.member_since)}
      ${row('Type', member.member_type)}
      ${row('Employer', member.employer)}
      ${row('Title', member.title)}
      ${row('Annual Pay', member.annual_pay ? '$' + fmtAmount(member.annual_pay) : null)}
      ${row('Clearance', member.clearance)}
      ${row('Risk Rating', member.risk_rating)}
      ${row('Branch', member.branch_affiliation)}
      ${row('Rank', member.rank_retired)}
      ${row('Pension/mo', member.pension_monthly ? '$' + fmtAmount(member.pension_monthly) : null)}
      ${row('VA Disability/mo', member.va_disability_monthly ? '$' + fmtAmount(member.va_disability_monthly) : null)}
    </table>
  </div>`);

  // Account section
  if (account.account_number_masked) {
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Account</div>
      <table class="data-table">
        ${row('Account', account.account_number_masked)}
        ${row('Type', account.account_type)}
        ${row('Product', account.product)}
        ${row('Opened', account.opened)}
        ${row('Status', account.status)}
        ${row('Avg Balance (6mo)', account.average_balance_6mo ? '$' + fmtAmount(account.average_balance_6mo) : null)}
        ${row('Avg Deposits (6mo)', account.avg_deposits_6mo ? '$' + fmtAmount(account.avg_deposits_6mo) : null)}
        ${row('Avg Withdrawals (6mo)', account.avg_withdrawals_6mo ? '$' + fmtAmount(account.avg_withdrawals_6mo) : null)}
        ${row('Primary Deposit', account.primary_deposit_source)}
        ${row('Geo Pattern', account.geographic_pattern)}
        ${row('Zelle History', account.zelle_history)}
        ${row('Wire History', account.wire_history)}
      </table>
    </div>`);
  }

  // Alert section
  if (alert.alert_id) {
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Alert</div>
      <table class="data-table">
        ${row('Alert ID', alert.alert_id)}
        ${row('Type', alert.alert_type)}
        ${row('Source', alert.alert_source)}
        ${row('Rule', alert.monitoring_rule)}
        ${row('Risk Score', alert.risk_score)}
        ${row('Priority', alert.priority)}
        ${row('SLA Deadline', alert.sla_deadline)}
      </table>
    </div>`);
  }

  // Card section
  if (card.card_number_masked) {
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Card</div>
      <table class="data-table">
        ${row('Card', card.card_number_masked)}
        ${row('Type', card.card_type)}
        ${row('Product', card.product)}
        ${row('Issued', card.issued)}
        ${row('Expiry', card.expiry)}
        ${row('Status', card.card_status)}
        ${card.last_legitimate_use ? row('Last Legit Use', `${card.last_legitimate_use.date} ${card.last_legitimate_use.time} — ${card.last_legitimate_use.merchant} (${card.last_legitimate_use.method})`) : ''}
      </table>
    </div>`);
  }

  // Triggering activity
  if (activity.suspicious_transactions || activity.payments || activity.scam_type) {
    const txns = activity.suspicious_transactions || activity.payments || [];
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
        Triggering Activity
        ${activity.total_approved != null ? `<span class="ml-2 text-red-600 normal-case font-semibold">$${fmtAmount(activity.total_approved)} at risk</span>` : ''}
        ${activity.total_amount != null ? `<span class="ml-2 text-red-600 normal-case font-semibold">$${fmtAmount(activity.total_amount)} at risk</span>` : ''}
      </div>
      ${activity.scam_type ? `
        <table class="data-table mb-3">
          ${row('Scam Type', activity.scam_type)}
          ${row('Platform', activity.contact_platform)}
          ${row('First Contact', activity.first_contact_date)}
          ${activity.scammer_persona ? row('Scammer Persona', activity.scammer_persona.name_used + ' — ' + activity.scammer_persona.claimed_identity) : ''}
        </table>
        ${activity.escalation_timeline ? `
          <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Timeline</div>
          <div class="space-y-1 mb-3">
            ${activity.escalation_timeline.map(e => `
              <div class="flex gap-2 text-xs">
                <span class="text-slate-400 shrink-0 w-24">${e.date}</span>
                <span class="text-slate-600">${escHtml(e.event)}</span>
              </div>`).join('')}
          </div>` : ''}
      ` : ''}
      ${txns.length > 0 ? `
        <div class="overflow-x-auto">
          <table class="data-table text-xs">
            <thead><tr>
              <th>Date/Time</th><th>Merchant</th>
              <th class="text-right">Amount</th>
              <th>Method</th><th>Location</th><th>Status</th>
            </tr></thead>
            <tbody>
              ${txns.map(t => `<tr>
                <td>${t.date || ''} ${t.time || ''}</td>
                <td>${escHtml(t.merchant || t.description || '')}</td>
                <td class="text-right font-medium ${(t.amount||0) < 0 ? 'text-red-600' : 'text-slate-700'}">$${fmtAmount(Math.abs(t.amount || 0))}</td>
                <td>${escHtml(t.method || t.type || '')}</td>
                <td>${escHtml(t.ip_geolocation || t.channel || '')}</td>
                <td class="${t.status === 'approved' || t.status === 'posted' ? 'text-red-600' : 'text-slate-500'}">${escHtml(t.status || '')}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>` : ''}
    </div>`);
  }

  // Reg E
  if (regE.total_unauthorized_amount != null) {
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Reg E / Compliance</div>
      <table class="data-table">
        ${row('Total Unauthorized', '$' + fmtAmount(regE.total_unauthorized_amount))}
        ${row('Net Loss Exposure', '$' + fmtAmount(regE.net_loss_exposure))}
        ${row('Member Liability', regE.member_liability)}
        ${row('Notification Deadline', regE.notification_deadline)}
        ${row('Provisional Credit Deadline', regE.provisional_credit_deadline)}
      </table>
    </div>`);
  }

  // Prior alerts
  const priorList = priorAlerts.prior_alerts || [];
  const priorClaims = priorAlerts.prior_fraud_claims || [];
  if (priorList.length > 0 || priorClaims.length > 0) {
    sections.push(`
    <div class="mb-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Prior Alerts & Claims</div>
      ${priorList.length > 0 ? `<div class="text-xs text-slate-500 mb-1">${priorList.length} prior alert(s)</div>` : '<div class="text-xs text-slate-400 italic">No prior alerts</div>'}
      ${priorClaims.length > 0 ? `<div class="text-xs text-slate-500">${priorClaims.length} prior claim(s)</div>` : '<div class="text-xs text-slate-400 italic">No prior claims</div>'}
    </div>`);
  }

  return sections.join('') || `<p class="text-sm text-slate-400 italic">No case data available.</p>`;
}

// ── Workflow Tab ─────────────────────────────────────────────────────
function renderInstanceWorkflow(inst) {
  const steps = inst.result?.steps || [];
  const isRunning = ['running','created'].includes(inst.status);
  const depth = inst.lineage.length;
  const indent = depth > 0 ? 'ml-4' : '';

  return `
  <div class="mb-4 ${indent}">
    <div class="flex items-center gap-2 mb-2">
      ${depth > 0 ? '<span class="text-slate-300">└</span>' : ''}
      <span class="font-medium text-slate-700 text-sm">${escHtml(workflowLabel(inst.workflow_type, inst.domain))}</span>
      <span class="text-xs ${statusInfo(inst.status).text} ${statusInfo(inst.status).bg} border ${statusInfo(inst.status).border} px-2 py-0.5 rounded-full">
        ${isRunning ? `<span class="pulse-dot inline-block w-1.5 h-1.5 rounded-full bg-current mr-1"></span>` : statusInfo(inst.status).icon + ' '}${inst.status}
      </span>
      ${inst.elapsed_seconds > 0 ? `<span class="text-xs text-slate-400">${inst.elapsed_seconds.toFixed(1)}s</span>` : ''}
    </div>

    ${inst.error ? `<div class="text-xs text-red-600 bg-red-50 border border-red-100 rounded p-2 mb-2">${escHtml(inst.error)}</div>` : ''}

    ${isRunning && steps.length === 0 ? `
    <div class="flex items-center gap-2 text-slate-400 text-xs py-2 px-3 bg-slate-50 rounded">
      <span class="pulse-dot w-2 h-2 rounded-full bg-sky-400"></span>
      Executing...
    </div>` : ''}

    ${steps.map((step, i) => renderStepCard(step, inst.instance_id, i)).join('')}
  </div>`;
}

function renderStepCard(step, instId, idx) {
  const prim = step.primitive || 'unknown';
  const cardId = `sc-${instId}-${idx}`;
  const pi = primInfo(prim);

  return `
  <div class="step-card">
    <div class="step-header" onclick="toggleStep('${cardId}')">
      <span class="text-xs font-semibold px-2 py-0.5 rounded ${pi.text} ${pi.bg}">${prim}</span>
      <span class="text-sm text-slate-700 flex-1">${escHtml(step.step_name || 'step')}</span>
      ${confPill(step.confidence)}
      <span class="text-slate-300 text-xs" id="${cardId}-icon">▼</span>
    </div>
    <div id="${cardId}" class="hidden px-4 py-3 bg-white border-t border-slate-100">
      ${renderStepBody(step)}
    </div>
  </div>`;
}

function toggleStep(id) {
  const el = document.getElementById(id);
  const icon = document.getElementById(id + '-icon');
  if (!el) return;
  const open = !el.classList.contains('hidden');
  el.classList.toggle('hidden', open);
  if (icon) icon.textContent = open ? '▼' : '▲';
}

function renderStepBody(step) {
  const prim = step.primitive;
  if (prim === 'classify') return `
    <div class="flex items-center gap-2 mb-2">
      <span class="font-semibold text-slate-800">${escHtml(step.category || '—')}</span>
      ${confPill(step.confidence)}
    </div>
    ${confBar(step.confidence)}`;

  if (prim === 'retrieve') {
    const sources = step.sources || [];
    return sources.length === 0 ? '<span class="text-sm text-slate-400">No sources recorded</span>' : `
    <div class="flex flex-wrap gap-1.5">
      ${sources.map(s => `<span class="text-xs font-mono bg-purple-50 text-purple-700 border border-purple-200 px-2 py-0.5 rounded">${escHtml(s)}</span>`).join('')}
    </div>`;
  }

  if (prim === 'investigate') return `
    ${step.finding ? `<p class="text-sm text-slate-700 leading-relaxed mb-2">${escHtml(step.finding)}</p>` : ''}
    ${confBar(step.confidence)}
    ${step.evidence_flags?.length ? `<div class="mt-2 flex flex-wrap gap-1">${step.evidence_flags.map(f=>`<span class="text-xs bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded">${escHtml(f)}</span>`).join('')}</div>` : ''}`;

  if (prim === 'think') return `
    ${step.decision ? `<div class="flex items-center gap-2 mb-2"><span class="font-semibold text-slate-800">${escHtml(step.decision)}</span>${confPill(step.confidence)}</div>` : ''}
    ${confBar(step.confidence)}
    ${step.reasoning ? `<details class="mt-2"><summary class="text-xs text-slate-400 cursor-pointer hover:text-slate-600">▼ Reasoning</summary><p class="mt-2 text-sm text-slate-600 leading-relaxed">${escHtml(step.reasoning)}</p></details>` : ''}`;

  if (prim === 'verify') {
    const ok = step.conforms;
    return `<div class="flex items-center gap-3 mb-2">
      <span class="text-xs font-medium border px-2.5 py-1 rounded-full ${ok ? 'text-green-700 bg-green-50 border-green-200' : 'text-red-700 bg-red-50 border-red-200'}">
        ${ok === true ? '✓ Conforms' : ok === false ? '✗ Non-conforming' : '— Unknown'}
      </span>
      ${step.violations != null ? `<span class="text-xs text-slate-500">${step.violations} violation${step.violations !== 1 ? 's' : ''}</span>` : ''}
    </div>${confBar(step.confidence)}`;
  }

  if (prim === 'challenge') {
    const ok = step.survives;
    return `<div class="flex items-center gap-3">
      <span class="text-xs font-medium border px-2.5 py-1 rounded-full ${ok ? 'text-green-700 bg-green-50 border-green-200' : 'text-red-700 bg-red-50 border-red-200'}">
        ${ok === true ? '✓ Survives' : ok === false ? '✗ Did not survive' : '— Unknown'}
      </span>
      ${step.vulnerabilities != null ? `<span class="text-xs text-slate-500">${step.vulnerabilities} vulnerabilit${step.vulnerabilities !== 1 ? 'ies' : 'y'}</span>` : ''}
    </div>`;
  }

  if (prim === 'generate') {
    const preview = step.artifact_preview || step.artifact || '';
    return preview ? `<pre class="text-xs text-slate-600 bg-slate-50 border border-slate-100 rounded p-3 whitespace-pre-wrap overflow-x-auto max-h-64">${escHtml(String(preview).slice(0, 500))}${String(preview).length > 500 ? '…' : ''}</pre>` : '';
  }

  return `<pre class="text-xs text-slate-500">${escHtml(JSON.stringify(step, null, 2))}</pre>`;
}

// ── Decisions Tab ────────────────────────────────────────────────────
function renderDecisionsTab(chain) {
  const sections = [];

  for (const inst of chain) {
    const steps = inst.result?.steps || [];
    const narratives = steps.filter(s => s.primitive === 'generate' && (s.artifact || s.artifact_preview));
    const decisions = steps.filter(s => ['think','investigate','verify','challenge'].includes(s.primitive));

    if (decisions.length === 0 && narratives.length === 0) continue;

    sections.push(`<div class="mb-5">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <span>${escHtml(workflowLabel(inst.workflow_type, inst.domain))}</span>
      </div>`);

    for (const step of decisions) {
      const prim = step.primitive;
      let content = '';
      if (prim === 'think') {
        content = `
          ${step.decision ? `<div class="font-semibold text-slate-800 mb-1">${escHtml(step.decision)}</div>` : ''}
          ${step.reasoning ? `<div class="text-sm text-slate-600 leading-relaxed">${escHtml(step.reasoning)}</div>` : ''}`;
      } else if (prim === 'investigate') {
        content = `<div class="text-sm text-slate-700 leading-relaxed">${escHtml(step.finding || '')}</div>`;
      } else if (prim === 'verify') {
        content = `<div class="text-sm">${step.conforms === true ? '✓ Conforms' : '✗ Non-conforming'} — ${step.violations || 0} violations</div>`;
      } else if (prim === 'challenge') {
        content = `<div class="text-sm">${step.survives ? '✓ Survives challenge' : '✗ Did not survive'} — ${step.vulnerabilities || 0} vulnerabilities</div>`;
      }
      if (content) {
        sections.push(`
          <div class="mb-3">
            <div class="text-xs font-medium text-slate-400 mb-1">${prim} · ${escHtml(step.step_name || '')}</div>
            ${content}
          </div>`);
      }
    }

    for (const step of narratives) {
      const artifact = step.artifact || step.artifact_preview || '';
      sections.push(`
        <div class="mb-4">
          <div class="text-xs font-medium text-slate-400 mb-1">Generated · ${escHtml(step.step_name || '')}</div>
          <div class="narrative">${escHtml(String(artifact))}</div>
        </div>`);
    }

    sections.push('</div>');
  }

  return sections.length > 0 ? sections.join('') :
    `<p class="text-sm text-slate-400 italic">No decisions or narratives yet.</p>`;
}

// ── Metadata Tab ─────────────────────────────────────────────────────
function renderMetaTab(chain, root) {
  const totalElapsed = chain.reduce((s, i) => s + (i.elapsed_seconds || 0), 0);
  const waitTime = chain.filter(i => isTerminal(i.status)).length > 0
    ? (chain.find(i => isTerminal(i.status))?.updated_at || root.created_at) - root.created_at
    : Date.now() / 1000 - root.created_at;

  return `
  <div class="mb-4">
    <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Timing</div>
    <table class="data-table">
      ${row('Started', new Date(root.created_at * 1000).toLocaleString())}
      ${row('Total LLM Processing', totalElapsed.toFixed(1) + 's')}
      ${row('Wall Clock Time', waitTime.toFixed(0) + 's')}
    </table>
  </div>
  <div class="mb-4">
    <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Delegation Chain</div>
    <table class="data-table">
      <thead><tr><th>Workflow</th><th>Domain</th><th>Status</th><th>Steps</th><th>Time</th><th>Tier</th></tr></thead>
      <tbody>
        ${chain.map(inst => `<tr>
          <td class="font-mono text-xs">${escHtml(inst.workflow_type)}</td>
          <td class="text-xs">${escHtml(inst.domain)}</td>
          <td><span class="text-xs ${statusInfo(inst.status).text}">${inst.status}</span></td>
          <td class="text-right">${inst.step_count}</td>
          <td>${inst.elapsed_seconds ? inst.elapsed_seconds.toFixed(1) + 's' : '—'}</td>
          <td class="text-xs text-slate-400">${inst.governance_tier}</td>
        </tr>`).join('')}
      </tbody>
    </table>
  </div>
  <div class="mb-4">
    <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Instance IDs</div>
    <table class="data-table">
      ${chain.map(inst => row(workflowLabel(inst.workflow_type, inst.domain), inst.instance_id)).join('')}
    </table>
  </div>`;
}

// ── Action Buttons ───────────────────────────────────────────────────
function renderActionButtons(chain, suspended) {
  const allTerminal = chain.every(i => isTerminal(i.status));
  if (allTerminal) return '';  // completed/terminated — no actions

  // Find suspended instance with a governance gate (pending_task = analyst action needed)
  const gatedSuspended = chain.find(i => i.status === 'suspended' && i.pending_task);
  if (gatedSuspended) {
    const task = gatedSuspended.pending_task;
    const taskId = task.task_id;
    const corrId = gatedSuspended.correlation_id;
    const instId = gatedSuspended.instance_id;

    return `
    <div class="bg-white rounded-xl border border-slate-200 p-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Analyst Actions</div>
      <div class="flex flex-wrap gap-2">
        <button onclick="openConfirm('approve', '${taskId}', '${corrId}')"
          class="action-btn flex items-center gap-2 px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors">
          ✓ Approve
        </button>
        <button onclick="openConfirm('reject', '${taskId}', '${corrId}')"
          class="action-btn flex items-center gap-2 px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition-colors">
          ✗ Reject
        </button>
        <button onclick="openConfirm('terminate', null, null, '${instId}')"
          class="action-btn flex items-center gap-2 px-4 py-2 bg-slate-600 text-white text-sm font-medium rounded-lg hover:bg-slate-700 transition-colors">
          ■ Terminate
        </button>
      </div>
    </div>`;
  }

  // Suspended but no governance gate — waiting for child delegations; only terminate available
  const delegationSuspended = chain.find(i => i.status === 'suspended');
  if (delegationSuspended) {
    return `
    <div class="bg-white rounded-xl border border-slate-200 p-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Analyst Actions</div>
      <div class="flex gap-2 items-center">
        <span class="text-xs text-slate-400 italic">Waiting for sub-investigations…</span>
        <button onclick="openConfirm('terminate', null, null, '${delegationSuspended.instance_id}')"
          class="action-btn flex items-center gap-2 px-4 py-2 bg-slate-600 text-white text-sm font-medium rounded-lg hover:bg-slate-700 transition-colors">
          ■ Terminate
        </button>
      </div>
    </div>`;
  }

  // Running — only terminate available
  const runningInst = chain.find(i => ['running','created'].includes(i.status));
  if (runningInst) {
    return `
    <div class="bg-white rounded-xl border border-slate-200 p-4">
      <div class="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Analyst Actions</div>
      <div class="flex gap-2">
        <button onclick="openConfirm('terminate', null, null, '${runningInst.instance_id}')"
          class="action-btn flex items-center gap-2 px-4 py-2 bg-slate-600 text-white text-sm font-medium rounded-lg hover:bg-slate-700 transition-colors">
          ■ Terminate
        </button>
      </div>
    </div>`;
  }

  return '';
}

// ── Confirm Modal ────────────────────────────────────────────────────
function openConfirm(type, taskId, corrId, instanceId) {
  pendingConfirmAction = { type, taskId, corrId, instanceId };

  const titles = { approve: 'Approve Governance Gate', reject: 'Reject Governance Gate',
    terminate: 'Terminate Workflow', resume: 'Resume Workflow' };
  const msgs = {
    approve: 'Approving will finalize the workflow and execute any downstream delegations.',
    reject: 'Rejecting will terminate this workflow instance. This cannot be undone.',
    terminate: 'Terminating will stop all processing immediately. This cannot be undone.',
    resume: 'Resume will re-run the workflow from the suspension point with optional input.',
  };
  const btnClasses = {
    approve: 'bg-green-600 hover:bg-green-700',
    reject: 'bg-red-600 hover:bg-red-700',
    terminate: 'bg-slate-700 hover:bg-slate-800',
    resume: 'bg-sky-600 hover:bg-sky-700',
  };
  const btnLabels = { approve: '✓ Approve', reject: '✗ Reject', terminate: '■ Terminate', resume: '▶ Resume' };

  document.getElementById('confirm-title').textContent = titles[type];
  document.getElementById('confirm-message').textContent = msgs[type];
  document.getElementById('confirm-notes-label').textContent = type === 'reject' ? 'Rejection Reason (required)' : 'Notes';
  document.getElementById('confirm-notes').value = '';
  document.getElementById('confirm-analyst').value = 'analyst';
  const btn = document.getElementById('confirm-action-btn');
  btn.className = `px-5 py-2 text-white text-sm font-medium rounded-lg transition-colors ${btnClasses[type]}`;
  btn.textContent = btnLabels[type];
  btn.disabled = false;

  document.getElementById('confirm-modal').classList.remove('hidden');
}

function closeConfirmModal() {
  document.getElementById('confirm-modal').classList.add('hidden');
  pendingConfirmAction = null;
}

function confirmReset() {
  if (!confirm('Reset all cases? This will delete all workflow instances from the database and cannot be undone.')) return;
  resetAll();
}

async function resetAll() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    if (!res.ok) throw new Error((await res.json().catch(() => ({detail: res.statusText}))).detail);
    selectedInstanceId = null;
    activeDetailTab = 'member';
    document.getElementById('tab-detail').innerHTML =
      `<div class="p-8 text-slate-400 text-sm text-center">Select a case to view details.</div>`;
    await loadBacklog();
    showToast('All cases cleared');
  } catch (e) {
    showToast(`Reset failed: ${e.message}`, 'error');
  }
}

async function executeConfirm() {
  if (!pendingConfirmAction) return;
  const { type, taskId, instanceId } = pendingConfirmAction;
  const analyst = document.getElementById('confirm-analyst').value.trim() || 'analyst';
  const notes = document.getElementById('confirm-notes').value.trim();

  if (type === 'reject' && !notes) { showToast('Rejection reason is required', 'error'); return; }

  const btn = document.getElementById('confirm-action-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Processing…`;

  closeConfirmModal();

  try {
    let endpoint, body;
    if (type === 'approve') {
      endpoint = `/api/tasks/${taskId}/approve`;
      body = { approver: analyst, notes };
    } else if (type === 'reject') {
      endpoint = `/api/tasks/${taskId}/reject`;
      body = { rejector: analyst, reason: notes };
    } else if (type === 'terminate') {
      endpoint = `/api/instances/${instanceId}/terminate`;
      body = { reason: notes || 'Terminated by analyst' };
    } else if (type === 'resume') {
      endpoint = `/api/instances/${instanceId}/resume`;
      body = { external_input: {}, resume_nonce: '' };
    }

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Unknown error');
    }

    const toastMsg = { approve: 'Approved — workflow resuming…', reject: 'Rejected',
      terminate: 'Terminated', resume: 'Resumed — workflow processing…' };
    showToast(toastMsg[type]);

    // Restart chain polling
    if (selectedInstanceId) {
      if (chainPollingTimer) clearInterval(chainPollingTimer);
      chainPollingTimer = setInterval(async () => {
        const c = await loadChain(selectedInstanceId);
        if (!c || c.every(i => isTerminal(i.status))) {
          clearInterval(chainPollingTimer); chainPollingTimer = null;
          await loadBacklog();
        }
      }, 2000);
      await loadChain(selectedInstanceId);
    }
    await loadBacklog();

  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  }
}

// ── Submit Modal ─────────────────────────────────────────────────────
async function loadCasesForModal() {
  try {
    const res = await fetch('/api/cases');
    if (!res.ok) return;
    availableCases = await res.json();
    const sel = document.getElementById('submit-case-file');
    sel.innerHTML = availableCases.map(c =>
      `<option value="${escAttr(c.path)}">${c.case_id} — ${formatFraudType(c.fraud_type)}</option>`
    ).join('');
    updateCaseDesc();
    sel.addEventListener('change', updateCaseDesc);
  } catch (e) {
    console.warn('Could not load cases:', e);
  }
}

function updateCaseDesc() {
  const val = document.getElementById('submit-case-file').value;
  const found = availableCases.find(c => c.path === val);
  document.getElementById('submit-case-desc').textContent = found?.description || '';
}

function openSubmitModal() {
  // Filter out cases already in the system (any stage except failed)
  const activeCaseIds = new Set(
    backlogData
      .filter(c => c.stage !== 'failed')
      .map(c => c.case_meta?.case_id)
      .filter(Boolean)
  );
  const sel = document.getElementById('submit-case-file');
  const filtered = availableCases.filter(c => !activeCaseIds.has(c.case_id));
  sel.innerHTML = filtered.length
    ? filtered.map(c => `<option value="${escAttr(c.path)}">${escHtml(c.case_id)} — ${formatFraudType(c.fraud_type)}</option>`).join('')
    : `<option value="">All cases already submitted</option>`;
  updateCaseDesc();
  document.getElementById('submit-modal').classList.remove('hidden');
  document.getElementById('submit-btn').disabled = !sel.value;
  document.getElementById('submit-btn').textContent = 'Run Workflow';
}

function closeSubmitModal() {
  document.getElementById('submit-modal').classList.add('hidden');
}

async function submitCase() {
  const caseFile = document.getElementById('submit-case-file').value;
  const workflow = document.getElementById('submit-workflow').value.trim();
  const domain = document.getElementById('submit-domain').value.trim();
  if (!caseFile) { showToast('Please select a case file'); return; }

  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.textContent = 'Submitting…';

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_file: caseFile, workflow, domain }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Unknown error');
    }
    const data = await res.json();
    closeSubmitModal();
    await loadBacklog();
    if (data.instance_id) {
      await selectCase(data.instance_id);
      // Scroll new card into view and briefly flash it
      const card = document.querySelector(`[data-id="${data.instance_id}"]`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.style.transition = 'box-shadow 0.3s';
        card.style.boxShadow = '0 0 0 3px #0ea5e9';
        setTimeout(() => { card.style.boxShadow = ''; }, 1500);
      }
    }
    const inst = backlogData.find(c => c.instance_id === data.instance_id);
    const caseId = inst?.case_meta?.case_id || data.instance_id?.slice(-8).toUpperCase();
    showToast(`Case ${caseId} submitted — workflow running`);
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
    btn.disabled = false;
    btn.textContent = 'Run Workflow';
  }
}

// ── Helpers ──────────────────────────────────────────────────────────
function isTerminal(status) {
  return ['completed', 'failed', 'terminated'].includes(status);
}

function statusInfo(status) {
  const map = {
    running: { icon:'●', text:'text-sky-700',    bg:'bg-sky-50',    border:'border-sky-200' },
    created: { icon:'●', text:'text-sky-700',    bg:'bg-sky-50',    border:'border-sky-200' },
    suspended:{ icon:'⏸', text:'text-amber-700', bg:'bg-amber-50',  border:'border-amber-200' },
    completed:{ icon:'✓', text:'text-green-700', bg:'bg-green-50',  border:'border-green-200' },
    failed:   { icon:'✗', text:'text-red-700',   bg:'bg-red-50',    border:'border-red-200' },
    terminated:{ icon:'■',text:'text-slate-600', bg:'bg-slate-50',  border:'border-slate-200' },
  };
  return map[status] || { icon:'○', text:'text-slate-500', bg:'bg-slate-50', border:'border-slate-200' };
}

function primInfo(prim) {
  const map = {
    classify:   { text:'text-sky-700',    bg:'bg-sky-50' },
    retrieve:   { text:'text-purple-700', bg:'bg-purple-50' },
    investigate:{ text:'text-amber-700',  bg:'bg-amber-50' },
    think:      { text:'text-orange-700', bg:'bg-orange-50' },
    verify:     { text:'text-green-700',  bg:'bg-green-50' },
    challenge:  { text:'text-red-700',    bg:'bg-red-50' },
    generate:   { text:'text-teal-700',   bg:'bg-teal-50' },
    act:        { text:'text-pink-700',   bg:'bg-pink-50' },
  };
  return map[prim] || { text:'text-slate-600', bg:'bg-slate-50' };
}

function priorityClass(p) {
  const m = {
    critical: 'bg-red-50 text-red-700 border-red-200',
    high:     'bg-orange-50 text-orange-700 border-orange-200',
    medium:   'bg-yellow-50 text-yellow-700 border-yellow-200',
    low:      'bg-slate-50 text-slate-500 border-slate-200',
  };
  return m[p] || 'bg-slate-50 text-slate-500 border-slate-200';
}

function confPill(v) {
  if (v == null) return '';
  const pct = Math.round(v * 100);
  const c = pct >= 80 ? 'text-green-700 bg-green-50 border-green-200'
    : pct >= 60 ? 'text-yellow-700 bg-yellow-50 border-yellow-200'
    : 'text-red-700 bg-red-50 border-red-200';
  return `<span class="text-xs font-medium ${c} border px-1.5 py-0.5 rounded">${pct}%</span>`;
}

function confBar(v) {
  if (v == null) return '';
  const pct = Math.round(v * 100);
  const c = pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500';
  return `<div class="flex items-center gap-2 my-1">
    <div class="flex-1 bg-slate-100 rounded-full h-1.5">
      <div class="conf-bar-fill ${c} h-1.5 rounded-full" style="width:${pct}%"></div>
    </div>
    <span class="text-xs text-slate-500 w-8 text-right">${pct}%</span>
  </div>`;
}

function row(label, value) {
  if (value == null || value === '' || value === undefined) return '';
  return `<tr><td class="text-slate-500 font-medium w-40">${escHtml(String(label))}</td><td>${escHtml(String(value))}</td></tr>`;
}

function fmtAmount(n) {
  if (n == null) return '0';
  return Number(n).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function fmtSla(sla) {
  if (!sla) return '';
  try {
    const d = new Date(sla);
    const now = new Date();
    const diff = d - now;
    if (diff < 0) return 'SLA expired';
    const h = Math.round(diff / 3600000);
    return h < 1 ? 'SLA <1h' : `SLA ${h}h`;
  } catch { return ''; }
}

function formatTimeAgo(ts) {
  if (!ts) return '';
  const diff = Date.now() / 1000 - ts;
  if (diff < 5) return 'just now';
  if (diff < 60) return Math.round(diff) + 's ago';
  if (diff < 3600) return Math.round(diff / 60) + 'm ago';
  if (diff < 86400) return Math.round(diff / 3600) + 'h ago';
  return Math.round(diff / 86400) + 'd ago';
}

function formatFraudType(ft) {
  if (!ft) return '—';
  return ft.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function workflowLabel(wt, domain) {
  const label = domain && domain !== wt ? domain : wt;
  return label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function escAttr(s) {
  if (s == null) return '';
  return String(s).replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function showToast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = type === 'error' ? '#dc2626' : '#1e293b';
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3500);
}
