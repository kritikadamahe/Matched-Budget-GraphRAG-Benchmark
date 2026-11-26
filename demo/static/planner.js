// Budget planner tab. Uses helpers from app.js ($, el, esc, fmt, tip, untip, COLOR, NAMES).
"use strict";

const PL = { cfg: null, corpus: null, report: null, reportId: null, poll: null, costs: null };
const STAGES = [["questions", "Write test questions"], ["ranking", "Rank chunks"], ["extraction", "Extract entities"],
                ["answering", "Answer at every budget"], ["report", "Build report"]];
const money = v => v == null ? "-" : v >= 100 ? "$" + Math.round(v).toLocaleString() : v >= 0.1 ? "$" + v.toFixed(2) : "$" + v.toFixed(4);
const api = async (path, body) => {
  const r = await fetch(path, body ? { method: "POST", body: JSON.stringify(body) } : undefined);
  const data = await r.json().catch(() => ({ error: "HTTP " + r.status }));
  if (!r.ok || data.error) throw new Error(data.error || "HTTP " + r.status);
  return data;
};

async function initPlanner() {
  PL.cfg = await api("/api/planner/config");
  const c = PL.cfg;
  $("#pl-model").value = c.model; $("#pl-pin").value = c.price_in; $("#pl-pout").value = c.price_out;
  $("#pl-formats").textContent = "PDF, Markdown, text, HTML, CSV, JSON · up to 200 files / 40 MB";
  const strats = $("#pl-strats"), prim = $("#pl-primary");
  Object.entries(c.strategies).forEach(([k, name]) => {
    strats.append(el("label", { className: "check" }, `<input type="checkbox" value="${k}" ${["random", "ketrag"].includes(k) ? "checked" : ""}><i style="--c:${COLOR(k)}"></i>${name}`));
    prim.append(el("option", { value: k, textContent: name }));
  });
  prim.value = "ketrag";
  $("#pl-files").onchange = e => upload([...e.target.files]);
  const drop = $("#pl-drop");
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => upload([...e.dataTransfer.files]));
  $("#pl-sample").onclick = useSample;
  $("#pl-example").onclick = useSample;
  ["#pl-pin", "#pl-pout"].forEach(s => $(s).addEventListener("input", debounce(refreshCosts, 250)));
  ["#pl-nq", "#pl-ns", "#pl-cap", "#pl-primary", "#pl-model"].forEach(s => $(s).addEventListener("input", debounce(refreshEstimate, 250)));
  strats.addEventListener("change", debounce(refreshEstimate, 100));
  $("#pl-target").oninput = () => { $("#pl-target-v").textContent = $("#pl-target").value + "%"; if (PL.report) renderReport(); };
  $("#pl-run").onclick = runPilot;
  setRunState();
}

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function spec() {
  return {
    n_questions: +$("#pl-nq").value, sample_chunks: +$("#pl-ns").value, max_usd: +$("#pl-cap").value,
    primary: $("#pl-primary").value, model: $("#pl-model").value.trim(), target: +$("#pl-target").value / 100,
    price_in: +$("#pl-pin").value, price_out: +$("#pl-pout").value, budgets: PL.cfg.budgets,
    strategies: [...document.querySelectorAll("#pl-strats input:checked")].map(i => i.value),
  };
}

// ---------------------------------------------------------------- 1. documents
async function upload(files) {
  if (!files.length) return;
  const box = $("#pl-corpus");
  box.innerHTML = `<div class="tiny">Reading ${files.length} file(s)…</div>`;
  try {
    const payload = await Promise.all(files.map(async f => ({ name: f.name, data: b64(await f.arrayBuffer()) })));
    box.innerHTML = `<div class="tiny">Parsing and chunking…</div>`;
    const r = await api("/api/planner/corpus", { files: payload });
    PL.corpus = { id: r.corpus_id, sample: false, ...r };
    box.innerHTML = corpusSummary(r) + `<details class="files"><summary>${r.files.length} file(s)</summary>` +
      r.files.map(f => `<div class="${f.ok ? "" : "err"}">${f.ok ? "✓" : "✗"} ${esc(f.file)} <span class="tiny">${f.ok ? f.words.toLocaleString() + " words" : esc(f.reason)}</span></div>`).join("") + "</details>";
  } catch (e) { box.innerHTML = `<div class="err">${esc(e.message)}</div>`; PL.corpus = null; }
  refreshCosts(); setRunState();
}
function b64(buf) {
  const bytes = new Uint8Array(buf); let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function corpusSummary(r) {
  return `<div class="stats3"><div><b>${r.n_documents.toLocaleString()}</b><span>documents</span></div>
    <div><b>${r.n_chunks.toLocaleString()}</b><span>chunks</span></div><div><b>${r.n_words.toLocaleString()}</b><span>words</span></div></div>`;
}
async function useSample() {
  PL.corpus = { id: "sample", sample: true, n_documents: 748, n_chunks: 759, n_words: 68648 };
  $("#pl-corpus").innerHTML = corpusSummary(PL.corpus) + `<div class="tiny">HotpotQA benchmark corpus with its 75 real questions and the logged results of the paper's run.</div>`;
  refreshCosts(); setRunState();
  $("#pl-report").innerHTML = `<div class="empty">Loading the example report…</div>`;
  try { PL.report = await api("/api/planner/sample"); PL.reportId = "sample"; renderReport(); }
  catch (e) { $("#pl-report").innerHTML = `<div class="empty err">${esc(e.message)}</div>`; }
}

// ---------------------------------------------------------------- 2. costs
async function refreshCosts() {
  if (!PL.corpus) return;
  try {
    PL.costs = await api("/api/planner/costs", { corpus_id: PL.corpus.id, price_in: +$("#pl-pin").value, price_out: +$("#pl-pout").value });
  } catch (e) { $("#pl-costs").innerHTML = `<div class="err">${esc(e.message)}</div>`; return; }
  const rows = PL.costs.rows, full = rows[rows.length - 1], cal = PL.costs.calibration;
  const ratio = full.usd / Math.max(full.usd_uncalibrated, 1e-9);
  $("#pl-costs").innerHTML = `<table class="ctab"><tr><th>Budget</th><th>Chunks</th><th>Tokens</th><th>Indexing cost</th></tr>` +
    rows.map(r => `<tr><td>${r.budget_pct}%</td><td>${r.n_chunks.toLocaleString()}</td><td>${((r.input_tokens + r.output_tokens) / 1e6).toFixed(2)}M</td>
      <td><b>${money(r.usd)}</b><div class="cbar" style="width:${100 * r.usd / full.usd}%"></div></td></tr>`).join("") + `</table>
    <p class="tiny">Answering 1,000 questions: about <b>${money(PL.costs.per_1000_questions_usd)}</b>.
    Calibrated on ${esc(cal.source)}. The usual rule of thumb (1.33 tokens/word, 250 output tokens per chunk) would say
    ${money(full.usd_uncalibrated)}, <b>${ratio.toFixed(1)}× too low</b>.</p>`;
  refreshEstimate();
}

// ---------------------------------------------------------------- 3. pilot
function setRunState() {
  const btn = $("#pl-run"), note = $("#pl-run-note");
  const can = PL.cfg?.can_run_pilot, real = PL.corpus && !PL.corpus.sample;
  btn.disabled = !(can && real) || !!PL.poll;
  note.innerHTML = !PL.corpus ? "Add documents first."
    : PL.corpus.sample ? "The sample corpus already has a full report (below), so no pilot is needed."
    : !can ? "Pilots make paid LLM calls. Restart the server with <code>./demo/run.sh --allow-live</code> to enable them."
    : PL.cfg.mock ? "Mock mode: the pilot runs offline and its numbers are meaningless (wiring test)." : "";
}
async function refreshEstimate() {
  const box = $("#pl-estimate");
  if (!PL.corpus || PL.corpus.sample) { box.innerHTML = ""; return; }
  try {
    const e = await api("/api/planner/estimate", { corpus_id: PL.corpus.id, spec: spec() });
    box.innerHTML = `<div class="est ${e.within_cap ? "" : "over"}">
      <div><span>Extraction (${e.sample_chunks} chunks)</span><b>${money(e.extraction_usd)}</b></div>
      <div><span>Test questions</span><b>${money(e.questions_usd)}</b></div>
      <div><span>Answers (${e.n_answer_calls} calls)</span><b>${money(e.answers_usd)}</b></div>
      <div class="tot"><span>Pilot estimate (upper bound)</span><b>${money(e.total_usd)}</b></div>
      ${e.within_cap ? "" : `<div class="warn">Above the spending cap: the pilot will stop when it reaches the cap.</div>`}</div>`;
  } catch (err) { box.innerHTML = `<div class="err">${esc(err.message)}</div>`; }
}
async function runPilot() {
  try {
    const { job_id } = await api("/api/planner/pilot", { corpus_id: PL.corpus.id, spec: spec() });
    PL.poll = setInterval(() => pollJob(job_id), 800);
    setRunState(); pollJob(job_id);
  } catch (e) { $("#pl-progress").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
async function pollJob(id) {
  let j;
  try { j = await api("/api/planner/job/" + id); } catch (e) { return; }
  const idx = STAGES.findIndex(s => s[0] === j.stage);
  $("#pl-progress").innerHTML = `<ol class="stages">` + STAGES.map(([k, label], i) => {
    const st = j.status === "done" || i < idx ? "done" : i === idx ? "active" : "";
    const pct = st === "active" && j.total ? Math.round(100 * j.done / j.total) : st === "done" ? 100 : 0;
    return `<li class="${st}"><span>${st === "done" ? "✓" : i + 1}</span>${label}${st === "active" ? ` <em>${j.done}/${j.total}</em>` : ""}
      <div class="pbar"><div style="width:${pct}%"></div></div></li>`;
  }).join("") + `</ol><div class="tiny">${esc(j.message || "")}</div>` + (j.error ? `<div class="err">${esc(j.error)}</div>` : "");
  if (j.status === "done" || j.status === "error") {
    clearInterval(PL.poll); PL.poll = null; setRunState();
    if (j.report) { PL.report = j.report; PL.reportId = id; renderReport(); $("#pl-report").scrollIntoView({ behavior: "smooth" }); }
  }
}

// ---------------------------------------------------------------- 4. report
function recommendJS(r, target) {             // same rule as planner/pilot.py recommend()
  const curve = r.quality[r.primary], B = r.budgets, cost = Object.fromEntries(r.cost_projection.map(x => [x.budget_pct, x.usd]));
  const full = curve["100"].mean, fullUsd = cost[100];
  if (full < 0.15) return { status: "inconclusive", full, fullUsd, budget: 100, usd: fullUsd };
  const goal = target * full; let pb = 0, pf = 0, chosen = 100;
  for (const b of B) {
    const f = curve[b].mean;
    if (f >= goal) { const it = f > pf ? pb + (goal - pf) / (f - pf) * (b - pb) : b; chosen = Math.min(b, Math.max(5, Math.ceil(it / 5) * 5)); break; }
    pb = b; pf = f;
  }
  const xs = [0, ...B], ys = [0, ...B.map(b => curve[b].mean)];
  let f = ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) if (chosen <= xs[i]) { f = ys[i - 1] + (ys[i] - ys[i - 1]) * (chosen - xs[i - 1]) / (xs[i] - xs[i - 1]); break; }
  const usd = fullUsd * chosen / 100;
  return { status: "ok", full, fullUsd, budget: chosen, usd, f, retained: f / full, savings: fullUsd - usd,
           interpolated: !B.includes(chosen), lo: Math.max(0, ...B.filter(b => b < chosen)), hi: Math.min(...B.filter(b => b >= chosen)) };
}
function renderReport() {
  const r = PL.report, target = +$("#pl-target").value / 100, rec = recommendJS(r, target);
  const names = r.names, P = r.primary, kindLabel = { sample: "Example · logged benchmark", pilot: "Pilot · your documents", simulated: "SIMULATED · not a real result" }[r.kind];
  const v = r.vector;
  const verdicts = [];
  if (rec.status === "ok") {
    verdicts.push(`Extract <b>${rec.budget}%</b> of the corpus to keep about <b>${Math.round(100 * rec.retained)}%</b> of full-extraction answer quality, for <b>${money(rec.usd)}</b> instead of ${money(rec.fullUsd)}.`);
    if (rec.interpolated) verdicts.push(`${rec.budget}% falls between the measured budgets ${rec.lo}% and ${rec.hi}%: no measured budget below ${rec.hi}% reached the target. Confirm with a pilot at ${rec.budget}% before committing.`);
    if (v) verdicts.push(v.mean >= rec.full - 0.02 ? `Plain vector RAG (no LLM extraction, F1 ${fmt(v.mean)}) matches full GraphRAG (F1 ${fmt(rec.full)}): GraphRAG indexing is hard to justify for this corpus.`
      : v.mean >= rec.f ? `At ${rec.budget}%, GraphRAG (F1 ${fmt(rec.f)}) does not beat plain vector RAG (F1 ${fmt(v.mean)}). GraphRAG pays off only near full extraction.`
      : `GraphRAG at ${rec.budget}% beats plain vector RAG (F1 ${fmt(rec.f)} vs ${fmt(v.mean)}).`);
    const others = r.strategies.filter(s => s !== P);
    if (others.length) {
      const spread = Math.max(...r.budgets.filter(b => b < 100).map(b => Math.max(...r.strategies.map(s => r.quality[s][b].mean)) - Math.min(...r.strategies.map(s => r.quality[s][b].mean))));
      verdicts.push(spread <= (r.mde_f1 || 0.1) ? `The selectors differ by at most ${fmt(spread)} F1 at any budget, below what ${r.n_questions} questions can detect (${fmt(r.mde_f1)}). Choose on cost and simplicity: Random needs no embeddings.`
        : `The selectors differ by up to ${fmt(spread)} F1 at some budget. Check the table before choosing one.`);
    }
  } else verdicts.push("Even full extraction answers few test questions (F1 < 0.15). The questions may not suit this corpus. Do not size a budget from this run.");

  const sp = r.pilot_spend || {};
  const hdr = `<div class="rep-h">
      <div><span class="badge ${r.kind}">${kindLabel}</span><h2>${esc(r.title)}</h2>
        <div class="tiny">${r.corpus.n_documents.toLocaleString()} documents · ${r.corpus.n_chunks.toLocaleString()} chunks · ${r.n_questions} test questions ·
        model <code>${esc(r.model)}</code> at $${r.price_in}/$${r.price_out} per 1M tokens · ${r.created_utc.replace("T", " ").slice(0, 16)} UTC</div></div>
      <div class="rep-actions">
        <a class="btn" href="/api/planner/report/${PL.reportId}.md?target=${target}" download="graphrag-budget-plan.md">⇩ Markdown</a>
        <a class="btn" href="/api/planner/report/${PL.reportId}.json" download="graphrag-budget-plan.json">⇩ JSON</a>
        <button class="btn" onclick="window.print()">Print / PDF</button></div></div>`;
  const summary = `<div class="rep-sum ${rec.status}">${verdicts[0]}</div>`;
  const kpis = rec.status === "ok" ? [
    ["Recommended budget", `${rec.budget}%`, rec.interpolated ? "interpolated" : "measured", "accent"],
    ["Projected indexing cost", money(rec.usd), `of ${money(rec.fullUsd)} for everything`],
    ["Savings", money(rec.savings), `${100 - rec.budget}% fewer LLM extraction calls`],
    ["Quality retained", `${Math.round(100 * rec.retained)}%`, `F1 ${fmt(rec.f)} vs ${fmt(rec.full)} · target ${Math.round(target * 100)}%`],
    ["Vector-RAG baseline", v ? `F1 ${fmt(v.mean)}` : "not run", v ? "no LLM indexing at all" : "only in pilots on your documents"],
  ] : [["Status", "Inconclusive", "see below"]];
  const kpiHtml = `<div class="rep-kpis">${kpis.map(([l, val, sub, cls]) => `<div class="kpi ${cls || ""}"><div class="l">${l}</div><div class="v">${val}</div><div class="l">${sub}</div></div>`).join("")}</div>`;

  const cost = Object.fromEntries(r.cost_projection.map(x => [x.budget_pct, x]));
  const table = `<table class="rtab"><tr><th>Budget</th><th>Chunks</th><th>Indexing cost</th>${r.strategies.map(s => `<th><span class="rowlab" style="--c:${COLOR(s)}"><i></i>${names[s]}</span></th>`).join("")}
      <th>Evidence extracted</th><th>Evidence retrieved</th></tr>` +
    r.budgets.map(b => `<tr class="${rec.status === "ok" && b === rec.hi ? "hl" : ""}"><td><b>${b}%</b></td><td>${cost[b].n_chunks.toLocaleString()}</td><td>${money(cost[b].usd)}</td>` +
      r.strategies.map(s => { const q = r.quality[s][b]; return `<td>${fmt(q.mean)} <span class="f">[${fmt(q.low)}, ${fmt(q.high)}]</span></td>`; }).join("") +
      `<td>${pct(r.evidence[P][b].selected)}</td><td>${pct(r.evidence[P][b].retrieved)}</td></tr>`).join("") +
    (v ? `<tr class="vec"><td><b>Vector RAG</b></td><td>0</td><td>$0 (LLM)</td><td colspan="${r.strategies.length}">${fmt(v.mean)} <span class="f">[${fmt(v.low)}, ${fmt(v.high)}]</span> · no extraction, passages by embedding similarity</td><td>-</td><td>-</td></tr>` : "") + "</table>";

  const ex = (r.examples || []).slice(0, 8);
  const exHtml = ex.length ? `<table class="rtab ex"><tr><th>Test question</th><th>Expected</th>${r.budgets.map(b => `<th>${b}%</th>`).join("")}${v ? "<th>Vector</th>" : ""}</tr>` +
    ex.map(e => `<tr><td>${esc(e.question)}<div class="f">${e.hops === 2 ? "two-hop · " : ""}${esc((e.gold_titles || []).join(" + "))}</div></td><td><b>${esc(e.answer)}</b></td>` +
      r.budgets.map(b => { const a = e.by_budget[b]; return `<td class="${a.f1 >= .99 ? "ok" : a.f1 > 0 ? "mid" : "no"}">${esc(a.answer)}</td>`; }).join("") +
      (e.vector ? `<td class="${e.vector.f1 >= .99 ? "ok" : e.vector.f1 > 0 ? "mid" : "no"}">${esc(e.vector.answer)}</td>` : "") + "</tr>").join("") + "</table>" : "";

  const cal = r.calibration;
  const method = `<div class="rep-cols"><div><h3>Method</h3><p>Matched-budget protocol: every budget uses the same chunks, LLM extractor, graph builder,
      personalised-PageRank retriever and answer generator; only the share of chunks sent to the extractor changes. Test questions
      ${r.kind === "sample" ? "are HotpotQA's own, with their gold paragraphs" : "were written by the LLM from one chunk or from two linked chunks (two-hop), so their source chunks are known"}.
      The recommendation is the smallest budget whose mean F1 reaches the quality target (interpolated, rounded up to 5%), using the <b>${names[P]}</b> selector.</p>
      <p>Cost model: input ≈ ${cal.in_base.toFixed(0)} + ${cal.in_per_word.toFixed(2)}×words, output ≈ ${cal.out_base.toFixed(0)} + ${cal.out_per_word.toFixed(2)}×words tokens per chunk, fitted on ${esc(cal.source)}.</p>
      ${sp.usd != null ? `<p>Pilot spend: <b>${money(sp.usd)}</b> (estimate ${money(sp.estimate_usd)}, cap ${money(sp.cap_usd)}), ${sp.seconds} s.</p>` : ""}</div>
      <div><h3>Caveats</h3><ul>${r.caveats.map(c => `<li>${esc(c)}</li>`).join("")}</ul></div></div>`;

  $("#pl-report").innerHTML = `<div class="panel rep">${hdr}${summary}${kpiHtml}
    <div class="rep-charts"><div><h3>Answer quality vs extraction budget</h3><div id="pl-chart-q" class="chart"></div></div>
      <div><h3>Cost vs quality</h3><div id="pl-chart-c" class="chart"></div></div></div>
    <h3>Decision notes</h3><ul class="verdicts">${verdicts.map(x => `<li>${x}</li>`).join("")}</ul>
    <h3>Budget table <span class="tiny">full-corpus cost projection · F1 with 95% bootstrap intervals · evidence columns for ${names[P]}</span></h3>${table}
    ${exHtml ? `<h3>Sample of test questions <span class="tiny">answers with the ${names[P]} selector</span></h3>${exHtml}` : ""}
    ${method}</div>`;
  qualityChart($("#pl-chart-q"), r, rec, target);
  frontierChart($("#pl-chart-c"), r, rec);
}

function svgFrame(W, H, L, B, T, R, xTicks, yTicks, xFmt, yFmt, xLab, yLab, x, y) {
  let s = `<g class="grid">`;
  yTicks.forEach(v => s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`);
  s += `</g>`;
  xTicks.forEach(v => s += `<text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${xFmt(v)}</text>`);
  s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${xLab}</text>`;
  s += `<text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${yLab}</text>`;
  return s;
}
function qualityChart(root, r, rec, target) {
  const W = 560, H = 300, L = 46, R = 16, T = 14, B = 40;
  const yMax = Math.max(0.2, Math.ceil((Math.max(...r.strategies.map(s => r.quality[s]["100"].high), r.vector?.high || 0) + 0.05) * 10) / 10);
  const x = b => L + (W - L - R) * b / 100, y = v => T + (H - T - B) * (1 - v / yMax);
  const P = r.primary, B_ = r.budgets;
  let s = `<svg viewBox="0 0 ${W} ${H}">` + svgFrame(W, H, L, B, T, R, [0, 25, 50, 75, 100], [0, yMax / 4, yMax / 2, 3 * yMax / 4, yMax],
    b => b + "%", v => v.toFixed(2), "Share of chunks extracted", "Token F1", x, y);
  const q = r.quality[P];
  s += `<polygon fill="${COLOR(P)}" opacity=".12" points="${[...B_.map(b => `${x(b)},${y(q[b].high)}`), ...B_.slice().reverse().map(b => `${x(b)},${y(q[b].low)}`)].join(" ")}"/>`;
  if (rec.status === "ok") {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(target * rec.full)}" y2="${y(target * rec.full)}" stroke="var(--text-2)" stroke-dasharray="5 4"/>`;
    s += `<text x="${L + 6}" y="${y(target * rec.full) - 5}">target: ${Math.round(target * 100)}% of full-extraction F1</text>`;
    s += `<line x1="${x(rec.budget)}" x2="${x(rec.budget)}" y1="${T}" y2="${H - B}" stroke="var(--accent)" stroke-width="2"/>`;
    s += `<text x="${x(rec.budget) - 5}" y="${T + 12}" text-anchor="end" style="fill:var(--accent);font-weight:700">recommended ${rec.budget}%</text>`;
  }
  if (r.vector) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(r.vector.mean)}" y2="${y(r.vector.mean)}" stroke="var(--s-oracle)" stroke-dasharray="1 4" stroke-width="2"/>`;
    s += `<text x="${L + 4}" y="${y(r.vector.mean) - 5}">vector RAG</text>`;
  }
  for (const st of r.strategies) {
    const pts = [[0, 0], ...B_.map(b => [b, r.quality[st][b].mean])];
    s += `<polyline fill="none" stroke="${COLOR(st)}" stroke-width="${st === P ? 2.5 : 1.5}" ${st === P ? "" : 'opacity=".8"'} points="${pts.map(([b, v]) => `${x(b)},${y(v)}`).join(" ")}"/>`;
    B_.forEach(b => s += `<circle cx="${x(b)}" cy="${y(r.quality[st][b].mean)}" r="${st === P ? 4.5 : 3.5}" fill="${COLOR(st)}" stroke="var(--surface)" stroke-width="2"/>`);
  }
  B_.forEach(b => s += `<rect class="hit" data-b="${b}" x="${x(b) - 18}" y="${T}" width="36" height="${H - T - B}" fill="transparent"/>`);
  s += `</svg><div class="clegend">${r.strategies.map(st => `<span><i style="--c:${COLOR(st)}"></i>${r.names[st]}${st === P ? " (band: 95% CI)" : ""}</span>`).join("")}</div>`;
  root.innerHTML = s;
  root.querySelectorAll(".hit").forEach(h => {
    const b = h.dataset.b;
    h.onmousemove = e => tip(e, `<b>${b}% budget</b> · ${money(r.cost_projection.find(c => c.budget_pct == b).usd)}<br>` +
      r.strategies.map(st => `${r.names[st]}: <b>${fmt(r.quality[st][b].mean)}</b>`).join("<br>"));
    h.onmouseleave = untip;
  });
}
function frontierChart(root, r, rec) {
  const W = 560, H = 300, L = 46, R = 16, T = 14, B = 40, P = r.primary;
  const pts = r.budgets.map(b => ({ b, c: r.cost_projection.find(c => c.budget_pct === b).usd, q: r.quality[P][b].mean }));
  const xMax = Math.max(...pts.map(p => p.c)) * 1.08, yMax = Math.max(0.2, Math.ceil((Math.max(...pts.map(p => p.q), r.vector?.mean || 0) + 0.05) * 10) / 10);
  const x = c => L + (W - L - R) * c / xMax, y = v => T + (H - T - B) * (1 - v / yMax);
  let s = `<svg viewBox="0 0 ${W} ${H}">` + svgFrame(W, H, L, B, T, R, [0, xMax / 4, xMax / 2, 3 * xMax / 4], [0, yMax / 2, yMax],
    c => c === 0 ? "$0" : money(c), v => v.toFixed(2), "Projected indexing cost (full corpus)", "Token F1", x, y);
  s += `<polyline fill="none" stroke="${COLOR(P)}" stroke-width="2" points="${[[0, 0], ...pts.map(p => [p.c, p.q])].map(([c, q]) => `${x(c)},${y(q)}`).join(" ")}"/>`;
  pts.forEach(p => {
    s += `<circle cx="${x(p.c)}" cy="${y(p.q)}" r="5" fill="${COLOR(P)}" stroke="var(--surface)" stroke-width="2"/>`;
    s += `<text x="${x(p.c) + 8}" y="${y(p.q) + 14}">${p.b}%</text>`;
  });
  if (rec.status === "ok") s += `<circle cx="${x(rec.usd)}" cy="${y(rec.f)}" r="8" fill="none" stroke="var(--accent)" stroke-width="2.5"/><text x="${x(rec.usd) - 10}" y="${y(rec.f) - 10}" text-anchor="end" style="fill:var(--accent);font-weight:700">${rec.budget}% · ${money(rec.usd)}</text>`;
  if (r.vector) s += `<rect x="${x(0) - 5}" y="${y(r.vector.mean) - 5}" width="10" height="10" transform="rotate(45 ${x(0)} ${y(r.vector.mean)})" fill="var(--s-oracle)"/><text x="${x(0) + 10}" y="${y(r.vector.mean) + 4}">vector RAG ($0 LLM)</text>`;
  s += `</svg><div class="clegend"><span><i style="--c:${COLOR(P)}"></i>${r.names[P]}: each point is one budget</span></div>`;
  root.innerHTML = s;
}
