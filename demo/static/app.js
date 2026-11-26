// Presentation demo front end. Data comes from demo/server.py (/api/meta, /api/ask).
"use strict";

const NAMES = { random: "Random", ketrag: "KET-RAG-style", lazygraphrag: "Topic coverage", fastgraphrag: "Entity centrality" };
const COLOR = s => `var(--s-${s})`;
// Oracle selection recall from Table III of the paper (main_6page.tex); not stored in the results folder.
const ORACLE = { hotpotqa: { 5: .253, 10: .506, 25: 1, 50: 1, 75: 1 }, musique: { 5: .406, 10: .816, 25: 1, 50: 1, 75: 1 } };
const TYPE_COLOR = { PERSON: "#2a78d6", ORGANIZATION: "#eb6834", LOCATION: "#1baf7a", EVENT: "#e87ba4", WORK: "#6250d6", OTHER: "#8a8983" };

let M;                         // /api/meta
const state = { budget: 10, budgetQ: "", askQ: null, askS: "ketrag", askB: 25, cmpQ: null, ssDs: "hotpotqa" };
const $ = s => document.querySelector(s);
const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const kOf = pct => Math.max(1, Math.floor(pct / 100 * M.n_chunks + 0.5));
const rankKey = s => s === "random" ? "random_rs0" : s;

// ------------------------------------------------------------------ boot
// Tabs work even if data loading fails, and one broken section never disables the others.
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
const SERVER_HELP = `This page needs the demo server, which supplies the benchmark data.<br>In a terminal run
  <code>./demo/run.sh</code> inside <code>Matched-Budget-GraphRAG-Benchmark</code>, then open
  <a href="http://127.0.0.1:8765">http://127.0.0.1:8765</a>. Opening index.html directly (file:// or a VS Code preview) will not work.`;
if (location.protocol === "file:") $("#loading").innerHTML = `<div class="err-box">${SERVER_HELP}</div>`;
else fetch("/api/meta").then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }).then(meta => {
  M = meta;
  M.pos = {};
  for (const [key, ranked] of Object.entries(M.rankings)) {
    M.pos[key] = {}; ranked.forEach((id, i) => M.pos[key][id] = i);
  }
  M.qById = Object.fromEntries(M.questions.map(q => [q.id, q]));
  state.askQ = state.cmpQ = M.questions[0].id;
  for (const init of [initIdea, initBudget, initAsk, initCompare, initResults, initTabs]) {
    try { init(); } catch (e) { console.error(e); banner(`${init.name} failed: ${e.message}`); }
  }
  if (typeof initPlanner === "function") initPlanner().catch(e => banner("Budget planner failed: " + e.message));
  $("#loading").remove();
}).catch(e => { $("#loading").innerHTML = `<div class="err-box"><b>Could not load data (${esc(e.message)}).</b><br>${SERVER_HELP}</div>`; });

function banner(msg) {
  const b = el("div", { className: "err-box banner" }, esc(msg)); document.body.prepend(b);
}

function initTabs() {
  // Deep link for recording: #ask/<question_id>/<strategy>/<budget> opens and runs that condition.
  const [tab, qid, s, b] = location.hash.slice(1).split("/");
  if (tab === "ask" && M.qById[qid]) { replay(qid, s, +b); return; }
  if (tab) showTab(tab);
}
function replay(qid, s, b) {
  state.askQ = qid; state.askS = M.strategies.includes(s) ? s : state.askS; state.askB = M.budgets.includes(b) ? b : state.askB;
  $("#ask-q").value = qid; $("#ask-custom").value = "";
  askControls(); showTab("ask"); runAsk();
}
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + name));
  history.replaceState(null, "", "#" + name);
  if (name === "idea") animatePipeline();
}

// ------------------------------------------------------------------ 1. idea
function initIdea() {
  const steps = [
    ["Corpus", "HotpotQA + MuSiQue, pooled"], ["Chunking", "250 words, 40 overlap"],
    ["Selector π", "the only variable", true], ["Budget cut", "keep top k = B·N"],
    ["LLM extraction", "entities + relations"], ["Knowledge graph", "merged by name"],
    ["PageRank retrieval", "5 seeds, top 20"], ["Answer", M.model], ["Evaluation", "EM · F1 · evidence"],
  ];
  const p = $("#pipeline");
  steps.forEach(([t, s, v], i) => {
    if (i) p.append(el("div", { className: "parrow" }, "→"));
    const d = el("div", { className: "pstep" + (v ? " var" : "") }, `<b>${t}</b><small>${esc(s)}</small>`);
    d.style.setProperty("--i", i); p.append(d);
  });
  const e2e = metric("e2e", "f1");
  const kpis = [
    [M.n_chunks, `chunks in the end-to-end corpus (${M.questions.length} questions)`],
    ["24", "end-to-end conditions: 4 budgets × (3 Random shuffles + 3 selectors)"],
    ["1,000", "questions in the LLM-free selection study (500 HotpotQA + 500 MuSiQue)"],
    [`${fmt(e2e.random[10].mean)} → ${fmt(e2e.random[100].mean)}`, "token F1, 10% → 100% budget"],
  ];
  $("#kpis").innerHTML = kpis.map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}
function animatePipeline() {                 // restart the CSS stagger
  document.querySelectorAll(".pstep").forEach(s => { s.style.animation = "none"; void s.offsetWidth; s.style.animation = ""; });
}

// ------------------------------------------------------------------ 2. budget explorer
function initBudget() {
  $("#n-chunks").textContent = M.n_chunks;
  const sel = $("#budget-q");
  M.questions.forEach(q => sel.append(el("option", { value: q.id, textContent: q.question })));
  sel.onchange = () => { state.budgetQ = sel.value; drawBudget(); };
  const range = $("#budget-range");
  range.oninput = () => { state.budget = +range.value; drawBudget(); };
  $("#budget-play").onclick = sweep;

  const grids = $("#grids");
  M.strategies.forEach(s => {
    const card = el("div", { className: "gcard" });
    card.style.setProperty("--c", COLOR(s));
    card.innerHTML = `<div class="gt"><i></i>${NAMES[s]}</div><div class="cells"></div><div class="gstats"></div>`;
    const cells = card.querySelector(".cells");
    M.chunks.forEach(c => {
      const d = el("div", { className: "cell" });
      d.onmouseenter = e => tip(e, `<b>${esc(c.title)}</b><br>rank ${M.pos[rankKey(s)][c.id] + 1} of ${M.n_chunks}` +
        (c.gold.length ? `<br>gold for ${c.gold.length} question(s)` : ""));
      d.onmouseleave = untip;
      cells.append(d);
    });
    card.dataset.s = s;
    grids.append(card);
  });
  drawBudget();
}
function drawBudget() {
  const k = kOf(state.budget);
  $("#budget-val").textContent = `${state.budget}%  (${k} chunks)`;
  const qs = state.budgetQ ? [M.qById[state.budgetQ]] : M.questions;
  const goldSet = new Set(qs.flatMap(q => q.gold_chunks));
  document.querySelectorAll(".gcard").forEach(card => {
    const pos = M.pos[rankKey(card.dataset.s)];
    const cells = card.querySelectorAll(".cell");
    M.chunks.forEach((c, i) => {
      const cl = cells[i].classList;
      cl.toggle("sel", pos[c.id] < k);
      cl.toggle("gold", goldSet.has(c.id));
      cl.toggle("dim", !!state.budgetQ && !goldSet.has(c.id) && !(pos[c.id] < k));
    });
    let rec = 0, all = 0;
    for (const q of qs) {
      const hit = q.gold_chunks.filter(id => pos[id] < k).length;
      rec += hit / q.gold_chunks.length; all += hit === q.gold_chunks.length ? 1 : 0;
    }
    rec /= qs.length;
    card.querySelector(".gstats").innerHTML =
      `<span>Gold evidence selected</span><span class="num">${pct(rec)}</span><div class="bar"><div style="width:${rec * 100}%"></div></div>` +
      `<span>Questions with <i>all</i> gold selected</span><span class="num">${all} / ${qs.length}</span>`;
  });
}
async function sweep() {
  const range = $("#budget-range"), btn = $("#budget-play");
  btn.disabled = true;
  for (let b = 1; b <= 100; b++) { range.value = state.budget = b; drawBudget(); await sleep(b < 30 ? 90 : 35); }
  btn.disabled = false;
}

// ------------------------------------------------------------------ 3. ask
function initAsk() {
  const sel = $("#ask-q");
  M.questions.forEach(q => sel.append(el("option", { value: q.id, textContent: (q.featured ? "★ " : "") + q.question })));
  sel.value = state.askQ;
  sel.onchange = () => { state.askQ = sel.value; $("#ask-custom").value = ""; runAsk(); };
  $("#ask-custom").oninput = () => { if ($("#ask-custom").value.trim()) sel.selectedIndex = -1; };
  $("#live-note").textContent = M.allow_live
    ? "Live LLM calls are ON: a typed question costs one answer call."
    : "Live LLM calls are off, so a typed question shows retrieval only. Start the server with --allow-live to answer it.";
  askControls();
  $("#ask-run").onclick = runAsk;
}
function askControls() {           // changing the selector or budget re-runs straight away
  segmented("#ask-strategy", M.strategies, s => NAMES[s], () => state.askS, v => { state.askS = v; runAsk(); }, true);
  segmented("#ask-budget", M.budgets, b => b + "%", () => state.askB, v => { state.askB = v; runAsk(); });
}
function segmented(root, values, label, get, set, dots = false) {
  const r = $(root); r.innerHTML = "";
  values.forEach(v => {
    const b = el("button", {}, (dots ? `<i style="--c:${COLOR(v)}"></i>` : "") + label(v));
    b.classList.toggle("active", get() === v);
    b.onclick = () => { set(v); r.querySelectorAll("button").forEach(x => x.classList.remove("active")); b.classList.add("active"); };
    r.append(b);
  });
}
async function runAsk() {
  const custom = $("#ask-custom").value.trim();
  const body = { strategy: state.askS, budget: state.askB };
  if (custom) body.question = custom; else body.question_id = state.askQ;
  const out = $("#ask-out"), btn = $("#ask-run");
  btn.disabled = true; btn.textContent = "Running…";
  out.innerHTML = `<div class="empty">Retrieving from the ${NAMES[state.askS]} graph at ${state.askB}%…</div>`;
  let r;
  try { r = await (await fetch("/api/ask", { method: "POST", body: JSON.stringify(body) })).json(); }
  catch (e) { r = { error: String(e) }; }
  btn.disabled = false; btn.textContent = "Run pipeline →";
  if (r.error) { out.innerHTML = `<div class="empty err">${esc(r.error)}</div>`; return; }
  await renderAsk(r);
}
function step(n, title, meta, body) {
  const s = el("div", { className: "step" });
  s.innerHTML = `<div class="step-h"><div class="step-n">${n}</div><h3>${title}</h3><div class="meta">${meta || ""}</div></div>`;
  s.append(typeof body === "string" ? el("div", {}, body) : body);
  return s;
}
async function renderAsk(r) {
  const out = $("#ask-out"); out.innerHTML = "";
  const head = el("div", { className: "step on" },
    `<div class="tiny">${NAMES[state.askS]} · ${state.askB}% budget · condition <code>${r.condition_id}</code></div><div class="qline">${esc(r.question)}</div>`);
  out.append(head);

  const goldChips = r.gold.length
    ? `<p class="tiny">Gold supporting paragraphs for this question (the selector never sees these):</p><div class="chips">` +
      r.gold.map(g => `<span class="chip ${g.selected ? "ok" : "no"}">${g.selected ? "✓" : "✗"} ${esc(g.title)}</span>`).join("") + "</div>"
    : `<p class="tiny">Custom question: no gold labels.</p>`;
  const steps = [
    step(1, "Selection", `${r.n_selected} of ${M.n_chunks} chunks sent to the LLM extractor`,
      `<p>The <b>${NAMES[state.askS]}</b> selector ranked every chunk without seeing any question; the top ${state.askB}% were extracted.</p>${goldChips}`),
    step(2, "Knowledge graph", `${r.graph.n_nodes.toLocaleString()} entities · ${r.graph.n_edges.toLocaleString()} relations`,
      `<p>Built only from the extracted chunks. Non-selected chunks are indexed nowhere.</p>`),
  ];
  const gbox = el("div");
  gbox.innerHTML = `<div class="chips" style="margin-bottom:8px">${r.entities.filter(e => e.seed).map(e => `<span class="chip seed">● ${esc(e.name)}</span>`).join("")}</div>`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "graphbox"); gbox.append(svg);
  gbox.append(el("div", { className: "clegend" }, Object.entries(TYPE_COLOR).map(([t, c]) => `<span><i style="--c:${c};height:10px;width:10px;border-radius:50%"></i>${t.toLowerCase()}</span>`).join("")));
  steps.push(step(3, "Personalised PageRank", "5 seed entities nearest the question → top 20 entities", gbox));

  const ctx = el("div");
  ctx.innerHTML = (r.facts.length ? `<b>Facts</b><ul class="facts">${r.facts.map(f => `<li>${esc(f)}</li>`).join("")}</ul>` : "<p class='tiny'>No facts.</p>") +
    `<div style="margin-top:10px"><b>Passages</b></div>` +
    (r.passages.length ? r.passages.map((p, i) => `<div class="passage ${p.gold ? "gold" : ""}"><div class="ph">[${i + 1}] ${esc(p.title)} ${p.gold ? '<span class="tag">GOLD</span>' : ""}</div>${esc(p.text)}</div>`).join("")
      : "<p class='tiny'>No passages fit the context.</p>");
  steps.push(step(4, "Context sent to the LLM", `${r.context_words} words (limit 700)`, ctx));

  let ans;
  if (!r.answer) {
    ans = `<p>Retrieval done. Answering this new question needs a live LLM call, which is off. Restart with <code>--allow-live</code>.</p>`;
  } else {
    const f1 = r.f1;
    const cls = f1 == null ? "" : f1 >= .99 ? "ok" : f1 > 0 ? "mid" : "no";
    ans = `<div class="answer-row">
      <div class="abox"><div class="l">Model answer</div><div class="v">${esc(r.answer.text)}</div></div>
      <div class="abox"><div class="l">Gold answer</div><div class="v">${esc(r.gold_answer ?? "-")}</div></div>
      ${f1 == null ? "" : `<div class="score ${cls}"><div class="l">F1</div><div class="v">${f1.toFixed(2)}</div><div class="l">EM ${r.em}</div></div>`}
    </div>
    ${r.answer.reasoning ? `<div class="reason">“${esc(r.answer.reasoning)}”</div>` : ""}
    ${r.logged ? `<div class="verify">${r.logged.answer === r.answer.text ? "✓ Identical to the answer logged in the benchmark run" : "⚠ Differs from the logged answer: " + esc(r.logged.answer)}${r.answer.cache_hit ? " (served from the answer cache)" : ""}.</div>` : ""}
    ${r.live_used ? `<div class="tiny">Live call: ${r.answer.tokens_in} input / ${r.answer.tokens_out} output tokens.</div>` : ""}`;
  }
  steps.push(step(5, "Answer", M.model, ans));

  for (const s of steps) { out.append(s); await sleep(60); s.classList.add("on"); if (s.contains(svg)) drawGraph(svg, r); await sleep(380); }
}
function drawGraph(svg, r) {
  const W = svg.clientWidth || 800, H = 360;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const nodes = r.entities.map((e, i) => ({ ...e, x: W / 2 + Math.cos(i * 2.4) * (40 + i * 8), y: H / 2 + Math.sin(i * 2.4) * (30 + i * 6), vx: 0, vy: 0 }));
  const idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
  const seen = new Set(), links = [];
  for (const e of r.edges) {
    const key = [e.source, e.target].sort().join("|");
    if (!seen.has(key)) { seen.add(key); links.push({ a: idx[e.source], b: idx[e.target], label: e.label }); }
  }
  for (let it = 0; it < 400; it++) {               // small force layout: repulsion, springs, centring
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = Math.max(dx * dx + dy * dy, 25);
      const f = 2600 / d2, d = Math.sqrt(d2); dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const l of links) {
      const a = nodes[l.a], b = nodes[l.b]; const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
      const f = (d - 90) * 0.02; a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
    }
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.004; n.vy += (H / 2 - n.y) * 0.006;
      n.x = Math.min(W - 60, Math.max(60, n.x + n.vx * 0.5)); n.y = Math.min(H - 20, Math.max(20, n.y + n.vy * 0.5));
      n.vx *= 0.6; n.vy *= 0.6;
    }
  }
  const ns = "http://www.w3.org/2000/svg";
  const mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  svg.innerHTML = "";
  links.forEach((l, i) => {
    const a = nodes[l.a], b = nodes[l.b];
    const line = mk("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, "stroke-width": 1.5, opacity: 0 });
    line.style.transition = `opacity .4s ${0.3 + i * 0.01}s`;
    line.onmouseenter = e => tip(e, `${esc(a.name)} — ${esc(l.label)} — ${esc(b.name)}`); line.onmouseleave = untip;
    line.style.pointerEvents = "stroke"; line.setAttribute("stroke-width", 3);
    svg.append(line); requestAnimationFrame(() => line.setAttribute("opacity", 1));
  });
  nodes.forEach((n, i) => {
    const g = mk("g", { opacity: 0 }); g.style.transition = `opacity .35s ${i * 0.035}s`;
    const rad = n.seed ? 9 : 6.5;
    const c = mk("circle", { cx: n.x, cy: n.y, r: rad, fill: TYPE_COLOR[n.type] || TYPE_COLOR.OTHER,
      stroke: n.seed ? "var(--text)" : "var(--surface)", "stroke-width": n.seed ? 2.5 : 2 });
    c.onmouseenter = e => tip(e, `<b>${esc(n.name)}</b><br>${n.type.toLowerCase()} · PageRank #${n.rank + 1}${n.seed ? " · seed" : ""}`); c.onmouseleave = untip;
    const t = mk("text", { x: n.x + rad + 3, y: n.y + 4 }); t.textContent = n.name.length > 26 ? n.name.slice(0, 25) + "…" : n.name;
    if (n.seed) t.setAttribute("font-weight", "700");
    g.append(c, t); svg.append(g); requestAnimationFrame(() => g.setAttribute("opacity", 1));
  });
}

// ------------------------------------------------------------------ 4. compare
function initCompare() {
  const sel = $("#cmp-q");
  M.questions.forEach(q => sel.append(el("option", { value: q.id, textContent: (q.featured ? "★ " : "") + q.question })));
  sel.onchange = () => { state.cmpQ = sel.value; drawCompare(); };
  drawCompare();
  const e2e = metric("e2e", "f1");
  let h = `<table class="cmp"><tr><th></th>${M.budgets.map(b => `<th>${b}%</th>`).join("")}</tr>`;
  for (const s of M.strategies)
    h += `<tr><th><span class="rowlab" style="--c:${COLOR(s)}"><i></i>${NAMES[s]}</span></th>` +
      M.budgets.map(b => { const v = e2e[s][b]; return `<td style="cursor:default"><b>${fmt(v.mean)}</b> <span class="f">[${fmt(v.lo)}, ${fmt(v.hi)}]</span></td>`; }).join("") + "</tr>";
  $("#cmp-overall").innerHTML = h + "</table>";
}
function drawCompare() {
  const q = M.qById[state.cmpQ], sc = M.scores[q.id] || {};
  $("#cmp-head").innerHTML = `<div class="q">${esc(q.question)}</div><div class="tiny">Gold answer: <b>${esc(q.answer)}</b> · ${q.gold_chunks.length} gold paragraphs</div>`;
  const cid = (s, b, sh = 0) => `hotpotqa__${s}__b${String(b).padStart(3, "0")}__seed42` + (s === "random" ? `__rs${sh}` : "");
  const cls = f => f >= .99 ? "ok" : f > 0 ? "mid" : "no";
  let h = `<table class="cmp"><tr><th></th>${M.budgets.map(b => `<th>${b}% budget</th>`).join("")}</tr>`;
  for (const s of M.strategies) {
    h += `<tr><th><span class="rowlab" style="--c:${COLOR(s)}"><i></i>${NAMES[s]}</span></th>`;
    for (const b of M.budgets) {
      if (s === "random") {
        const rs = M.shuffles.map(sh => sc[cid(s, b, sh)]).filter(Boolean);
        const mean = rs.reduce((a, r) => a + r.f1, 0) / (rs.length || 1);
        h += `<td class="${cls(mean)}" data-s="${s}" data-b="${b}">` +
          rs.map((r, i) => `<span class="sh">#${i}: ${esc(r.answer)} <span class="f">F1 ${r.f1.toFixed(2)}</span></span>`).join("") + "</td>";
      } else {
        const r = sc[cid(s, b)];
        h += r ? `<td class="${cls(r.f1)}" data-s="${s}" data-b="${b}">${esc(r.answer)}<div class="f">F1 ${r.f1.toFixed(2)}</div></td>` : "<td>-</td>";
      }
    }
    h += "</tr>";
  }
  $("#cmp-table").innerHTML = h + "</table>";
  document.querySelectorAll("#cmp-table td[data-s]").forEach(td => td.onclick = () => replay(q.id, td.dataset.s, +td.dataset.b));
}

// ------------------------------------------------------------------ 5. results
function metric(which, name, dataset = "hotpotqa") {
  const rows = which === "e2e" ? M.e2e : M.selection_study;
  const out = {};
  for (const r of rows) {
    if (r.metric !== name || r.dataset !== dataset) continue;
    (out[r.strategy] ??= {})[+r.budget_pct] = { mean: +r.mean, lo: +r.ci_low, hi: +r.ci_high };
  }
  return out;
}
function initResults() {
  segmented("#ss-ds", ["hotpotqa", "musique"], d => d === "hotpotqa" ? "HotpotQA" : "MuSiQue", () => state.ssDs, v => { state.ssDs = v; drawSel(); });
  drawSel();
  const e2e = metric("e2e", "f1");
  lineChart($("#chart-e2e"), {
    xs: M.budgets, yMax: 0.8, yLabel: "Token F1", fmtY: v => v.toFixed(1),
    series: M.strategies.map(s => ({ name: NAMES[s], color: COLOR(s), pts: M.budgets.map(b => e2e[s][b]) })),
  });
}
function drawSel() {
  const d = metric("sel", "selection_recall", state.ssDs), xs = [5, 10, 25, 50, 75];
  lineChart($("#chart-sel"), {
    xs, yMax: 1, yLabel: "Gold chunks selected", fmtY: v => Math.round(v * 100) + "%",
    series: [
      { name: "Oracle (upper bound)", color: COLOR("oracle"), dash: true, pts: xs.map(b => ({ mean: ORACLE[state.ssDs][b] })) },
      ...M.strategies.map(s => ({ name: NAMES[s], color: COLOR(s), pts: xs.map(b => d[s][b]) })),
    ],
  });
}
function lineChart(root, { xs, yMax, yLabel, fmtY, series }) {
  const W = 560, H = 320, L = 52, R = 14, T = 14, B = 40;
  const x = i => L + (W - L - R) * i / (xs.length - 1), y = v => T + (H - T - B) * (1 - v / yMax);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} by budget">`;
  s += `<g class="grid">`;
  for (let t = 0; t <= 4; t++) { const v = yMax * t / 4; s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${fmtY(v)}</text>`; }
  s += `</g>`;
  xs.forEach((b, i) => s += `<text x="${x(i)}" y="${H - B + 18}" text-anchor="middle">${b}%</text>`);
  s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Extraction budget</text>`;
  s += `<text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
  series.forEach((se, k) => {
    const off = (k - (series.length - 1) / 2) * 3;          // nudge sideways so intervals don't overlap
    se.pts.forEach((p, i) => { if (p && p.lo != null) s += `<line x1="${x(i) + off}" x2="${x(i) + off}" y1="${y(p.lo)}" y2="${y(p.hi)}" stroke="${se.color}" stroke-width="1.5" opacity=".55"/>`; });
    s += `<polyline fill="none" stroke="${se.color}" stroke-width="2" ${se.dash ? 'stroke-dasharray="5 4"' : ""} points="${se.pts.map((p, i) => `${x(i) + off},${y(p.mean)}`).join(" ")}"/>`;
    se.pts.forEach((p, i) => s += `<circle cx="${x(i) + off}" cy="${y(p.mean)}" r="4" fill="${se.color}" stroke="var(--surface)" stroke-width="2"/>`);
  });
  xs.forEach((b, i) => s += `<rect class="hit" data-i="${i}" x="${x(i) - 24}" y="${T}" width="48" height="${H - T - B}" fill="transparent"/>`);
  s += `</svg><div class="clegend">${series.map(se => `<span><i style="--c:${se.color}${se.dash ? ";background:repeating-linear-gradient(90deg,var(--c) 0 5px,transparent 5px 8px)" : ""}"></i>${se.name}</span>`).join("")}</div>`;
  root.innerHTML = s;
  root.querySelectorAll(".hit").forEach(r => {
    const i = +r.dataset.i;
    r.onmousemove = e => tip(e, `<b>${xs[i]}% budget</b><br>` + series.map(se => {
      const p = se.pts[i]; return `${se.name}: <b>${fmtY === undefined ? p.mean : fmt(p.mean)}</b>${p.lo != null ? ` [${fmt(p.lo)}, ${fmt(p.hi)}]` : ""}`;
    }).join("<br>"));
    r.onmouseleave = untip;
  });
}

// ------------------------------------------------------------------ utils
function fmt(v) { return v == null || isNaN(v) ? "-" : (+v).toFixed(3).replace(/^0/, ""); }
function pct(v) { return (v * 100).toFixed(1) + "%"; }
function tip(e, html) {
  const t = $("#tooltip"); t.innerHTML = html; t.style.opacity = 1;
  const x = Math.min(e.clientX + 14, innerWidth - t.offsetWidth - 8);
  t.style.left = x + "px"; t.style.top = (e.clientY + 14) + "px";
}
function untip() { $("#tooltip").style.opacity = 0; }
