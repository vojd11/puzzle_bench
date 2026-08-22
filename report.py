#!/usr/bin/env python3
"""Interactive HTML dashboard for zebra-bench results.

    python report.py results.jsonl [--out charts/report.html] [--open]

Reads a JSONL produced by bench.py and writes ONE self-contained HTML file
(no CDN, no network, works offline and as a Claude Artifact). Charts are drawn
as inline SVG with hover tooltips, a model on/off legend, light/dark themes,
and a sortable summary table.

Sibling of plots.py — same numbers, but a UI instead of PNGs.
"""
from __future__ import annotations

import argparse
import json
import os
import webbrowser
from collections import defaultdict

LEVEL_KEY = lambda lv: tuple(int(x) for x in lv.split("x"))  # noqa: E731


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def build_payload(rows, pass_ratio):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][r["level"]].append(r)
    models = sorted(agg)
    levels = sorted({r["level"] for r in rows}, key=LEVEL_KEY)

    per = {}          # model -> level -> metrics
    reached = {}      # model -> index of highest passed level (0 = none)
    totals = {}       # model -> overall aggregates
    for m in models:
        per[m] = {}
        best = -1
        stopped = False
        all_rs = []
        for lv in levels:
            rs = agg[m].get(lv)
            if not rs:
                continue
            all_rs += rs
            n = len(rs)
            solved = sum(1 for r in rs if r.get("correct"))
            pr = solved / n
            per[m][lv] = {
                "n": n,
                "solved": solved,
                "pass_rate": pr,
                "cell_acc": mean([r.get("cell_acc") for r in rs]),
                "latency": mean([r.get("latency") for r in rs]),
                "ptokens": mean([r.get("prompt_tokens") for r in rs]),
                "ctokens": mean([r.get("completion_tokens") for r in rs]),
                "parse_fail": sum(1 for r in rs if not r.get("parsed")),
                "code_flag": sum(1 for r in rs if r.get("code_flag")),
                "errors": sum(1 for r in rs if r.get("error")),
            }
            if not stopped:
                if pr >= pass_ratio:
                    best = levels.index(lv)
                else:
                    stopped = True
        reached[m] = best  # -1 means none
        totals[m] = {
            "n": len(all_rs),
            "solved": sum(1 for r in all_rs if r.get("correct")),
            "cell_acc": mean([r.get("cell_acc") for r in all_rs]),
            "latency": mean([r.get("latency") for r in all_rs]),
            "ctokens": mean([r.get("completion_tokens") for r in all_rs]),
            "parse_fail": sum(1 for r in all_rs if not r.get("parsed")),
            "code_flag": sum(1 for r in all_rs if r.get("code_flag")),
            "errors": sum(1 for r in all_rs if r.get("error")),
        }

    # rank models by highest level reached, then pass rate at that level, then cell acc
    def rank_key(m):
        idx = reached[m]
        lv = levels[idx] if idx >= 0 else None
        tie = per[m].get(lv, {}).get("pass_rate", 0) if lv else 0
        return (idx, tie, totals[m]["cell_acc"])

    order = sorted(models, key=rank_key, reverse=True)

    any_tokens = any(
        per[m][lv]["ctokens"] for m in models for lv in per[m] if per[m][lv]["ctokens"]
    )
    return {
        "models": models,
        "order": order,
        "levels": levels,
        "per": per,
        "reached": reached,
        "totals": totals,
        "pass_ratio": pass_ratio,
        "n_rows": len(rows),
        "has_tokens": any_tokens,
    }


def render_html(payload, title):
    data_json = json.dumps(payload, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)


# ---------------------------------------------------------------- template ---
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  color-scheme: light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --card:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --good:#0ca30c; --bad:#d03b3b; --warn:#fab219;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#4a3aa7; --s8:#e34948;
}
:root[data-theme="dark"], :root:where(:not([data-theme="light"])){}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    --plane:#0d0d0d; --surface:#1a1a19; --card:#1f1f1e;
    --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --good:#0ca30c; --bad:#d03b3b; --warn:#fab219;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  }
}
:root[data-theme="dark"]{
  --plane:#0d0d0d; --surface:#1a1a19; --card:#1f1f1e;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --good:#0ca30c; --bad:#d03b3b; --warn:#fab219;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:28px 20px 80px}
header.top{display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap}
h1{font-size:26px; margin:0 0 4px; letter-spacing:-0.02em}
.sub{color:var(--ink2); font-size:14px; margin:0}
.theme-btn{
  border:1px solid var(--ring); background:var(--card); color:var(--ink2);
  border-radius:9px; padding:7px 12px; font-size:13px; cursor:pointer; font-family:inherit;
}
.theme-btn:hover{color:var(--ink)}
.tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:22px 0}
.tile{background:var(--card); border:1px solid var(--ring); border-radius:14px; padding:16px 18px}
.tile .k{font-size:12.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em}
.tile .v{font-size:30px; font-weight:650; margin-top:4px; letter-spacing:-0.02em}
.tile .v small{font-size:15px; font-weight:500; color:var(--ink2)}
.tile .note{font-size:12.5px; color:var(--ink2); margin-top:2px}
.legend{display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 20px}
.chip{
  display:inline-flex; align-items:center; gap:7px; padding:5px 11px 5px 9px;
  border:1px solid var(--ring); border-radius:999px; background:var(--card);
  cursor:pointer; user-select:none; font-size:13px; color:var(--ink);
  transition:opacity .12s;
}
.chip .sw{width:11px; height:11px; border-radius:3px; flex:none}
.chip.off{opacity:.34}
.chip.off .name{text-decoration:line-through}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:18px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}}
.card{background:var(--card); border:1px solid var(--ring); border-radius:16px; padding:18px 18px 12px}
.card h2{font-size:15.5px; margin:0 0 2px; letter-spacing:-0.01em}
.card p.desc{font-size:12.5px; color:var(--ink2); margin:0 0 8px}
svg{display:block; width:100%; height:auto; overflow:visible}
.axis{stroke:var(--axis); stroke-width:1}
.grid-l{stroke:var(--grid); stroke-width:1}
.tick{fill:var(--muted); font-size:11px}
.tick.tab{font-variant-numeric:tabular-nums}
.mk-lbl{fill:var(--ink2); font-size:10.5px; font-variant-numeric:tabular-nums}
.passline{stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1}
.dot{stroke:var(--surface); stroke-width:2}
.tt{
  position:fixed; pointer-events:none; z-index:20; background:var(--card);
  border:1px solid var(--ring); border-radius:10px; padding:8px 10px; font-size:12.5px;
  box-shadow:0 6px 24px rgba(0,0,0,.16); min-width:120px; opacity:0; transition:opacity .08s;
}
.tt .h{font-weight:650; margin-bottom:4px}
.tt .row{display:flex; align-items:center; gap:7px; justify-content:space-between}
.tt .row .lft{display:flex; align-items:center; gap:6px}
.tt .sw{width:9px; height:9px; border-radius:2px; flex:none}
.tt b{font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto; border:1px solid var(--ring); border-radius:16px; margin-top:6px}
table{border-collapse:collapse; width:100%; font-size:13px; background:var(--card)}
th,td{padding:9px 12px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left; position:sticky; left:0; background:var(--card)}
thead th{
  color:var(--ink2); font-weight:600; cursor:pointer; user-select:none;
  border-bottom:1px solid var(--axis); position:sticky; top:0; z-index:1;
}
thead th:hover{color:var(--ink)}
tbody tr:hover td{background:color-mix(in srgb,var(--s1) 7%,var(--card))}
td.num{font-variant-numeric:tabular-nums}
.mdl{display:inline-flex; align-items:center; gap:7px}
.mdl .sw{width:10px; height:10px; border-radius:3px}
.bar-cell{position:relative}
.bar-fill{position:absolute; inset:0; opacity:.16; border-radius:0}
.foot{color:var(--muted); font-size:12px; margin-top:26px; text-align:center}
.badge{display:inline-block; padding:1px 7px; border-radius:6px; font-size:11px; font-weight:600}
.sec-title{font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:30px 0 10px}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div>
      <h1>__TITLE__</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <button class="theme-btn" id="themeBtn">◐ Theme</button>
  </header>

  <div class="tiles" id="tiles"></div>

  <div class="sec-title">Filter models</div>
  <div class="legend" id="legend"></div>

  <div class="card" style="margin-bottom:18px">
    <h2>Highest level cleared</h2>
    <p class="desc">Ranking. A level counts as cleared when the pass rate meets the threshold; the ladder stops at the first miss. Longer bar = solved bigger grids.</p>
    <svg id="chartLadder"></svg>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Solve rate by level</h2>
      <p class="desc">Share of puzzles solved fully correctly. Dashed line = pass threshold.</p>
      <svg id="chartPass"></svg>
    </div>
    <div class="card">
      <h2>Cell accuracy by level</h2>
      <p class="desc">Partial credit — average share of correct grid cells. Shows "almost solved".</p>
      <svg id="chartCell"></svg>
    </div>
  </div>

  <div class="grid2" id="costRow" style="margin-top:18px">
    <div class="card">
      <h2>Latency per puzzle</h2>
      <p class="desc">Mean wall-clock seconds per attempt, by level.</p>
      <svg id="chartLat"></svg>
    </div>
    <div class="card">
      <h2>Completion tokens per puzzle</h2>
      <p class="desc">Mean output tokens per attempt, by level.</p>
      <svg id="chartTok"></svg>
    </div>
  </div>

  <div class="sec-title">Full results</div>
  <div class="tablewrap">
    <table id="summary"><thead></thead><tbody></tbody></table>
  </div>

  <p class="foot" id="foot"></p>
</div>

<div class="tt" id="tt"></div>

<script>
const DATA = __DATA__;
const SC = ['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const cssv = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = i => cssv(SC[i % SC.length]);
const modelColor = {};
DATA.order.forEach((m,i)=>{ modelColor[m] = SC[i % SC.length]; });
const active = new Set(DATA.order);
const tt = document.getElementById('tt');
const SVGNS = 'http://www.w3.org/2000/svg';
const fmtPct = x => (x*100).toFixed(0)+'%';
const fmtPct1 = x => (x*100).toFixed(1)+'%';
const shortName = m => m.includes('/') ? m.split('/').slice(-1)[0] : m;
// HTML-escape anything data-derived (model names) before it goes into innerHTML.
const esc = s => String(s).replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sn = m => esc(shortName(m));   // safe short name for HTML contexts
const clip = (s,n=13) => s.length>n ? s.slice(0,n-1)+'…' : s;

function el(tag, attrs={}, parent=null){
  const e = document.createElementNS(SVGNS, tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]);
  if(parent) parent.appendChild(e);
  return e;
}
function showTT(html, evt){
  tt.innerHTML = html; tt.style.opacity = 1;
  const pad=14, w=tt.offsetWidth, h=tt.offsetHeight;
  let x=evt.clientX+pad, y=evt.clientY+pad;
  if(x+w>innerWidth) x=evt.clientX-w-pad;
  if(y+h>innerHeight) y=evt.clientY-h-pad;
  tt.style.left=x+'px'; tt.style.top=y+'px';
}
const hideTT = ()=>{ tt.style.opacity=0; };

// ---- header, tiles ----
function topModel(){
  for(const m of DATA.order) return m;
}
function renderHeader(){
  document.getElementById('subtitle').textContent =
    `${DATA.models.length} models · ${DATA.levels.length} levels (${DATA.levels[0]}–${DATA.levels[DATA.levels.length-1]}) · ${DATA.n_rows} attempts · pass threshold ${fmtPct(DATA.pass_ratio)}`;
  document.getElementById('foot').textContent =
    'Generated by report.py from zebra-bench results · self-contained, offline · click column headers to sort, chips to filter';
}
function renderTiles(){
  const best = DATA.order[0];
  const bi = DATA.reached[best];
  const bestLv = bi>=0 ? DATA.levels[bi] : 'none';
  // hardest level any model cleared
  let hardest=-1, hardestModel='—';
  for(const m of DATA.models){ if(DATA.reached[m]>hardest){hardest=DATA.reached[m]; hardestModel=m;} }
  // overall best cell acc
  let bestCell=-1, bestCellM='—';
  for(const m of DATA.models){ const c=DATA.totals[m].cell_acc; if(c>bestCell){bestCell=c; bestCellM=m;} }
  const tiles = [
    {k:'Top model', v:sn(best), note:`cleared ${bestLv}`},
    {k:'Hardest level cleared', v:hardest>=0?DATA.levels[hardest]:'none', note:`by ${sn(hardestModel)}`},
    {k:'Best cell accuracy', v:fmtPct(bestCell), note:sn(bestCellM)},
    {k:'Total attempts', v:String(DATA.n_rows), note:`${DATA.models.length} models`},
  ];
  document.getElementById('tiles').innerHTML = tiles.map(t=>
    `<div class="tile"><div class="k">${t.k}</div><div class="v">${t.v}</div><div class="note">${t.note}</div></div>`
  ).join('');
}

// ---- legend ----
function renderLegend(){
  const box = document.getElementById('legend');
  box.innerHTML='';
  DATA.order.forEach(m=>{
    const c = document.createElement('div');
    c.className='chip'+(active.has(m)?'':' off');
    c.innerHTML = `<span class="sw" style="background:${color(DATA.order.indexOf(m))}"></span><span class="name">${sn(m)}</span>`;
    c.onclick = ()=>{
      if(active.has(m)) active.delete(m); else active.add(m);
      if(active.size===0) active.add(m); // never empty
      c.className='chip'+(active.has(m)?'':' off');
      renderAll();
    };
    box.appendChild(c);
  });
}
const shown = ()=> DATA.order.filter(m=>active.has(m));

// ---- ladder (horizontal bars) ----
function renderLadder(){
  const svg = document.getElementById('chartLadder');
  svg.innerHTML='';
  const ms = shown();
  const L = DATA.levels.length;
  const W=920, rowH=34, padT=8, padB=34, padL=150, padR=20;
  const H = padT+padB+ms.length*rowH;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const plotW = W-padL-padR;
  const x = i => padL + (i/L)*plotW;          // i in [0..L] units of levels
  // vertical gridlines per level
  for(let i=0;i<=L;i++){
    el('line',{x1:x(i),y1:padT-2,x2:x(i),y2:padT+ms.length*rowH,class:'grid-l'},svg);
    if(i<L){
      const t=el('text',{x:(x(i)+x(i+1))/2,y:padT+ms.length*rowH+16,'text-anchor':'middle',class:'tick tab'},svg);
      t.textContent=DATA.levels[i];
    }
  }
  const t0=el('text',{x:padL,y:padT+ms.length*rowH+30,'text-anchor':'start',class:'tick'},svg);
  t0.textContent='none';
  ms.forEach((m,r)=>{
    const idx = DATA.reached[m]; // -1..L-1
    const y = padT + r*rowH + 6;
    const bh = rowH-14;
    const w = idx>=0 ? x(idx+1)-padL : 0;
    // track
    el('rect',{x:padL,y,width:plotW,height:bh,rx:6,fill:'var(--grid)','opacity':0.4},svg);
    if(idx>=0){
      const rect=el('rect',{x:padL,y,width:Math.max(w,3),height:bh,rx:6,fill:color(DATA.order.indexOf(m))},svg);
      rect.style.cursor='pointer';
      rect.addEventListener('mousemove',e=>showTT(
        `<div class="h">${sn(m)}</div>cleared up to <b>${DATA.levels[idx]}</b>`,e));
      rect.addEventListener('mouseleave',hideTT);
    }
    const lbl=el('text',{x:padL-12,y:y+bh/2+4,'text-anchor':'end',class:'mk-lbl'},svg);
    lbl.style.fill='var(--ink)'; lbl.style.fontSize='12px';
    lbl.textContent=shortName(m);
    const tag=el('text',{x:idx>=0?x(idx+1)+8:padL+8,y:y+bh/2+4,class:'mk-lbl'},svg);
    tag.textContent= idx>=0?DATA.levels[idx]:'none';
  });
}

// ---- grouped bars (pass rate) ----
function renderPass(){
  const svg=document.getElementById('chartPass');
  svg.innerHTML='';
  const ms=shown(), lv=DATA.levels;
  const W=560,H=300,padL=42,padR=12,padT=12,padB=40;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const y=v=>padT+plotH*(1-v);
  // y grid 0..1
  for(let g=0;g<=5;g++){
    const v=g/5;
    el('line',{x1:padL,y1:y(v),x2:W-padR,y2:y(v),class:'grid-l'},svg);
    const t=el('text',{x:padL-7,y:y(v)+3,'text-anchor':'end',class:'tick tab'},svg);
    t.textContent=fmtPct(v);
  }
  // pass threshold
  el('line',{x1:padL,y1:y(DATA.pass_ratio),x2:W-padR,y2:y(DATA.pass_ratio),class:'passline'},svg);
  const bandW=plotW/lv.length;
  const n=ms.length;
  const gw=Math.min(bandW*0.8/n, 26);
  lv.forEach((L,li)=>{
    const cx=padL+bandW*(li+0.5);
    const t=el('text',{x:cx,y:H-padB+16,'text-anchor':'middle',class:'tick tab'},svg);
    t.textContent=L;
    ms.forEach((m,mi)=>{
      const d=DATA.per[m][L];
      if(!d) return;
      const bx=cx-(n*gw)/2+mi*gw+1;
      const val=d.pass_rate;
      const hh=plotH*val;
      const rect=el('rect',{x:bx,y:y(val),width:gw-2,height:Math.max(hh,val>0?2:0),rx:3,
        fill:color(DATA.order.indexOf(m))},svg);
      rect.style.cursor='pointer';
      rect.addEventListener('mousemove',e=>showTT(
        `<div class="h">${sn(m)} · ${L}</div>`+
        `<div class="row"><span class="lft"><span class="sw" style="background:${color(DATA.order.indexOf(m))}"></span>solve rate</span><b>${fmtPct1(val)}</b></div>`+
        `<div class="row"><span class="lft">solved</span><b>${d.solved}/${d.n}</b></div>`,e));
      rect.addEventListener('mouseleave',hideTT);
    });
  });
  axisTitle(svg,'level','solved',W,H,padL,padB);
}

// ---- line chart helper ----
function lineChart(svgId, accessor, fmtY, yMax, yTicks){
  const svg=document.getElementById(svgId);
  svg.innerHTML='';
  const ms=shown(), lv=DATA.levels;
  const W=560,H=300,padL=52,padR=84,padT=12,padB=40;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  const plotW=W-padL-padR, plotH=H-padT-padB;
  // compute max
  let mx=yMax;
  if(mx==null){
    mx=0;
    ms.forEach(m=>lv.forEach(L=>{const d=DATA.per[m][L]; if(d){const v=accessor(d); if(v>mx)mx=v;}}));
    mx=mx*1.1||1;
  }
  const x=i=>padL+(lv.length<=1?plotW/2:(i/(lv.length-1))*plotW);
  const y=v=>padT+plotH*(1-v/mx);
  const nt=yTicks||5;
  for(let g=0;g<=nt;g++){
    const v=mx*g/nt;
    el('line',{x1:padL,y1:y(v),x2:W-padR,y2:y(v),class:'grid-l'},svg);
    const t=el('text',{x:padL-7,y:y(v)+3,'text-anchor':'end',class:'tick tab'},svg);
    t.textContent=fmtY(v);
  }
  lv.forEach((L,i)=>{
    const t=el('text',{x:x(i),y:H-padB+16,'text-anchor':'middle',class:'tick tab'},svg);
    t.textContent=L;
  });
  const endLabels=[]; // {y, wantY, m, ci}
  ms.forEach(m=>{
    const ci=DATA.order.indexOf(m);
    const pts=[];
    lv.forEach((L,i)=>{const d=DATA.per[m][L]; if(d) pts.push([i,accessor(d),L,d]);});
    if(pts.length===0) return;
    let dstr='';
    pts.forEach((p,k)=>{ dstr+=(k?'L':'M')+x(p[0])+' '+y(p[1]); });
    el('path',{d:dstr,fill:'none',stroke:color(ci),'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
    pts.forEach(p=>{
      const c=el('circle',{cx:x(p[0]),cy:y(p[1]),r:4.5,fill:color(ci),class:'dot'},svg);
      c.style.cursor='pointer';
      c.addEventListener('mousemove',e=>showTT(
        `<div class="h">${sn(m)} · ${p[2]}</div>`+
        `<div class="row"><span class="lft"><span class="sw" style="background:${color(ci)}"></span>value</span><b>${fmtY(p[1])}</b></div>`,e));
      c.addEventListener('mouseleave',hideTT);
    });
    const last=pts[pts.length-1];
    endLabels.push({x:x(last[0])+8, wantY:y(last[1])+3, m, ci});
  });
  // de-collide end labels vertically (relief for contrast + identity)
  endLabels.sort((a,b)=>a.wantY-b.wantY);
  const gap=13; let prev=-1e9;
  endLabels.forEach(o=>{ o.y=Math.max(o.wantY, prev+gap); prev=o.y; });
  // pull back into the plot band if they overflowed the bottom
  const over=endLabels.length?endLabels[endLabels.length-1].y-(padT+plotH):0;
  if(over>0) endLabels.forEach(o=>o.y-=over);
  endLabels.forEach(o=>{
    if(o.y!==o.wantY-((over>0)?over:0)){
      el('line',{x1:o.x-5,y1:o.wantY-3,x2:o.x-1,y2:o.y-3,stroke:color(o.ci),'stroke-width':1,'opacity':0.5},svg);
    }
    const t=el('text',{x:o.x,y:o.y,class:'mk-lbl'},svg);
    t.style.fill=color(o.ci); t.textContent=clip(shortName(o.m));
  });
}
function axisTitle(svg,xt,yt,W,H,padL,padB){
  const t=el('text',{x:W/2,y:H-4,'text-anchor':'middle',class:'tick'},svg);
  t.textContent=xt;
}

// ---- table ----
let sortState={key:'rank',dir:1};
function tableRows(){
  const rows=[];
  DATA.order.forEach((m,i)=>{
    const t=DATA.totals[m];
    const idx=DATA.reached[m];
    rows.push({
      model:m, ci:i,
      rank:DATA.order.indexOf(m),
      cleared: idx>=0?DATA.levels[idx]:'none',
      clearedIdx: idx,
      solve: t.n?t.solved/t.n:0,
      cell: t.cell_acc,
      lat: t.latency,
      tok: t.ctokens,
      pf: t.parse_fail,
      cf: t.code_flag,
      err: t.errors,
      n: t.n,
    });
  });
  return rows;
}
function renderTable(){
  const cols=[
    ['model','Model',false],['cleared','Cleared',true],['solve','Solve rate',true],
    ['cell','Cell acc',true],['lat','Latency',true],
    ...(DATA.has_tokens?[['tok','Out tokens',true]]:[]),
    ['pf','Parse fail',true],['cf','Code flag',true],['err','Errors',true],['n','Attempts',true],
  ];
  const thead=document.querySelector('#summary thead');
  thead.innerHTML='<tr>'+cols.map(c=>{
    const arrow = sortState.key===c[0] ? (sortState.dir>0?' ▲':' ▼') : '';
    return `<th data-k="${c[0]}">${c[1]}${arrow}</th>`;
  }).join('')+'</tr>';
  thead.querySelectorAll('th').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(sortState.key===k) sortState.dir*=-1;
      else { sortState.key=k; sortState.dir = (k==='model'||k==='cleared')?1:-1; }
      renderTable();
    };
  });
  let rows=tableRows();
  const k=sortState.key;
  rows.sort((a,b)=>{
    let av,bv;
    if(k==='cleared'){av=a.clearedIdx;bv=b.clearedIdx;}
    else if(k==='rank'){av=a.rank;bv=b.rank;}
    else {av=a[k];bv=b[k];}
    if(typeof av==='string') return sortState.dir*av.localeCompare(bv);
    return sortState.dir*(av-bv);
  });
  const maxLat=Math.max(...rows.map(r=>r.lat),0.001);
  const tbody=document.querySelector('#summary tbody');
  tbody.innerHTML=rows.map(r=>{
    const sw=`<span class="sw" style="background:${color(r.ci)}"></span>`;
    const clearedBadge = r.clearedIdx>=0
      ? `<span class="badge" style="background:color-mix(in srgb,${color(r.ci)} 20%,transparent);color:var(--ink)">${r.cleared}</span>`
      : `<span class="badge" style="background:color-mix(in srgb,var(--bad) 16%,transparent)">none</span>`;
    const cells=[
      `<td><span class="mdl">${sw}${sn(r.model)}</span></td>`,
      `<td>${clearedBadge}</td>`,
      `<td class="num">${fmtPct1(r.solve)}</td>`,
      `<td class="num">${fmtPct1(r.cell)}</td>`,
      `<td class="num bar-cell"><div class="bar-fill" style="background:${color(r.ci)};width:${(r.lat/maxLat*100).toFixed(0)}%"></div>${r.lat.toFixed(2)}s</td>`,
      ...(DATA.has_tokens?[`<td class="num">${r.tok?Math.round(r.tok):'—'}</td>`]:[]),
      `<td class="num" style="${r.pf?'color:var(--warn)':''}">${r.pf}</td>`,
      `<td class="num" style="${r.cf?'color:var(--warn)':''}">${r.cf}</td>`,
      `<td class="num" style="${r.err?'color:var(--bad)':''}">${r.err}</td>`,
      `<td class="num">${r.n}</td>`,
    ];
    return '<tr>'+cells.join('')+'</tr>';
  }).join('');
}

// ---- render everything ----
function renderAll(){
  renderLadder();
  renderPass();
  lineChart('chartCell', d=>d.cell_acc, fmtPct, 1.0, 5);
  lineChart('chartLat', d=>d.latency, v=>v.toFixed(v<10?1:0)+'s', null, 5);
  if(DATA.has_tokens){
    document.getElementById('costRow').style.display='';
    lineChart('chartTok', d=>d.ctokens, v=>v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0), null, 5);
  } else {
    // hide token chart, keep latency full width
    document.getElementById('chartTok').closest('.card').style.display='none';
  }
  renderTable();
}

// theme toggle
const themeBtn=document.getElementById('themeBtn');
themeBtn.onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next = cur==='dark' ? 'light' : (cur==='light' ? 'dark' :
    (matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next);
  renderAll();
};
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change',renderAll);
window.addEventListener('resize',()=>{ /* svg is viewBox-scaled; only tooltip needs no redraw */ });

renderHeader();
renderTiles();
renderLegend();
renderAll();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results.jsonl")
    ap.add_argument("--out", default="charts/report.html")
    ap.add_argument("--pass-ratio", type=float, default=0.67)
    ap.add_argument("--title", default="Zebra Bench — LLM logic-puzzle leaderboard")
    ap.add_argument("--open", action="store_true", help="open the report in a browser")
    a = ap.parse_args()

    rows = load(a.results)
    payload = build_payload(rows, a.pass_ratio)
    html = render_html(payload, a.title)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"dashboard -> {a.out}  ({payload['n_rows']} attempts, {len(payload['models'])} models)")
    if a.open:
        webbrowser.open("file://" + os.path.abspath(a.out))


if __name__ == "__main__":
    main()
