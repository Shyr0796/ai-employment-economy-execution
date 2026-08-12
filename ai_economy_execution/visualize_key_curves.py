from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


STRATEGIES: dict[str, dict[str, Any]] = {
    "passive_safety_net": {
        "label": "被动保障",
        "color": "#a7b0bb",
        "dash": "7 5",
    },
    "active_demand": {
        "label": "主动需求",
        "color": "#2f8cff",
        "dash": "",
        "highlight": True,
    },
    "productivity_dividend": {
        "label": "生产率红利",
        "color": "#35d0a6",
        "dash": "3 4",
    },
    "fiscal_guard": {
        "label": "财政守纪律",
        "color": "#6f7884",
        "dash": "1 5",
    },
    "active_demand_regulation": {
        "label": "主动需求＋监管",
        "color": "#f2b84b",
        "dash": "10 4 2 4",
    },
}


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _trailing_average(values: list[float], window: int = 3) -> list[float]:
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _series(
    histories: dict[str, list[dict[str, str]]],
    field: str,
    transform: Any,
    *,
    smooth: bool = True,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for strategy, meta in STRATEGIES.items():
        rows = histories[strategy]
        values = [transform(float(row[field]), rows) for row in rows]
        if smooth:
            values = _trailing_average(values)
        result.append(
            {
                "name": meta["label"],
                "key": strategy,
                "color": meta["color"],
                "dash": meta["dash"],
                "highlight": bool(meta.get("highlight", False)),
                "points": [
                    {"x": int(row["month"]), "y": round(value, 6)}
                    for row, value in zip(rows, values)
                ],
            }
        )
    return result


def build_spec(result_dir: Path) -> dict[str, Any]:
    path_root = result_dir / "paths" / "seed_1" / "mixed_competition"
    histories = {
        strategy: _read_history(path_root / strategy / "metrics.csv")
        for strategy in STRATEGIES
    }
    for strategy, history in histories.items():
        if len(history) != 120:
            raise ValueError(f"{strategy} has {len(history)} rows; expected 120")

    results = json.loads(
        (result_dir / "strategy_results.json").read_text(encoding="utf-8")
    )
    aggregate = {
        row["strategy"]: row
        for row in results["aggregate"]
        if row["culture"] == "mixed_competition"
    }
    active = aggregate["active_demand"]
    passive = aggregate["passive_safety_net"]
    consumption_delta = (
        active["real_consumption_gain_pct_mean"]
        - passive["real_consumption_gain_pct_mean"]
    )
    peak_delta = (
        active["peak_unemployment_change_pp_mean"]
        - passive["peak_unemployment_change_pp_mean"]
    )
    exit_delta = active["cumulative_firm_exits_mean"] - passive["cumulative_firm_exits_mean"]

    baseline_consumption = {
        strategy: float(history[23]["real_consumption"])
        for strategy, history in histories.items()
    }

    return {
        "eyebrow": "URBAN CUP / AI EMPLOYMENT / V6 SMOKE",
        "title": "AI冲击之后，主动需求管理只部分缓解失业峰值",
        "subtitle": (
            "混合企业文化、完整竞争下的五种政府策略。曲线显示机制如何展开，"
            "但单种子结果不能用作政策排名。"
        ),
        "focus": "主动需求管理",
        "kpis": [
            {
                "label": "累计实际消费",
                "value": f"{consumption_delta:+.3f} pp",
                "delta": "主动需求相对被动保障",
                "tone": "positive" if consumption_delta > 0 else "negative",
            },
            {
                "label": "失业峰值",
                "value": f"{peak_delta:+.2f} pp",
                "delta": "越低越好；相对被动保障",
                "tone": "positive" if peak_delta < 0 else "negative",
            },
            {
                "label": "企业退出",
                "value": f"{int(active['cumulative_firm_exits_mean'])}",
                "delta": f"{exit_delta:+.0f} 家 vs 被动保障",
                "tone": "positive" if exit_delta < 0 else "negative",
            },
            {
                "label": "证据规模",
                "value": "N = 1",
                "delta": "500户 · 42条路径 · 120个月",
                "tone": "warning",
            },
        ],
        "charts": [
            {
                "kind": "line",
                "title": "失业率",
                "subtitle": "3个月移动平均；越低越好",
                "unit": "%",
                "includeZero": True,
                "series": _series(
                    histories,
                    "unemployment_rate",
                    lambda value, _rows: 100.0 * value,
                ),
                "note": "第25月启动AI。主动需求仅小幅压低峰值，无法消除完整竞争带来的就业冲击。",
            },
            {
                "kind": "line",
                "title": "居民实际消费指数",
                "subtitle": "第24月 = 100；3个月移动平均；越高越好",
                "unit": "指数",
                "reference": 100.0,
                "series": _series(
                    histories,
                    "real_consumption",
                    lambda value, rows: 100.0
                    * value
                    / baseline_consumption[rows[0]["government_policy_strategy"]],
                ),
                "note": "个人AI支出已从福利性实际消费中剔除；指数不把工作软件购买误当作生活改善。",
            },
            {
                "kind": "line",
                "title": "存续企业数量",
                "subtitle": "当月企业数；越高不必然越好，但快速下坠表示退出压力",
                "unit": "家",
                "includeZero": True,
                "step": True,
                "series": _series(
                    histories,
                    "firm_count",
                    lambda value, _rows: value,
                    smooth=False,
                ),
                "note": "主动需求在本路径中退出29家，被动保障退出31家；监管策略退出37家，提示规则可能引发内生调整。",
            },
            {
                "kind": "line",
                "title": "政府实际采购",
                "subtitle": "3个月移动平均；千价值单位/月",
                "unit": "千",
                "includeZero": True,
                "series": _series(
                    histories,
                    "government_real_procurement",
                    lambda value, _rows: value / 1000.0,
                ),
                "note": "主动需求含定向就业稳定采购；生产率红利策略只在被动保障之上增加红利采购。",
            },
        ],
        "source": (
            f"Source: {result_dir.as_posix()}. "
            "Population 500; seed 1; months 1–120; AI shock at month 25; mixed firm cultures with full competition. "
            "Lines use three-month trailing means except firm count. Month-24 consumption is the normalization reference. "
            "All series are exploratory model output, not observed national statistics or causal estimates."
        ),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Urban Cup v6 关键动态曲线</title>
  <style>
    :root{--bg:#0b0d10;--surface:#11151b;--raised:#171c24;--text:#f4f7fb;--muted:#a7b0bb;--faint:#6f7884;--grid:#2a3039;--focus:#2f8cff;--positive:#35d0a6;--warning:#f2b84b;--negative:#ff6b7a;--sans:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;--mono:"SFMono-Regular",Consolas,"Liberation Mono",monospace}
    *{box-sizing:border-box}html{background:var(--bg)}body{margin:0;background:radial-gradient(circle at 74% -8%,rgba(47,140,255,.11),transparent 31rem),var(--bg);color:var(--text);font-family:var(--sans);-webkit-font-smoothing:antialiased}.shell{width:min(1220px,calc(100% - 48px));margin:auto;padding:46px 0 58px}.eyebrow{margin:0 0 16px;color:var(--focus);font:12px var(--mono);letter-spacing:.12em}.title{max-width:980px;margin:0;font-size:clamp(34px,5vw,56px);line-height:1.06;letter-spacing:-.042em}.subtitle{max-width:900px;margin:22px 0 0;color:var(--muted);font-size:18px;line-height:1.7}.focus{display:inline-flex;margin-top:18px;padding:6px 10px;border:1px solid rgba(47,140,255,.3);border-radius:8px;background:rgba(47,140,255,.12);color:#dcecff;font:12px var(--mono)}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:46px 0 62px}.kpi{min-height:132px;padding:20px;border:1px solid #20262e;border-radius:17px;background:linear-gradient(145deg,rgba(255,255,255,.018),transparent 55%),var(--surface)}.kpi-label{color:var(--muted);font-size:12px}.kpi-value{margin-top:17px;font:650 clamp(24px,3vw,34px) var(--mono);letter-spacing:-.04em}.kpi-delta{margin-top:10px;color:var(--faint);font:11px var(--mono)}.kpi[data-tone=positive] .kpi-delta{color:var(--positive)}.kpi[data-tone=negative] .kpi-delta{color:var(--negative)}.kpi[data-tone=warning] .kpi-delta{color:var(--warning)}.section-head{display:flex;justify-content:space-between;align-items:end;gap:18px;padding-bottom:14px;margin-bottom:18px;border-bottom:1px solid var(--grid)}.section-head h2{margin:0;font-size:24px}.section-head span{color:var(--faint);font:11px var(--mono)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{min-width:0;padding:23px;border:1px solid #20262e;border-radius:17px;background:var(--surface);overflow:hidden;animation:rise .5s cubic-bezier(.22,1,.36,1) both}.card:nth-child(2){animation-delay:.05s}.card:nth-child(3){animation-delay:.1s}.card:nth-child(4){animation-delay:.15s}.card h3{margin:0;font-size:17px}.chart-subtitle{min-height:19px;margin:6px 0 13px;color:var(--faint);font-size:12px;line-height:1.55}.plot svg{display:block;width:100%;height:auto;overflow:visible}.note{margin:12px 0 0;padding-top:11px;border-top:1px solid rgba(42,48,57,.75);color:var(--faint);font-size:10px;line-height:1.6}.source{margin-top:48px;padding-top:17px;border-top:1px solid var(--grid);color:var(--faint);font-size:11px;line-height:1.75}.source strong{color:var(--muted)}.tick,.axis,.end-label,.shock-label{font-family:var(--mono);font-variant-numeric:tabular-nums}.tick{fill:var(--faint);font-size:10px}.axis{fill:var(--muted);font-size:10px}.end-label{font-size:10px;font-weight:650}.shock-label{fill:var(--warning);font-size:9px}.gridline{stroke:var(--grid);stroke-width:1;vector-effect:non-scaling-stroke}@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}@media(max-width:900px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.grid{grid-template-columns:1fr}}@media(max-width:560px){.shell{width:min(100% - 26px,1220px);padding-top:28px}.kpis{gap:10px;margin-bottom:46px}.kpi{min-height:112px;padding:15px}.card{padding:17px 12px}.section-head{align-items:start;flex-direction:column}.subtitle{font-size:16px}}@media(prefers-reduced-motion:reduce){.card{animation:none}}@media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact}.shell{width:100%;padding:20px}.card,.kpi{break-inside:avoid}}
  </style>
</head>
<body><main class="shell">
  <header><p class="eyebrow" id="eyebrow"></p><h1 class="title" id="title"></h1><p class="subtitle" id="subtitle"></p><span class="focus" id="focus"></span></header>
  <section class="kpis" id="kpis" aria-label="关键指标"></section>
  <section><div class="section-head"><h2>关键动态曲线</h2><span>DIRECT LABELS / MONTH 1–120 / SHOCK M25</span></div><div class="grid" id="charts"></div></section>
  <footer class="source"><strong>数据、周期与变换</strong><br><span id="source"></span></footer>
</main>
<script>
const DATA=/*__DATA__*/;const NS="http://www.w3.org/2000/svg";
function n(tag,a={},t=null){const x=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))x.setAttribute(k,v);if(t!==null)x.textContent=t;return x}
function h(tag,c,t){const x=document.createElement(tag);if(c)x.className=c;if(t!==undefined)x.textContent=t;return x}
function scale(v,d0,d1,r0,r1){return r0+(v-d0)/(d1-d0)*(r1-r0)}
function format(v,unit){if(unit==="%")return v.toFixed(1)+"%";if(unit==="家")return Math.round(v)+" 家";if(unit==="指数")return v.toFixed(1);return v.toFixed(1)+" 千"}
function linePath(points,x,y,step){if(!points.length)return"";let d=`M${x(points[0].x)},${y(points[0].y)}`;for(let i=1;i<points.length;i++){const p=points[i],prev=points[i-1];d+=step?` H${x(p.x)} V${y(p.y)}`:` L${x(p.x)},${y(p.y)}`}return d}
function render(chart){const W=780,H=350,m={l:58,r:150,t:28,b:42},pw=W-m.l-m.r,ph=H-m.t-m.b;const values=chart.series.flatMap(s=>s.points.map(p=>p.y));let lo=Math.min(...values),hi=Math.max(...values);if(chart.includeZero)lo=0;if(Number.isFinite(chart.reference)){lo=Math.min(lo,chart.reference);hi=Math.max(hi,chart.reference)}const pad=(hi-lo||1)*.08;if(!chart.includeZero)lo-=pad;hi+=pad;const x=v=>scale(v,1,120,m.l,m.l+pw),y=v=>scale(v,lo,hi,m.t+ph,m.t);const svg=n("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":`${chart.title}，五种策略从第1月到第120月`});svg.append(n("title",{},chart.title));for(let i=0;i<=4;i++){const val=lo+(hi-lo)*i/4,yy=y(val);svg.append(n("line",{x1:m.l,y1:yy,x2:m.l+pw,y2:yy,class:"gridline"}));svg.append(n("text",{x:m.l-9,y:yy+3,"text-anchor":"end",class:"tick"},format(val,chart.unit)))}for(const month of [1,25,60,90,120]){const xx=x(month);svg.append(n("line",{x1:xx,y1:m.t,x2:xx,y2:m.t+ph,class:"gridline"}));svg.append(n("text",{x:xx,y:H-12,"text-anchor":"middle",class:"tick"},`M${month}`))}if(Number.isFinite(chart.reference)){const yy=y(chart.reference);svg.append(n("line",{x1:m.l,y1:yy,x2:m.l+pw,y2:yy,stroke:"#a7b0bb","stroke-width":1,"stroke-dasharray":"4 4",opacity:.55}))}const sx=x(25);svg.append(n("line",{x1:sx,y1:m.t,x2:sx,y2:m.t+ph,stroke:"#f2b84b","stroke-width":1.5,"stroke-dasharray":"5 5"}));svg.append(n("text",{x:sx+5,y:m.t+10,class:"shock-label"},"AI冲击"));for(const s of [...chart.series].sort((a,b)=>a.highlight-b.highlight)){svg.append(n("path",{d:linePath(s.points,x,y,chart.step),fill:"none",stroke:s.color,"stroke-width":s.highlight?3.2:1.8,"stroke-dasharray":s.dash||"","stroke-linejoin":"round","stroke-linecap":"round",opacity:s.highlight?1:.82,"vector-effect":"non-scaling-stroke"}))}let labels=chart.series.map(s=>({s,raw:y(s.points.at(-1).y),value:s.points.at(-1).y})).sort((a,b)=>a.raw-b.raw);const gap=16;labels.forEach((d,i)=>{d.ly=Math.max(d.raw,i?labels[i-1].ly+gap:m.t+7)});const overflow=labels.at(-1).ly-(m.t+ph-3);if(overflow>0)labels.forEach(d=>d.ly-=overflow);labels.forEach(d=>{const ex=x(120),ey=y(d.value);svg.append(n("line",{x1:ex,y1:ey,x2:m.l+pw+10,y2:d.ly,stroke:d.s.color,"stroke-width":1,opacity:.7}));svg.append(n("circle",{cx:ex,cy:ey,r:d.s.highlight?3.7:2.6,fill:d.s.color}));svg.append(n("text",{x:m.l+pw+14,y:d.ly+3,class:"end-label",fill:d.s.color},`${d.s.name} ${format(d.value,chart.unit)}`))});return svg}
function mount(){document.getElementById("eyebrow").textContent=DATA.eyebrow;document.getElementById("title").textContent=DATA.title;document.getElementById("subtitle").textContent=DATA.subtitle;document.getElementById("focus").textContent="FOCUS / "+DATA.focus;const k=document.getElementById("kpis");DATA.kpis.forEach(v=>{const c=h("article","kpi");c.dataset.tone=v.tone;c.append(h("div","kpi-label",v.label),h("div","kpi-value",v.value),h("div","kpi-delta",v.delta));k.append(c)});const charts=document.getElementById("charts");DATA.charts.forEach(v=>{const c=h("article","card");c.append(h("h3","",v.title),h("p","chart-subtitle",v.subtitle));const p=h("div","plot");p.append(render(v));c.append(p,h("p","note",v.note));charts.append(c)});document.getElementById("source").textContent=DATA.source;document.title=DATA.title;resizeParent()}
function resizeParent(){if(window.parent!==window)window.parent.postMessage({type:"urban-cup-key-curves-height",height:document.documentElement.scrollHeight},"*")}
mount();addEventListener("resize",()=>requestAnimationFrame(resizeParent));
</script></body></html>'''


def write_dashboard(result_dir: Path) -> tuple[Path, Path]:
    spec = build_spec(result_dir)
    json_path = result_dir / "key_curves_data.json"
    html_path = result_dir / "key_curves.html"
    json_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html_path.write_text(HTML_TEMPLATE.replace("/*__DATA__*/", payload), encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the strategy-experiment key-curves dashboard"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="A strategy result directory containing strategy_results.json",
    )
    args = parser.parse_args()
    json_path, html_path = write_dashboard(args.result_dir.resolve())
    print(json.dumps({"data": str(json_path), "html": str(html_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
