#!/usr/bin/env python3
"""Summarize a results JSONL into report.md + recall/QPS plots.

    python report.py $ANNS_WORK/results/single_gpu.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import load_jsonl

TARGETS = [0.90, 0.95, 0.99]
COLORS = {"cagra": "#1f77b4", "jasper": "#d62728", "tagore": "#2ca02c", "scalegann": "#9467bd"}
STYLES = ["-", "--", ":", "-."]


def _key(r):
    return (r["kind"], r["dataset"], r["system"], r.get("variant", ""), r.get("searcher", ""), r.get("dtype", ""))


def latest(records):
    """Keep the last successful record per key (re-runs override)."""
    out = {}
    for r in records:
        if r.get("ok"):
            out[_key(r)] = r
    return list(out.values())


def failures(records):
    """Steps whose most recent attempt failed and that never succeeded later."""
    last = {}
    for r in records:
        last[_key(r)] = r
    return [r for r in last.values() if not r.get("ok")]


def pareto(points):
    pts = sorted(points, key=lambda p: (-p["recall"], -p["qps"]))
    front, best = [], -1.0
    for p in pts:
        if p["qps"] > best:
            front.append(p)
            best = p["qps"]
    return sorted(front, key=lambda p: p["recall"])


def qps_at(points, target):
    ok = [p["qps"] for p in points if p["recall"] >= target]
    return max(ok) if ok else None


def label(r):
    s = f"{r['system']}-{r['variant']}"
    if r["kind"] == "search" and r["searcher"] == "native":
        s += " (native search)"
    return s


def fmt(v, nd=1):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", help="output dir (default: next to results)")
    a = ap.parse_args()
    raw = load_jsonl(a.results)
    recs = latest(raw)
    failed = failures(raw)
    exports = {(r["dataset"], r["system"], r["variant"]): r for r in recs if r["kind"] == "export"}
    recs = [r for r in recs if r["kind"] in ("build", "search")]
    out = Path(a.out) if a.out else Path(a.results).parent
    out.mkdir(parents=True, exist_ok=True)
    env_p = Path(a.results).parent / "environment.json"
    env = json.loads(env_p.read_text()) if env_p.exists() else {}

    builds = defaultdict(list)
    searches = defaultdict(list)
    for r in recs:
        (builds if r["kind"] == "build" else searches)[r["dataset"]].append(r)

    md = ["# GPU ANNS single-GPU evaluation", ""]
    if env:
        md += ["**Environment:** " + ", ".join(f"{k}={v}" for k, v in env.items()), ""]
    md += ["Build time `core` = library-reported / in-process timed construction (data already in host "
           "memory; ScaleGANN = partition + Σ shard builds + merge). `e2e` = wall clock of the build "
           "process(es) incl. load and save. Search: cuVS CAGRA kernel on every graph unless noted "
           "`native`; QPS = max over sweep points meeting the recall target.", ""]

    for ds in sorted(set(builds) | set(searches)):
        md += [f"## {ds}", "", "### Build", "",
               "| system | core (s) | e2e (s) | peak GPU mem (GiB) | stages | graph |",
               "|---|---:|---:|---:|---|---|"]
        for r in sorted(builds[ds], key=lambda r: (r["system"], r["variant"])):
            st = ", ".join(f"{k}={fmt(v, 2)}" for k, v in (r.get("stages") or {}).items())
            g = r.get("graph") or exports.get((ds, r["system"], r["variant"]), {}).get("graph") or {}
            gs_file = Path(a.results).parent.parent / ds / f"{r['system']}-{r['variant']}" / "graph_stats.json"
            if not g and gs_file.exists():  # results from before export records existed
                g = json.loads(gs_file.read_text()).get("graph", {})
            gs = f"deg {g.get('degree')}" + (f", avg {g['avg_out_degree']:.1f}" if "avg_out_degree" in g else "")
            if r["system"] == "scalegann":
                gs += f", {r.get('n_shards')} shards, dup {r.get('duplication', 0):.2f}x"
            md.append(f"| {r['system']}-{r['variant']} | {fmt(r.get('build_core_s'), 2)} | "
                      f"{fmt(r.get('build_e2e_s'), 2)} | {fmt(r.get('peak_gpu_mem_gb'), 2)} | {st} | {gs} |")
        md += ["", "### Search (k=10)", "",
               "| graph | searcher | dtype | max recall | " + " | ".join(f"QPS@{t}" for t in TARGETS) + " |",
               "|---|---|---|---:|" + "---:|" * len(TARGETS)]
        for r in sorted(searches[ds], key=lambda r: (r["dtype"], r["system"], r["variant"], r["searcher"])):
            pts = r.get("points", [])
            mr = max((p["recall"] for p in pts), default=None)
            md.append(f"| {r['system']}-{r['variant']} | {r['searcher']} | {r['dtype']} | {fmt(mr, 4)} | "
                      + " | ".join(fmt(qps_at(pts, t), 0) for t in TARGETS) + " |")
        md.append("")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            continue
        for dt in sorted({r["dtype"] for r in searches[ds]}):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            seen = defaultdict(int)
            for r in sorted(searches[ds], key=label):
                if r["dtype"] != dt and not (r["searcher"] == "native" and dt == "f16"):
                    continue
                if r["searcher"] == "native" and dt == "native" and "f16" in {x["dtype"] for x in searches[ds]}:
                    continue  # Jasper native search is fp16: show it on the f16 plot only
                front = pareto(r.get("points", []))
                if not front:
                    continue
                ls = STYLES[seen[r["system"]] % len(STYLES)]
                seen[r["system"]] += 1
                ax.plot([p["recall"] for p in front], [p["qps"] for p in front], ls, marker="o", ms=3,
                        color=COLORS.get(r["system"]), label=label(r))
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("QPS")
            ax.set_yscale("log")
            ax.set_xlim(left=0.8)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{ds} — CAGRA search kernel, {dt} vectors")
            ax.legend(fontsize=8)
            fig.tight_layout()
            png = out / f"{ds}_{dt}_recall_qps.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
            md += [f"![{ds} {dt}]({png.name})", ""]

    if failed:
        md += ["## Failed steps", "", "| dataset | step | system | log | last lines |", "|---|---|---|---|---|"]
        for r in failed:
            tail = (r.get("error_tail") or "").strip().splitlines()[-3:]
            md.append(f"| {r['dataset']} | {r['kind']} {r.get('searcher', '')} {r.get('dtype', '')} | "
                      f"{r['system']}-{r.get('variant', '')} | `{r.get('log', '')}` | "
                      + "<br>".join(t.replace("|", "\\|") for t in tail) + " |")
        md.append("")

    (out / "report.md").write_text("\n".join(md))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
