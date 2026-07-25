#!/usr/bin/env python3
"""Charts for zebra-bench results.

  python plots.py results.jsonl [--outdir charts]

Produces:
  01_pass_rate.png      solve rate per level, per model
  02_cell_accuracy.png  average share of correct grid cells per level
  03_max_level.png      highest level passed per model
  04_cost.png           latency and completion tokens per level
  summary.csv           the same numbers in a table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVEL_KEY = lambda lv: tuple(int(x) for x in lv.split("x"))  # noqa: E731


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][r["level"]].append(r)
    return agg


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def bar_grouped(ax, models, levels, values, ylabel, title, pct=False):
    n = max(len(models), 1)
    width = 0.8 / n
    for i, m in enumerate(models):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(levels))]
        ax.bar(xs, [values[m].get(lv, 0) for lv in levels], width=width, label=m)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlabel("level (houses x properties)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if pct:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results.jsonl")
    ap.add_argument("--outdir", default="charts")
    ap.add_argument("--pass-ratio", type=float, default=0.67)
    a = ap.parse_args()

    rows = load(a.results)
    if not rows:
        raise SystemExit("no results")
    os.makedirs(a.outdir, exist_ok=True)
    agg = aggregate(rows)
    models = sorted(agg)
    levels = sorted({r["level"] for r in rows}, key=LEVEL_KEY)

    pass_rate, cell_acc, latency, tokens = ({m: {} for m in models} for _ in range(4))
    for m in models:
        for lv, rs in agg[m].items():
            pass_rate[m][lv] = sum(1 for r in rs if r.get("correct")) / len(rs)
            cell_acc[m][lv] = mean([r.get("cell_acc") for r in rs])
            latency[m][lv] = mean([r.get("latency") for r in rs])
            tokens[m][lv] = mean([r.get("completion_tokens") for r in rs])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_grouped(ax, models, levels, pass_rate, "fully correct", "Zebra benchmark — solve rate by level", pct=True)
    ax.axhline(a.pass_ratio, ls="--", c="grey", lw=1)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/01_pass_rate.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in models:
        ax.plot(levels, [cell_acc[m].get(lv, 0) for lv in levels], marker="o", label=m)
    ax.set_ylim(0, 1.05); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("level"); ax.set_ylabel("correct grid cells")
    ax.set_title("Partial credit — average cell accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/02_cell_accuracy.png", dpi=150); plt.close(fig)

    reached = {}
    for m in models:
        best = "none"
        for lv in levels:
            if pass_rate[m].get(lv, 0) >= a.pass_ratio:
                best = lv
            else:
                break
        reached[m] = levels.index(best) + 1 if best != "none" else 0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(models, [reached[m] for m in models], color="#6ea8fe")
    ax.set_xticks(range(len(levels) + 1)); ax.set_xticklabels(["none"] + levels)
    ax.set_xlabel("highest level passed"); ax.set_title("How far up the ladder each model got")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/03_max_level.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models:
        axes[0].plot(levels, [latency[m].get(lv, 0) for lv in levels], marker="o", label=m)
        axes[1].plot(levels, [tokens[m].get(lv, 0) for lv in levels], marker="o", label=m)
    axes[0].set_title("Latency per puzzle (s)"); axes[1].set_title("Completion tokens per puzzle")
    for ax in axes:
        ax.set_xlabel("level"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.outdir}/04_cost.png", dpi=150); plt.close(fig)

    with open(f"{a.outdir}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "level", "attempts", "solved", "pass_rate", "cell_acc",
                    "answer_only_acc", "parse_fail", "code_flagged", "errors",
                    "avg_latency_s", "avg_completion_tokens"])
        for m in models:
            for lv in levels:
                rs = agg[m].get(lv, [])
                if not rs:
                    continue
                w.writerow([m, lv, len(rs), sum(1 for r in rs if r.get("correct")),
                            f"{pass_rate[m][lv]:.3f}", f"{cell_acc[m][lv]:.3f}",
                            f"{mean([1.0 if r.get('answer_correct') else 0.0 for r in rs]):.3f}",
                            sum(1 for r in rs if not r.get("parsed")),
                            sum(1 for r in rs if r.get("code_flag")),
                            sum(1 for r in rs if r.get("error")),
                            f"{latency[m][lv]:.1f}", f"{tokens[m][lv]:.0f}"])
    print(f"charts + summary.csv -> {a.outdir}/")


if __name__ == "__main__":
    main()
