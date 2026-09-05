#!/usr/bin/env python3
"""Generate a SYNTHETIC sample results file for previewing the dashboard.

These numbers are made up (a plausible-looking curve per model), NOT a real
benchmark run. Use it only to see what `report.py` / `plots.py` produce before
you have real results.

    python make_sample.py            # -> sample_results.jsonl
    python report.py sample_results.jsonl --open

Do not commit real claims off this file.
"""
from __future__ import annotations

import json
import random

LEVELS = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]

# name -> (skill, base latency at 3x3 s, latency growth, tokens at 3x3, token growth)
MODELS = {
    "openai/gpt-5":              (0.93, 6.0, 1.9, 900, 1.7),
    "anthropic/claude-sonnet-4.5": (0.88, 5.2, 1.8, 1100, 1.75),
    "google/gemini-2.5-pro":     (0.80, 7.5, 2.0, 1400, 1.8),
    "openai/gpt-4o-mini":        (0.55, 2.1, 1.5, 500, 1.5),
    "meta/llama-3.1-70b":        (0.42, 3.0, 1.6, 700, 1.55),
    "qwen/qwen2.5-7b":           (0.28, 1.4, 1.4, 400, 1.45),
}

CATS = ["Color", "Nationality", "Drink", "Pet", "Sport", "Food", "Job"]
VALS = {c: [f"{c[0]}{i}" for i in range(7)] for c in CATS}


def solve_prob(skill: float, N: int) -> float:
    """Chance a model fully solves an NxN puzzle given a skill in [0,1]."""
    difficulty = (N - 3) / 4.0  # 0 at 3x3, 1 at 7x7
    return max(0.0, min(1.0, skill - difficulty * (1.35 - skill)))


def main() -> None:
    rng = random.Random(20260725)
    attempts = 5
    out = open("sample_results.jsonl", "w", encoding="utf-8")
    for model, (skill, lat0, latg, tok0, tokg) in MODELS.items():
        step = 0
        for (N, M) in LEVELS:
            p_solve = solve_prob(skill, N)
            solved_here = 0
            for a in range(attempts):
                full = rng.random() < p_solve
                # cell accuracy: correlated with solving, degrades with size
                if full:
                    cell = 1.0
                else:
                    base = p_solve * 0.9 + 0.05
                    cell = max(0.0, min(0.99, rng.gauss(base, 0.12)))
                cells_total = N * M
                cells_correct = round(cell * cells_total)
                cell = cells_correct / cells_total
                lat = max(0.3, rng.gauss(lat0 * latg ** step, lat0 * 0.15))
                ctok = int(max(50, rng.gauss(tok0 * tokg ** step, tok0 * 0.2)))
                ptok = int(cells_total * 40 + 300)
                parsed = rng.random() < min(0.999, 0.80 + skill * 0.2)
                if not parsed:
                    full = False
                    cell = 0.0
                    cells_correct = 0
                answer_correct = full or (rng.random() < cell * 0.7)
                rec = {
                    "model": model, "level": f"{N}x{M}", "N": N, "M": M, "attempt": a,
                    "seed": f"{N}h{M}p-m-{rng.getrandbits(32):08x}",
                    "clues": int(cells_total * rng.uniform(0.7, 1.1)),
                    "question": f"Which {rng.choice(CATS[:M])} belongs to a given house?",
                    "expected_answer": rng.choice(VALS[CATS[0]]),
                    "parsed": parsed,
                    "cells_correct": cells_correct, "cells_total": cells_total,
                    "cell_acc": round(cell, 4),
                    "grid_correct": full, "answer_correct": bool(answer_correct),
                    "correct": full,
                    "latency": round(lat, 2),
                    "prompt_tokens": ptok, "completion_tokens": ctok,
                    "reasoning_tokens": int(ctok * skill * 0.4) if skill > 0.6 else None,
                    "finish_reason": "stop",
                    "code_flag": (not parsed) and rng.random() < 0.15,
                    "reply_chars": ctok * 4, "error": None,
                }
                out.write(json.dumps(rec) + "\n")
                if full:
                    solved_here += 1
            step += 1
            # stop the ladder once a model clearly fails a level (like the real bench)
            if solved_here / attempts < 0.4:
                break
    out.close()
    print("wrote sample_results.jsonl (SYNTHETIC preview data)")


if __name__ == "__main__":
    main()
