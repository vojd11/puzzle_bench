#!/usr/bin/env python3
"""Zebra-puzzle benchmark for LLMs over any OpenAI-compatible /chat/completions API.

Ladder: 3x3 -> 4x4 -> 5x5 -> 6x6 -> 7x7 (houses x properties).
Each puzzle is sent in a FRESH context (no history, no few-shot).
The model must solve it by reasoning only — writing/executing code is forbidden
by the prompt, and replies that look like code are flagged.
A level is passed if the share of fully correct answers >= --pass-ratio;
the ladder stops at the first failed level.

Puzzles are cached in the SQLite DB (bench.db): at each level up to
attempts-1 attempts reuse random puzzles already stored there (deterministically
chosen, so every model still faces the identical puzzle set), the remaining
attempt uses a fresh puzzle that a background thread generates while the level
runs — and that new puzzle joins the pool for future runs.

Examples
--------
  export OPENAI_API_KEY=sk-...
  python bench.py --model gpt-4o-mini --attempts 3
  python bench.py --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
                  --model anthropic/claude-sonnet-4.5 --model openai/gpt-5 --parallel 3
  python bench.py --mock perfect            # no network, tests the pipeline
  python bench.py --dry-run                 # just print a 3x3 prompt
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import db
import zebra

DEFAULT_LEVELS = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]


# --------------------------------------------------------------- API call --
class ApiError(Exception):
    pass


def call_api(base_url: str, api_key: str, model: str, prompt: str, *,
             max_tokens: int, temperature: Optional[float], tokens_param: str,
             timeout: int, retries: int = 4, extra_body: Optional[Dict] = None) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            tokens_param: max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
            usage = payload.get("usage") or {}
            choice = (payload.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):  # some gateways return content parts
                text = "".join(part.get("text", "") for part in text)
            return {"text": text, "latency": time.time() - t0,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "finish_reason": choice.get("finish_reason")}
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (408, 409, 429) or e.code >= 500:
                time.sleep(2 ** attempt + random.random())
                continue
            raise ApiError(last)
        except Exception as e:  # timeouts, connection resets
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt + random.random())
    raise ApiError(last or "unknown error")


def mock_reply(p: Dict, mode: str) -> Dict:
    """Offline stand-in for a model, for testing the harness itself."""
    time.sleep(0.01)
    ans = zebra.solve_puzzle_answer(p)
    if mode == "perfect":
        text = "Reasoning omitted.\n" + ans
    elif mode == "dumb":
        text = "I think the answer is unclear."
    else:  # noisy: gets easy levels right, degrades with size
        ok = random.random() < max(0.0, 1.25 - 0.22 * p["N"])
        if ok:
            text = "Reasoning omitted.\n" + ans
        else:
            obj = json.loads(ans)
            first = next(iter(obj["grid"]))
            obj["grid"][first] = list(reversed(obj["grid"][first]))
            text = "Reasoning omitted.\n" + json.dumps(obj)
    return {"text": text, "latency": 0.01, "prompt_tokens": None,
            "completion_tokens": None, "reasoning_tokens": None, "finish_reason": "stop"}


# ------------------------------------------------------------- benchmark ---
# ---------------------------------------------------------- puzzle sources --
def obtain_puzzle(N: int, M: int, difficulty: str, num: int) -> Dict:
    """Puzzle for a seed, loaded from the DB cache when it's already there.

    build_puzzle runs the full generator + uniqueness solver, so repeat runs
    (and every model after the first in a paired run) reuse the stored copy;
    only a cache miss pays for generation, and the build is stored for next time.
    """
    seed = zebra.seed_code(N, M, difficulty, num)
    p = db.get_puzzle(seed)
    if p is None:
        p = zebra.build_puzzle(N, M, difficulty, num)
        db.save_puzzle(p)
    return p


def plan_puzzles(levels, attempts: int, difficulty: str, base_seed: str) -> Dict:
    """Decide where each attempt's puzzle comes from — once per run.

    Up to attempts-1 slots per level take random puzzles from the stored DB
    pool (if there are that many); the remaining slots get freshly generated
    puzzles. The picks are derived from the base seed only (never from the
    model name), so every model in the run gets the identical puzzle at each
    (level, attempt) — the paired comparison the --seed help promises.

    Recipes: ("seed", code) reuses a stored puzzle; ("gen", num) loads or
    builds the puzzle for that RNG number. Generated numbers skip seeds
    already in the pool, so a rerun keeps adding new puzzles instead of
    repeating stored ones, and no puzzle appears twice within a level.
    """
    plan = {}
    for (N, M) in levels:
        rng = random.Random(f"{base_seed}|{difficulty}|{N}x{M}")
        pool = db.list_puzzle_seeds(N, M, difficulty)
        k = min(len(pool), attempts - 1)
        picks = rng.sample(pool, k) if k else []
        taken = set(pool)
        gen_nums = []
        for _ in range(attempts - k):
            while True:  # skip stored seeds: the new puzzle must grow the pool
                n = rng.getrandbits(32)
                if zebra.seed_code(N, M, difficulty, n) not in taken:
                    gen_nums.append(n)
                    break
        plan[(N, M)] = ([("seed", s) for s in picks]
                        + [("gen", n) for n in gen_nums])
    return plan


def puzzle_from_recipe(N: int, M: int, difficulty: str, recipe) -> Dict:
    """Materialize a plan recipe into a puzzle dict."""
    kind, v = recipe
    if kind == "seed":
        p = db.get_puzzle(v)
        if p is not None:
            return p
        parsed = zebra.parse_seed(v)  # row vanished (db cleared?) — rebuild
        return zebra.build_puzzle(parsed["N"], parsed["M"], parsed["difficulty"], parsed["num"])
    return obtain_puzzle(N, M, difficulty, v)


def start_prefetch(difficulty: str, N: int, M: int, recipes):
    """Build the level's not-yet-stored puzzles in a background thread.

    Generation is pure CPU while attempts mostly wait on the API, so the
    planned ("gen", …) puzzles are built on the side — that's the "meanwhile
    start generating new" half of the pool strategy. Newest-needed first: the
    main thread builds the earliest slot itself right away anyway.
    """
    class _Prefetch:
        def __init__(self):
            self._stop = threading.Event()
            self._started = False
            self._thread = threading.Thread(target=self._work, daemon=True)

        def _work(self):
            for recipe in reversed(recipes):
                if self._stop.is_set():
                    return
                if recipe[0] == "gen":
                    try:
                        obtain_puzzle(N, M, difficulty, recipe[1])
                    except Exception:
                        return  # main thread rebuilds and surfaces the error

        def start(self):
            self._thread.start()
            self._started = True

        def stop(self):
            self._stop.set()
            if self._started:
                self._thread.join()

    pf = _Prefetch()
    if db.is_enabled() and any(r[0] == "gen" for r in recipes):
        pf.start()
    return pf


def attempt_from_recipe(args, model: str, N: int, M: int, recipe, idx: int) -> Dict:
    """One graded attempt against the puzzle a plan recipe points to."""
    p = puzzle_from_recipe(N, M, args.difficulty, recipe)
    return run_attempt(args, model, p, idx)


def run_attempt(args, model: str, p: Dict, idx: int) -> Dict:
    prompt = zebra.render_prompt(p)
    rec = {"model": model, "level": f"{p['N']}x{p['M']}", "N": p["N"], "M": p["M"], "attempt": idx,
           "seed": p["seed"], "clues": len(p["clue_texts"]),
           "question": p["question"]["text"], "expected_answer": p["question"]["answer"]}
    try:
        if args.mock:
            r = mock_reply(p, args.mock)
        else:
            r = call_api(args.base_url, args.api_key, model, prompt,
                         max_tokens=args.max_tokens, temperature=args.temperature,
                         tokens_param=args.tokens_param, timeout=args.timeout)
        g = zebra.grade(p, r["text"])
        rec.update(g)
        rec.update({k: r.get(k) for k in
                    ("latency", "prompt_tokens", "completion_tokens", "reasoning_tokens", "finish_reason")})
        rec["code_flag"] = zebra.looks_like_code(r["text"])
        # strict mode: a reply that writes code to solve the puzzle is disqualified,
        # even if the grid it produced happens to be right (it broke the no-code rule).
        # A disqualification is a full forfeit — zero credit on every metric — so a
        # code-cheater can't inflate cell accuracy or win the leaderboard tiebreaker.
        if getattr(args, "strict_no_code", False) and rec["code_flag"]:
            rec.update({"correct": False, "grid_correct": False, "answer_correct": False,
                        "cells_correct": 0, "cell_acc": 0.0, "disqualified": True})
        rec["reply_chars"] = len(r["text"])
        if args.save_replies:
            rec["reply"] = r["text"]
        rec["error"] = None
    except Exception as e:
        rec.update({"parsed": False, "correct": False, "cell_acc": 0.0,
                    "grid_correct": False, "answer_correct": False,
                    "code_flag": False, "error": f"{type(e).__name__}: {e}"})
    return rec


def run_model(args, model: str, out, plan: Dict) -> Dict:
    print(f"\n=== {model} ===", flush=True)
    summary = {"model": model, "max_level": None, "levels": []}
    # The puzzle plan is computed once per run (see plan_puzzles), from the base
    # seed ONLY (never the model name), so every model gets the identical puzzle
    # at each (level, attempt) — a paired comparison, as the --seed help and the
    # UI both promise.
    for (N, M) in args.levels:
        recipes = plan[(N, M)]
        results: List[Dict] = []
        pf = start_prefetch(args.difficulty, N, M, recipes)
        try:
            if args.parallel > 1 and not args.mock:
                with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
                    futs = [ex.submit(attempt_from_recipe, args, model, N, M, recipe, i)
                            for i, recipe in enumerate(recipes)]
                    results = [f.result() for f in futs]
            else:
                for i, recipe in enumerate(recipes):
                    results.append(attempt_from_recipe(args, model, N, M, recipe, i))
                    time.sleep(args.sleep)
        finally:
            pf.stop()
        for r in results:
            out.write(json.dumps(r) + "\n")
            db.save_result(r, run_id=getattr(args, "run_id", None))
        out.flush()
        ok = sum(1 for r in results if r["correct"])
        acc = sum(r.get("cell_acc") or 0 for r in results) / len(results)
        ratio = ok / len(results)
        errs = sum(1 for r in results if r.get("error"))
        flags = sum(1 for r in results if r.get("code_flag"))
        dq = sum(1 for r in results if r.get("disqualified"))
        code_note = (f" | {dq} disqualified (code)" if dq
                     else f" | {flags} code-flagged" if flags else "")
        print(f"  {N}x{M}: {ok}/{len(results)} solved | cell acc {acc:.0%}"
              + (f" | {errs} errors" if errs else "") + code_note, flush=True)
        summary["levels"].append({"level": f"{N}x{M}", "solved": ok, "attempts": len(results),
                                  "pass_ratio": ratio, "cell_acc": acc})
        if ratio >= args.pass_ratio:
            summary["max_level"] = f"{N}x{M}"
        else:
            print(f"  -> stopped at {N}x{M}", flush=True)
            break
    print(f"  highest level passed: {summary['max_level'] or 'none'}", flush=True)
    db.finish_run(getattr(args, "run_id", None), summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model", action="append", default=[], help="repeatable")
    ap.add_argument("--levels", default="", help='e.g. "3x3,4x4,5x5" (default 3x3..7x7)')
    ap.add_argument("--attempts", type=int, default=3, help="puzzles per level")
    ap.add_argument("--pass-ratio", type=float, default=0.67)
    ap.add_argument("--strict-no-code", action="store_true",
                    help="disqualify (mark incorrect) any reply that contains code, even if the grid is right")
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--tokens-param", default="max_tokens",
                    help="use max_completion_tokens for newer OpenAI reasoning models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-temperature", action="store_true", help="omit temperature (reasoning models)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--seed", default="zebra-bench-v1", help="base seed; same seeds => same puzzles for every model")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--db", default=db.DEFAULT_DB,
                    help="SQLite file for puzzles/results/providers ('' disables)")
    ap.add_argument("--save-replies", action="store_true")
    ap.add_argument("--mock", choices=["perfect", "noisy", "dumb"], help="offline self-test, no API calls")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    args = ap.parse_args()

    if args.no_temperature:
        args.temperature = None
    args.levels = ([tuple(int(x) for x in lv.lower().split("x")) for lv in args.levels.split(",") if lv]
                   or DEFAULT_LEVELS)

    if args.dry_run:
        N, M = args.levels[0]
        p = zebra.build_puzzle(N, M, args.difficulty, 0xDEADBEEF)
        print(zebra.render_prompt(p))
        print("\n--- expected ---")
        print(json.dumps(p["solution_grid"], indent=1))
        print(p["question"]["text"], "->", p["question"]["answer"])
        return

    models = args.model or (["mock"] if args.mock else [])
    if not models:
        sys.exit("give at least one --model")
    args.api_key = "" if args.mock else os.environ.get(args.api_key_env, "")
    if not args.mock and not args.api_key:
        sys.exit(f"missing API key in ${args.api_key_env}")

    if args.db:
        db.init(args.db)
        args.run_id = db.start_run("cli", {
            "base_url": args.base_url, "models": models, "levels":
            [f"{n}x{m}" for (n, m) in args.levels], "attempts": args.attempts,
            "pass_ratio": args.pass_ratio, "difficulty": args.difficulty,
            "seed": args.seed, "strict_no_code": args.strict_no_code,
            "mock": args.mock})
        if not args.mock:
            pid = db.upsert_provider(args.base_url, args.api_key,
                                     tokens_param=args.tokens_param)
            db.remember_models(pid, models)

    summaries = []
    plan = plan_puzzles(args.levels, args.attempts, args.difficulty, args.seed)
    with open(args.out, "a", encoding="utf-8") as out:
        for model in models:
            summaries.append(run_model(args, model, out, plan))

    print("\n=== summary ===")
    for s in summaries:
        print(f"{s['model']}: highest passed = {s['max_level'] or 'none'}")
    print(f"\nresults -> {args.out}" + (f" and {args.db}" if args.db else ""))
    print(f"now run:  python plots.py {args.out}")


if __name__ == "__main__":
    main()
