#!/usr/bin/env python3
"""Zebra-puzzle benchmark for LLMs over any OpenAI-compatible /chat/completions API.

Ladder: 3x3 -> 4x4 -> 5x5 -> 6x6 -> 7x7 (houses x properties).
Each puzzle is sent in a FRESH context (no history, no few-shot).
The model must solve it by reasoning only — writing/executing code is forbidden
by the prompt, and replies that look like code are flagged.
A level is passed if the share of fully correct answers >= --pass-ratio;
the ladder stops at the first failed level.

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
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

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
def run_attempt(args, model: str, N: int, M: int, num: int, idx: int) -> Dict:
    p = zebra.build_puzzle(N, M, args.difficulty, num)
    prompt = zebra.render_prompt(p)
    rec = {"model": model, "level": f"{N}x{M}", "N": N, "M": M, "attempt": idx,
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


def run_model(args, model: str, out) -> Dict:
    print(f"\n=== {model} ===", flush=True)
    summary = {"model": model, "max_level": None, "levels": []}
    # Seed from the base seed ONLY (not the model name) so every model gets the
    # identical puzzle at each (level, attempt) — a paired comparison, as the
    # --seed help and the UI both promise. Same seed => same puzzles for all models.
    rng = random.Random(args.seed)
    for (N, M) in args.levels:
        nums = [rng.getrandbits(32) for _ in range(args.attempts)]
        results: List[Dict] = []
        if args.parallel > 1 and not args.mock:
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = [ex.submit(run_attempt, args, model, N, M, num, i)
                        for i, num in enumerate(nums)]
                results = [f.result() for f in futs]
        else:
            for i, num in enumerate(nums):
                results.append(run_attempt(args, model, N, M, num, i))
                time.sleep(args.sleep)
        for r in results:
            out.write(json.dumps(r) + "\n")
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

    summaries = []
    with open(args.out, "a", encoding="utf-8") as out:
        for model in models:
            summaries.append(run_model(args, model, out))

    print("\n=== summary ===")
    for s in summaries:
        print(f"{s['model']}: highest passed = {s['max_level'] or 'none'}")
    print(f"\nresults -> {args.out}\nnow run:  python plots.py {args.out}")


if __name__ == "__main__":
    main()
