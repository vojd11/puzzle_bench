#!/usr/bin/env python3
"""Local web UI to run the zebra benchmark: paste an API key, pick models, hit Run.

    python serve.py                       # open http://127.0.0.1:8000
    python serve.py --port 8080 --results results.jsonl

Pure stdlib — nothing to install. The page is served from, and the LLM calls
are made by, THIS local process, so:

  * your API key is POSTed only to 127.0.0.1 and used for the outbound calls;
    to spare you re-pasting it, the provider (base URL, key, model names) is
    kept in the local SQLite file (bench.db, git-ignored) and offered back as
    a "saved" preset on the next start;
  * there is no browser CORS problem (the browser talks to this server, the
    server talks to the provider).

Puzzles, results and runs go to the same DB (results.jsonl is still written
as a sibling, so plots.py/report.py keep working). Progress streams live over
Server-Sent Events; the embedded dashboard (report.py) re-renders after every
level. Binds to 127.0.0.1 by default; pass --host to change it, but note that
there is no auth — only expose it on a network you trust (the server prints a
warning when bound off loopback).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs

import bench
import db
import report
import zebra

RESULTS_FILE = "results.jsonl"
WRITE_LOCK = threading.Lock()
RUNS: dict = {}  # run_id -> {"events": [...], "cond": Condition, "stop": Event, "done": bool}
LAST_PASS_RATIO = 0.67  # keeps the embedded dashboard consistent with the last run

DEFAULT_LEVELS = "3x3,4x4,5x5,6x6,7x7"
KEEP_FINISHED_RUNS = 12  # cap retained finished runs so RUNS doesn't grow unbounded


def _prune_runs():
    """Drop old finished runs, keeping the most recent few (ids are time-ordered)."""
    finished = sorted(rid for rid, r in RUNS.items() if r.get("done"))
    for rid in finished[:-KEEP_FINISHED_RUNS] if len(finished) > KEEP_FINISHED_RUNS else []:
        RUNS.pop(rid, None)

PRESETS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "o4-mini"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["anthropic/claude-sonnet-4.5", "openai/gpt-5",
                   "google/gemini-2.5-pro", "meta-llama/llama-3.1-70b-instruct"],
    },
    "local": {
        "label": "Local (Ollama / vLLM)",
        "base_url": "http://localhost:11434/v1",
        "models": ["llama3.1", "qwen2.5"],
    },
    "mock": {"label": "Mock — no key, offline self-test",
             "base_url": "", "models": ["mock-a", "mock-b"]},
}


# ------------------------------------------------------------- run engine ---
MAX_HOUSES = min(len(c["values"]) for c in zebra.DEFAULT_CATS)  # values available per property
MAX_PROPS = len(zebra.DEFAULT_CATS)                             # properties available


def parse_levels(s: str):
    """Parse 'NxM,NxM,…' into [(N, M), …], with clear errors and size bounds."""
    out = []
    for tok in s.replace(" ", "").lower().split(","):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)x(\d+)", tok)
        if not m:
            raise ValueError(f"bad level '{tok}' — use NxM like 4x4")
        n, p = int(m.group(1)), int(m.group(2))
        if not (2 <= n <= MAX_HOUSES) or not (2 <= p <= MAX_PROPS):
            raise ValueError(
                f"level '{tok}' out of range — houses 2..{MAX_HOUSES}, properties 2..{MAX_PROPS}")
        out.append((n, p))
    if not out:
        raise ValueError("no levels given")
    return out


def _write(rec: dict, run_id=None):
    with WRITE_LOCK:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    db.save_result(rec, run_id=run_id)


def _read_results() -> list:
    if db.is_enabled():
        return db.load_results()
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


# ------------------------------------------------------------ manual mode ---
# Workflow for models with no API (web chats, a friend, paper): generate a
# puzzle here, copy the prompt into the model by hand, paste its reply back.
# The graded attempt lands in the same results file (flagged "manual": true),
# and every record keeps the puzzle seed so you can always see — and replay —
# exactly which puzzle was sent.

DIFFICULTIES = ("easy", "medium", "hard")

# Built puzzles are deterministic per seed, and building one runs the full
# generator + uniqueness solver, so cache them: grading/saving a manual reply
# must not regenerate the puzzle it was answered against. The DB is the
# persistent tier of that cache (survives restarts); the dict is the hot tier.
_PUZZLE_CACHE: dict = {}
_PUZZLE_CACHE_MAX = 64
_PUZZLE_CACHE_LOCK = threading.Lock()


def _cached_puzzle(seed: str) -> dict:
    parsed = zebra.parse_seed(seed)
    if not parsed:
        raise ValueError("bad seed code — expected something like 4h4p-m-1a2b3c4d")
    with _PUZZLE_CACHE_LOCK:
        p = _PUZZLE_CACHE.get(seed)
    if p is None:
        p = db.get_puzzle(seed)
        if p is None:
            p = zebra.build_puzzle(parsed["N"], parsed["M"], parsed["difficulty"], parsed["num"])
            db.save_puzzle(p)
        with _PUZZLE_CACHE_LOCK:
            if len(_PUZZLE_CACHE) >= _PUZZLE_CACHE_MAX:
                _PUZZLE_CACHE.pop(next(iter(_PUZZLE_CACHE)))
            _PUZZLE_CACHE[seed] = p
    return p


def manual_puzzle(cfg: dict) -> dict:
    seed = (cfg.get("seed") or "").strip()
    if seed:
        p = _cached_puzzle(seed)  # replay: DB-first via the cache, build only on a miss
        src = "replayed"
    else:
        N, M = parse_levels(cfg.get("level") or "4x4")[0]
        difficulty = cfg.get("difficulty") or "medium"
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {', '.join(DIFFICULTIES)}")
        # DB first: serve a stored puzzle of this level/difficulty that manual
        # mode hasn't sent yet; only generate a brand-new one when the pool has
        # nothing unused (or storage is off). A previously sent puzzle would
        # just duplicate an attempt, so "used" disqualifies it here.
        unused = [s for s in db.list_puzzle_seeds(N, M, difficulty)
                  if s not in db.manual_used_seeds()]
        if unused:
            p = _cached_puzzle(random.choice(unused))
            src = "from pool"
        else:
            p = zebra.build_puzzle(N, M, difficulty, random.getrandbits(32))
            db.save_puzzle(p)
            src = "newly generated"
    bench.log(f"manual puzzle {src}: {p['seed']} ({p['N']}x{p['M']} {p['difficulty']}, "
              f"{len(p['clue_texts'])} clues)")
    return {
        "seed": p["seed"],
        "level": f"{p['N']}x{p['M']}",
        "difficulty": p["difficulty"],
        "clues": len(p["clue_texts"]),
        "prompt": zebra.render_prompt(p),
        "question": p["question"]["text"],
        "expected_grid": p["solution_grid"],
        "expected_answer": p["question"]["answer"],
    }


def manual_submit(cfg: dict) -> dict:
    # save=False grades only (preview verdict) without touching the results file
    save = cfg.get("save", True)
    if not isinstance(save, bool):
        save = bool(save)
    model = (cfg.get("model") or "").strip()
    if not model:
        raise ValueError("model name is required")
    seed = (cfg.get("seed") or "").strip()
    if not seed:
        raise ValueError("bad seed code — expected something like 4h4p-m-1a2b3c4d")
    reply = cfg.get("reply") or ""
    if not reply.strip():
        raise ValueError("paste the model's reply first")
    p = _cached_puzzle(seed)
    lvl = f"{p['N']}x{p['M']}"

    # attempt index = how many records this model already has at this level
    if db.is_enabled():
        attempt = db.count_attempts(model, lvl)
    else:
        attempt = sum(1 for r in _read_results()
                      if r.get("model") == model and r.get("level") == lvl)
    rec = {"model": model, "level": lvl, "N": p["N"], "M": p["M"], "attempt": attempt,
           "seed": p["seed"], "clues": len(p["clue_texts"]),
           "question": p["question"]["text"], "expected_answer": p["question"]["answer"],
           "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "manual": True}
    rec.update(zebra.grade(p, reply))
    obj = zebra.extract_json(reply)
    rec["given_answer"] = obj.get("answer") if isinstance(obj, dict) else None
    rec["code_flag"] = zebra.looks_like_code(reply)
    if cfg.get("strict_no_code") and rec["code_flag"]:
        rec.update({"correct": False, "grid_correct": False, "answer_correct": False,
                    "cells_correct": 0, "cell_acc": 0.0, "disqualified": True})
    rec.update({k: None for k in
                ("latency", "prompt_tokens", "completion_tokens", "reasoning_tokens")})
    rec["finish_reason"] = "manual"
    rec["reply_chars"] = len(reply)
    rec["reply"] = reply
    rec["error"] = None
    if save:
        _write(rec)
    verdict = ("disqualified" if rec.get("disqualified")
               else "solved" if rec.get("correct") else "failed")
    bench.log(f"manual result {model} @ {lvl} seed {p['seed']}: {verdict} | "
              f"cells {(rec.get('cell_acc') or 0):.0%}"
              + (" | saved" if save else " | graded only"))
    return {"record": {k: rec.get(k) for k in
                       ("model", "level", "seed", "ts", "attempt", "correct", "cell_acc",
                        "parsed", "code_flag", "disqualified", "answer_correct",
                        "grid_correct", "cells_correct", "cells_total", "given_answer")},
            "saved": save,
            "expected_grid": p["solution_grid"],
            "expected_answer": p["question"]["answer"]}


def manual_history(limit: int = 60) -> list:
    if db.is_enabled():
        return db.manual_history(limit)
    rows = [r for r in _read_results() if r.get("manual")]
    rows.sort(key=lambda r: r.get("ts") or "")
    out = []
    for r in rows[-limit:]:
        out.append({k: r.get(k) for k in
                    ("ts", "model", "level", "seed", "clues", "correct", "cell_acc",
                     "parsed", "code_flag", "disqualified", "given_answer",
                     "expected_answer")})
        out[-1]["answer_correct"] = r.get("answer_correct")
    return out[::-1]


def _tally(rec: dict) -> dict:
    return {
        "attempt": rec.get("attempt"),
        "correct": bool(rec.get("correct")),
        "cell_acc": rec.get("cell_acc") or 0.0,
        "parsed": bool(rec.get("parsed")),
        "code_flag": bool(rec.get("code_flag")),
        "disqualified": bool(rec.get("disqualified")),
        "error": rec.get("error"),
        "latency": rec.get("latency"),
        "seed": rec.get("seed"),
    }


def do_run(run_id: str, cfg: dict):
    run = RUNS[run_id]
    stop = run["stop"]

    def emit(ev: str, **kw):
        # Append to a shared event log instead of handing events to a single
        # queue consumer: any SSE connection — including one that reconnects —
        # replays from the log, so no event is lost to a queue race.
        with run["cond"]:
            run["events"].append({"ev": ev, **kw})
            run["cond"].notify_all()

    try:
        args = SimpleNamespace(
            mock=cfg.get("mock") or None,
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            max_tokens=cfg["max_tokens"],
            temperature=(None if cfg.get("no_temperature") else cfg["temperature"]),
            tokens_param=cfg["tokens_param"],
            timeout=cfg["timeout"],
            difficulty=cfg["difficulty"],
            strict_no_code=cfg.get("strict_no_code", False),
            save_replies=False,
        )
        levels = parse_levels(cfg["levels"])
        attempts = int(cfg["attempts"])
        pass_ratio = float(cfg["pass_ratio"])
        global LAST_PASS_RATIO
        LAST_PASS_RATIO = pass_ratio
        parallel = max(1, int(cfg["parallel"]))
        seed = str(cfg["seed"])
        models = cfg["models"]
        # puzzle plan computed once per run from the base seed, so every model
        # faces the identical puzzle set (paired comparison)
        plan = bench.plan_puzzles(levels, attempts, args.difficulty, seed)

        # provenance row for this run (the api_key is filtered out by start_run)
        run_id = db.start_run("ui", cfg)

        if cfg.get("fresh"):
            with WRITE_LOCK:
                open(RESULTS_FILE, "w", encoding="utf-8").close()
            db.clear_results()

        emit("start", models=models,
             levels=[f"{n}x{m}" for n, m in levels], attempts=attempts,
             pass_ratio=pass_ratio, mock=bool(args.mock))
        bench.log(f"run {run_id}: models {', '.join(models)} | levels "
                  f"{[f'{n}x{m}' for n, m in levels]} | {attempts} attempts/level | "
                  f"{args.difficulty}"
                  + (" | MOCK" if args.mock else "")
                  + (f" | parallel {parallel}" if parallel > 1 else ""))

        for model in models:
            if stop.is_set():
                break
            emit("model_start", model=model)
            max_level = None
            for (N, M) in levels:
                if stop.is_set():
                    break
                lvl = f"{N}x{M}"
                recipes = plan[(N, M)]
                results = []
                pf = bench.start_prefetch(args.difficulty, N, M, recipes)
                try:
                    if parallel > 1 and not args.mock:
                        with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
                            futs = {ex.submit(bench.attempt_from_recipe,
                                              args, model, N, M, recipe, i): i
                                    for i, recipe in enumerate(recipes)}
                            for fut in cf.as_completed(futs):
                                rec = fut.result()
                                results.append(rec)
                                _write(rec, run_id)
                                emit("attempt", model=model, level=lvl, **_tally(rec))
                    else:
                        for i, recipe in enumerate(recipes):
                            if stop.is_set():
                                break
                            rec = bench.attempt_from_recipe(args, model, N, M, recipe, i)
                            results.append(rec)
                            _write(rec, run_id)
                            emit("attempt", model=model, level=lvl, **_tally(rec))
                            time.sleep(float(cfg.get("sleep", 0) or 0))
                finally:
                    pf.stop()
                if not results:
                    break
                ok = sum(1 for r in results if r.get("correct"))
                ratio = ok / len(results)
                acc = sum(r.get("cell_acc") or 0 for r in results) / len(results)
                errs = sum(1 for r in results if r.get("error"))
                emit("level_done", model=model, level=lvl, solved=ok,
                     attempts=len(results), pass_ratio=ratio, cell_acc=acc, errors=errs)
                bench.log(f"level {model} @ {lvl}: {ok}/{len(results)} solved, "
                          f"cells {acc:.0%} -> {'pass' if ratio >= pass_ratio else 'fail'}"
                          + (f" | {errs} errors" if errs else ""))
                if ratio >= pass_ratio:
                    max_level = lvl
                else:
                    emit("model_stop", model=model, level=lvl)
                    break
            emit("model_done", model=model, max_level=max_level)
            db.finish_run(run_id, {"model": model, "max_level": max_level})
            bench.log(f"model done: {model} — highest level passed: {max_level or 'none'}")
        emit("done", stopped=stop.is_set())
        bench.log("run finished" + (" (stopped)" if stop.is_set() else ""))
    except Exception as e:  # surface config / network errors to the UI
        bench.log(f"run error: {type(e).__name__}: {e}")
        emit("error", message=f"{type(e).__name__}: {e}")
        emit("done", stopped=True)
    finally:
        with run["cond"]:
            run["done"] = True
            run["cond"].notify_all()


def validate_and_normalize(cfg: dict) -> dict:
    mock = (cfg.get("mock") or "").strip() or None
    models = cfg.get("models") or []
    if isinstance(models, str):
        models = models.replace(",", "\n").split("\n")
    models = [m.strip() for m in models if m and m.strip()]
    # de-dupe, keep order
    seen, uniq = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    if not uniq:
        raise ValueError("choose at least one model")

    api_key = (cfg.get("api_key") or "").strip()
    if not mock and not api_key:
        api_key = os.environ.get(cfg.get("api_key_env") or "OPENAI_API_KEY", "")
    if not mock and not api_key:
        # nothing pasted and nothing in the env — fall back to a key saved for
        # this provider on an earlier run
        saved = db.get_provider((cfg.get("base_url") or "").strip()) or {}
        api_key = (saved.get("api_key") or "").strip()
    if not mock and not api_key:
        raise ValueError("API key is required (or pick a mock mode to test without one)")

    base_url = (cfg.get("base_url") or "https://api.openai.com/v1").strip()
    if not mock and not re.match(r"https?://", base_url):
        raise ValueError("base URL must start with http:// or https://")

    levels = cfg.get("levels") or DEFAULT_LEVELS
    parse_levels(levels)  # validate now so the user gets an immediate, clear error

    def num(key, default, cast):
        """Cast an optional numeric field, treating missing/'' as the default."""
        v = cfg.get(key)
        if v in (None, ""):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")

    attempts = num("attempts", 3, int)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    pass_ratio = num("pass_ratio", 0.67, float)
    if not 0 < pass_ratio <= 1:
        raise ValueError("pass_ratio must be between 0 (exclusive) and 1")
    parallel = num("parallel", 1, int)
    if parallel < 1:
        raise ValueError("parallel must be at least 1")

    return {
        "mock": mock,
        "models": uniq,
        "base_url": base_url,
        "api_key": api_key,
        "levels": levels,
        "attempts": attempts,
        "pass_ratio": pass_ratio,
        "difficulty": cfg.get("difficulty") or "medium",
        "parallel": parallel,
        "max_tokens": num("max_tokens", 16000, int),
        "tokens_param": cfg.get("tokens_param") or "max_tokens",
        "temperature": num("temperature", 0.0, float),
        "no_temperature": bool(cfg.get("no_temperature")),
        "timeout": num("timeout", 600, int),
        "sleep": num("sleep", 0.0, float),
        "seed": str(cfg.get("seed") or "zebra-bench-v1"),
        "fresh": bool(cfg.get("fresh")),
        "strict_no_code": bool(cfg.get("strict_no_code")),
    }


# --------------------------------------------------------------- handler ----
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    # ---- GET ----
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, RUNNER_HTML.encode())
        elif u.path == "/report":
            self._send(200, self._render_report().encode())
        elif u.path == "/api/presets":
            payload = dict(PRESETS)
            for p in (db.list_providers() if db.is_enabled() else []):
                # note: the api key itself is never sent to the browser — only
                # that one exists; the server fills it in at run time
                payload[f"saved:{p['id']}"] = {
                    "label": p["label"], "base_url": p["base_url"],
                    "models": p["models"], "saved": True, "has_key": p["has_key"],
                    "tokens_param": p["tokens_param"],
                }
            self._json(200, payload)
        elif u.path == "/api/events":
            self._sse(parse_qs(u.query).get("id", [""])[0])
        elif u.path == "/api/manual/history":
            self._json(200, manual_history())
        elif u.path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._send(404, b"not found", "text/plain")

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            cfg = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad JSON"})

        if u.path == "/api/run":
            try:
                norm = validate_and_normalize(cfg)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            if not norm["mock"]:
                # remember the provider (base URL + key) and its model names so
                # the next visit offers them back as a saved preset
                preset = (cfg.get("preset") or "").strip()
                label = preset if (preset in PRESETS
                                   and PRESETS[preset].get("base_url") == norm["base_url"]) else None
                pid = db.upsert_provider(norm["base_url"], norm["api_key"], label=label,
                                         tokens_param=norm["tokens_param"])
                db.remember_models(pid, norm["models"])
            _prune_runs()
            run_id = f"run-{int(time.time()*1000)}-{random.randint(1000,9999)}"
            RUNS[run_id] = {"events": [], "cond": threading.Condition(),
                            "stop": threading.Event(), "done": False}
            threading.Thread(target=do_run, args=(run_id, norm), daemon=True).start()
            return self._json(200, {"id": run_id, "models": norm["models"]})

        if u.path == "/api/stop":
            rid = parse_qs(u.query).get("id", [""])[0]
            if rid in RUNS:
                RUNS[rid]["stop"].set()
            return self._json(200, {"ok": True})

        if u.path in ("/api/manual/puzzle", "/api/manual/submit"):
            try:
                if u.path == "/api/manual/puzzle":
                    return self._json(200, manual_puzzle(cfg))
                return self._json(200, manual_submit(cfg))
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})

        self._json(404, {"error": "not found"})

    # ---- helpers ----
    def _render_report(self) -> str:
        try:
            rows = db.load_results() if db.is_enabled() else report.load(RESULTS_FILE)
        except (FileNotFoundError, SystemExit):
            rows = []
        if not rows:
            return ("<!doctype html><meta charset='utf-8'>"
                    "<body style='font-family:system-ui;color:#898781;"
                    "display:grid;place-items:center;height:90vh;margin:0;"
                    "background:#f9f9f7'>"
                    "<p>No results yet — configure a run on the left and press "
                    "<b>Run benchmark</b>.</p></body>")
        payload = report.build_payload(rows, LAST_PASS_RATIO)
        return report.render_html(payload, "Live results")

    def _sse(self, run_id: str):
        run = RUNS.get(run_id)
        if not run:
            return self._json(404, {"error": "unknown run id"})
        # EventSource auto-reconnect sends Last-Event-ID, so a client that
        # dropped (network blip, laptop sleep) resumes where it left off; a
        # fresh connection replays the whole run from the start.
        try:
            idx = int(self.headers.get("Last-Event-ID") or -1)
        except ValueError:
            idx = -1
        events, cond = run["events"], run["cond"]
        self.close_connection = True  # streaming, no Content-Length -> close when done
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                with cond:
                    if idx + 1 >= len(events) and not run["done"]:
                        cond.wait(timeout=15)
                    new = events[idx + 1:]
                    idx = len(events) - 1
                    finished = run["done"] and idx + 1 >= len(events)
                if not new:
                    if finished:
                        break
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                for off, ev in enumerate(new):
                    self.wfile.write(
                        f"id: {idx - len(new) + off + 1}\ndata: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                if finished:
                    break
        except OSError:
            # client went away — the run keeps going (results land in the
            # results file); it can reconnect and resume via Last-Event-ID
            pass


# ------------------------------------------------------------------ page ----
RUNNER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Zebra Bench — runner</title>
<style>
:root{
  color-scheme: light dark;
  --plane:#f9f9f7; --surface:#fcfcfb; --card:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --accent:#2a78d6; --good:#0ca30c; --bad:#d03b3b; --warn:#c98500;
}
@media (prefers-color-scheme: dark){:root{
  --plane:#0d0d0d; --surface:#1a1a19; --card:#1f1f1e;
  --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --accent:#3987e5; --good:#0ca30c; --bad:#e66767; --warn:#eda100;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14.5px;line-height:1.5}
.wrap{max-width:1280px;margin:0 auto;padding:22px 18px 70px}
h1{font-size:22px;margin:0 0 2px;letter-spacing:-.02em}
.sub{color:var(--ink2);font-size:13px;margin:0 0 18px}
.cols{display:grid;grid-template-columns:380px 1fr;gap:18px;align-items:start}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--ring);border-radius:15px;padding:16px 16px}
.card h2{font-size:14px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
label{display:block;font-size:12.5px;color:var(--ink2);margin:11px 0 4px;font-weight:600}
input,select,textarea{width:100%;background:var(--surface);color:var(--ink);
  border:1px solid var(--ring);border-radius:9px;padding:8px 10px;font:inherit;font-size:13.5px}
input:focus,select:focus,textarea:focus{outline:2px solid color-mix(in srgb,var(--accent) 45%,transparent);outline-offset:0}
textarea{resize:vertical;min-height:78px;font-variant-numeric:tabular-nums}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.hint{font-size:11.5px;color:var(--muted);margin-top:4px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}
.pchip{border:1px solid var(--ring);background:var(--surface);color:var(--ink2);
  border-radius:999px;padding:4px 10px;font-size:12px;cursor:pointer}
.pchip:hover{color:var(--ink);border-color:var(--accent)}
.btns{display:flex;gap:10px;margin-top:16px}
button.primary{flex:1;background:var(--accent);color:#fff;border:none;border-radius:10px;
  padding:11px 14px;font:inherit;font-weight:650;font-size:14px;cursor:pointer}
button.primary:disabled{opacity:.5;cursor:default}
button.ghost{background:var(--card);color:var(--ink2);border:1px solid var(--ring);
  border-radius:10px;padding:11px 14px;font:inherit;cursor:pointer}
button.ghost:hover{color:var(--ink)}
details{margin-top:12px;border-top:1px solid var(--grid);padding-top:8px}
summary{cursor:pointer;font-size:12.5px;color:var(--ink2);font-weight:600}
.keynote{font-size:11.5px;color:var(--muted);margin-top:6px;display:flex;gap:6px;align-items:flex-start}
.keynote b{color:var(--good)}
.checkline{display:flex;align-items:center;gap:8px;margin-top:11px}
.checkline input{width:auto}
.checkline label{margin:0;font-weight:500;color:var(--ink)}
/* run status */
.status{min-height:60px}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:650}
.pill.idle{background:color-mix(in srgb,var(--muted) 18%,transparent);color:var(--ink2)}
.pill.run{background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent)}
.pill.done{background:color-mix(in srgb,var(--good) 20%,transparent);color:var(--good)}
.pill.err{background:color-mix(in srgb,var(--bad) 20%,transparent);color:var(--bad)}
.mtable{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:12px}
.mtable th,.mtable td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:center;white-space:nowrap}
.mtable th:first-child,.mtable td:first-child{text-align:left}
.mtable th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.cellbox{display:inline-flex;flex-direction:column;align-items:center;gap:2px;min-width:34px}
.cellbox .bar{width:30px;height:5px;border-radius:3px;background:var(--grid);overflow:hidden}
.cellbox .bar i{display:block;height:100%;background:var(--accent)}
.cellbox small{font-size:10.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
.state-pass{color:var(--good)} .state-fail{color:var(--bad)} .state-run{color:var(--accent)}
.log{margin-top:12px;background:var(--surface);border:1px solid var(--ring);border-radius:9px;
  padding:9px 11px;max-height:150px;overflow:auto;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink2);white-space:pre-wrap}
.log .ok{color:var(--good)} .log .no{color:var(--bad)} .log .dim{color:var(--muted)}
.dash{margin-top:18px}
.dash .bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.dash h2{margin:0;font-size:14px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.dash a{font-size:12.5px;color:var(--accent);text-decoration:none}
iframe{width:100%;height:1400px;border:1px solid var(--ring);border-radius:15px;background:var(--card)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Zebra Bench — run a benchmark</h1>
  <p class="sub">Paste an API key, choose one or more models, press Run. Progress streams live; the dashboard below updates after each level.</p>

  <div class="cols">
    <!-- ============ config ============ -->
    <div class="card" id="cfgCard">
      <h2>Provider &amp; models</h2>

      <label>Provider preset</label>
      <select id="preset">
        <option value="openai">OpenAI</option>
        <option value="openrouter">OpenRouter</option>
        <option value="local">Local (Ollama / vLLM)</option>
        <option value="mock">Mock — no key, offline self-test</option>
      </select>

      <label>Base URL</label>
      <input id="baseUrl" value="https://api.openai.com/v1" spellcheck="false">

      <div id="keyBlock">
        <label>API key</label>
        <input id="apiKey" type="password" placeholder="sk-…" autocomplete="off" spellcheck="false">
        <div class="keynote">🔒 <span>Sent only to this local process. The provider &amp; key are kept in the <b>local DB</b> (bench.db, git-ignored) so saved presets can reuse them.</span></div>
      </div>

      <div id="mockBlock" style="display:none">
        <label>Mock mode</label>
        <select id="mockMode">
          <option value="perfect">perfect — always solves</option>
          <option value="noisy" selected>noisy — degrades with size</option>
          <option value="dumb">dumb — never solves</option>
        </select>
        <div class="hint">Runs the whole pipeline with no network, to try the UI.</div>
      </div>

      <label>Models <span class="hint" style="font-weight:400">(one per line)</span></label>
      <textarea id="models" spellcheck="false" placeholder="gpt-4o-mini&#10;gpt-4o"></textarea>
      <div class="chips" id="modelChips"></div>

      <h2 style="margin-top:18px">Benchmark</h2>
      <div class="row3">
        <div><label>Attempts / level</label><input id="attempts" type="number" min="1" value="3"></div>
        <div><label>Pass ratio</label><input id="passRatio" type="number" step="0.01" min="0" max="1" value="0.67"></div>
        <div><label>Parallel</label><input id="parallel" type="number" min="1" value="1"></div>
      </div>
      <label>Levels (houses×properties)</label>
      <input id="levels" value="3x3,4x4,5x5,6x6,7x7" spellcheck="false">
      <div class="row">
        <div><label>Difficulty</label>
          <select id="difficulty"><option>easy</option><option selected>medium</option><option>hard</option></select>
        </div>
        <div><label>Max tokens</label><input id="maxTokens" type="number" value="16000"></div>
      </div>

      <details>
        <summary>Advanced (reasoning models, seeds)</summary>
        <label>Tokens parameter</label>
        <select id="tokensParam">
          <option value="max_tokens">max_tokens (default)</option>
          <option value="max_completion_tokens">max_completion_tokens (o-series / reasoning)</option>
        </select>
        <div class="row">
          <div><label>Temperature</label><input id="temperature" type="number" step="0.1" value="0"></div>
          <div><label>Timeout (s)</label><input id="timeout" type="number" value="600"></div>
        </div>
        <div class="checkline"><input type="checkbox" id="noTemp"><label for="noTemp">Omit temperature (some reasoning models reject it)</label></div>
        <label>Seed base</label>
        <input id="seed" value="zebra-bench-v1" spellcheck="false">
        <div class="hint">Same seed ⇒ same puzzles for every model, so scores are comparable.</div>
      </details>

      <div class="checkline"><input type="checkbox" id="strictNoCode"><label for="strictNoCode">Strict no-code — disqualify replies that contain code, even if the grid is right</label></div>
      <div class="checkline"><input type="checkbox" id="fresh"><label for="fresh">Start fresh — clear previous results first</label></div>

      <div class="btns">
        <button class="primary" id="runBtn">Run benchmark</button>
        <button class="ghost" id="stopBtn" disabled>Stop</button>
      </div>
    </div>

    <!-- ============ live status ============ -->
    <div>
      <div class="card status">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <h2 style="margin:0">Run status</h2>
          <span class="pill idle" id="statePill">idle</span>
        </div>
        <div id="mtableWrap"><p class="hint" style="margin:14px 0 0">No run yet.</p></div>
        <div class="log" id="log" style="display:none"></div>
      </div>

      <div class="dash">
        <div class="bar">
          <h2>Dashboard</h2>
          <a href="/report" target="_blank" rel="noopener">Open full dashboard ↗</a>
        </div>
        <iframe id="dash" src="/report" title="dashboard"></iframe>
      </div>
    </div>
  </div>

  <!-- ============ manual mode ============ -->
  <div class="card" id="manualCard" style="margin-top:18px">
    <h2>Manual mode — send a puzzle by hand</h2>
    <p class="hint" style="margin-top:0">For models without an API (web chats and such): generate a puzzle, copy the prompt to the model yourself, paste its reply back. Results go to the same file and dashboard, and every entry keeps the puzzle seed so you always see which puzzle was sent.</p>

    <div class="cols" style="grid-template-columns:380px 1fr;margin-top:8px">
      <div>
        <label>Model name</label>
        <input id="manualModel" placeholder="gpt-5 (web)" spellcheck="false">

        <label>Level (houses×properties)</label>
        <input id="manualLevel" value="4x4" spellcheck="false">

        <label>Difficulty</label>
        <select id="manualDifficulty"><option>easy</option><option selected>medium</option><option>hard</option></select>

        <label>Seed code <span class="hint" style="font-weight:400">(optional — replay a specific puzzle; overrides level/difficulty)</span></label>
        <input id="manualSeed" placeholder="4h4p-m-1a2b3c4d" spellcheck="false">

        <div class="btns"><button class="primary" id="genPuzzleBtn">Generate puzzle</button></div>
        <div class="btns" style="margin-top:8px;display:none" id="manualActions">
          <button class="ghost" id="copyPromptBtn">Copy prompt</button>
          <button class="ghost" id="newSeedBtn">Generate another</button>
        </div>
        <div class="checkline" style="margin-top:14px;display:none" id="manualGradeRow"><input type="checkbox" id="manualStrict"><label for="manualStrict">Strict no-code — a reply containing code counts as a fail</label></div>
        <div class="btns" style="display:none" id="manualGradeBtns"><button class="primary" id="gradeBtn">Grade only</button><button class="ghost" id="saveBtn">Save result</button></div>
        <div id="manualVerdict" style="margin-top:10px;display:none"></div>
      </div>

      <div>
        <div id="manualPuzzle" style="display:none">
          <p class="hint" id="manualMeta" style="margin-top:0"></p>
          <label>Prompt — copy and send it to the model</label>
          <textarea id="manualPrompt" readonly style="min-height:180px;font-size:12.5px"></textarea>
          <details style="margin-top:8px"><summary>Show expected answer (spoiler)</summary>
            <pre id="manualExpected" style="font-size:12px;white-space:pre-wrap;color:var(--ink2)"></pre>
          </details>
          <label>Model reply — paste exactly what it answered</label>
          <textarea id="manualReply" style="min-height:120px" spellcheck="false"></textarea>
        </div>
        <h2 style="margin-top:0" id="manualHistoryTitle">Sent puzzles</h2>
        <div id="manualHistory"><p class="hint" style="margin:4px 0 0">Nothing sent yet.</p></div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let PRESETS = {};
let es = null, runId = null, levelsOrder = [], modelsOrder = [], cells = {};

fetch('/api/presets').then(r=>r.json()).then(p=>{ PRESETS=p; buildPresetSelect(); applyPreset(); });

$('preset').onchange = applyPreset;
function buildPresetSelect(){
  // rebuilt from /api/presets: the four built-ins, then every provider saved
  // in the DB (its key stays server-side — the browser only learns it exists)
  const sel=$('preset'), cur=sel.value;
  sel.innerHTML='';
  Object.entries(PRESETS).forEach(([k,v])=>{
    const o=document.createElement('option');
    o.value=k;
    o.textContent = v.saved ? (v.label+' — saved') : v.label;
    sel.appendChild(o);
  });
  if(cur && PRESETS[cur]) sel.value=cur;
}
function applyPreset(){
  const p = PRESETS[$('preset').value]; if(!p) return;
  $('baseUrl').value = p.base_url;
  $('models').value = p.models.join('\n');
  renderModelChips(p.models);
  const isMock = $('preset').value === 'mock';
  $('keyBlock').style.display = isMock ? 'none' : '';
  $('mockBlock').style.display = isMock ? '' : 'none';
  if(p.tokens_param) $('tokensParam').value = p.tokens_param;
  $('apiKey').placeholder = (p.saved && p.has_key) ? 'saved — leave empty to reuse' : 'sk-…';
}
function renderModelChips(models){
  const box = $('modelChips'); box.innerHTML='';
  models.forEach(m=>{
    const c = document.createElement('span'); c.className='pchip'; c.textContent='+ '+m;
    c.title = 'add '+m;
    c.onclick = ()=>{
      const cur = $('models').value.split('\n').map(s=>s.trim()).filter(Boolean);
      if(!cur.includes(m)){ cur.push(m); $('models').value = cur.join('\n'); }
    };
    box.appendChild(c);
  });
}

function collectCfg(){
  const isMock = $('preset').value === 'mock';
  return {
    mock: isMock ? $('mockMode').value : '',
    preset: $('preset').value,
    base_url: $('baseUrl').value,
    api_key: $('apiKey').value,
    models: $('models').value,
    levels: $('levels').value,
    attempts: +$('attempts').value,
    pass_ratio: +$('passRatio').value,
    parallel: +$('parallel').value,
    difficulty: $('difficulty').value,
    max_tokens: +$('maxTokens').value,
    tokens_param: $('tokensParam').value,
    temperature: +$('temperature').value,
    no_temperature: $('noTemp').checked,
    timeout: +$('timeout').value,
    seed: $('seed').value,
    fresh: $('fresh').checked,
    strict_no_code: $('strictNoCode').checked,
  };
}

function setState(cls, txt){ const p=$('statePill'); p.className='pill '+cls; p.textContent=txt; }
function logLine(html, cls){
  const el=$('log'); el.style.display='block';
  const d=document.createElement('div'); if(cls) d.className=cls; d.innerHTML=html;
  el.appendChild(d); el.scrollTop=el.scrollHeight;
}
function shortName(m){ return m.includes('/') ? m.split('/').slice(-1)[0] : m; }
// HTML-escape data-derived strings (model names, error text) before innerHTML.
const esc = s => String(s).replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sn = m => esc(shortName(m));

function buildTable(){
  cells={};
  let h='<table class="mtable"><thead><tr><th>Model</th>';
  levelsOrder.forEach(l=> h+=`<th>${l}</th>`);
  h+='<th>Result</th></tr></thead><tbody>';
  modelsOrder.forEach(m=>{
    h+=`<tr data-m="${cssEsc(m)}"><td>${sn(m)}</td>`;
    levelsOrder.forEach(l=>{
      h+=`<td data-cell="${cssEsc(m)}||${l}"><span class="cellbox"><span class="bar"><i style="width:0%"></i></span><small>–</small></span></td>`;
    });
    h+=`<td data-res="${cssEsc(m)}" class="state-run">…</td></tr>`;
  });
  h+='</tbody></table>';
  $('mtableWrap').innerHTML=h;
}
const cssEsc = s => s.replace(/"/g,'&quot;');
function cellFor(m,l){ return $('mtableWrap').querySelector(`[data-cell="${cssEsc(m)}||${l}"]`); }

function onEvent(ev){
  if(ev.ev==='start'){
    modelsOrder=ev.models; levelsOrder=ev.levels; buildTable();
    logLine(`<span class="dim">Running ${ev.models.length} model(s) × ${ev.levels.length} level(s) × ${ev.attempts} attempts${ev.mock?' · MOCK':''}</span>`);
  } else if(ev.ev==='attempt'){
    const td=cellFor(ev.model,ev.level); if(td){
      let s=+(td.dataset.solved||0), n=+(td.dataset.n||0);
      n++; if(ev.correct) s++;
      td.dataset.solved=s; td.dataset.n=n;
      const bar=td.querySelector('i'), lab=td.querySelector('small');
      if(ev.disqualified) td.dataset.dq=(+(td.dataset.dq||0))+1;
      bar.style.width=(s/n*100)+'%';
      bar.style.background = ev.error ? 'var(--bad)' : (td.dataset.dq>0 ? 'var(--warn)' : 'var(--accent)');
      lab.textContent = td.dataset.dq>0 ? `${s}/${n} ⚠${td.dataset.dq}` : `${s}/${n}`;
    }
    if(ev.error) logLine(`<span class="no">✗ ${sn(ev.model)} ${ev.level}: ${esc(ev.error)}</span>`);
    else if(ev.disqualified) logLine(`<span class="no">⚠ ${sn(ev.model)} ${ev.level}: disqualified — reply contained code</span>`);
  } else if(ev.ev==='level_done'){
    const td=cellFor(ev.model,ev.level);
    const pass = ev.pass_ratio>=passRatioNow();
    if(td){ td.querySelector('i').style.background = pass?'var(--good)':'var(--bad)'; }
    logLine(`${pass?'<span class="ok">✓</span>':'<span class="no">✗</span>'} ${sn(ev.model)} <b>${ev.level}</b> — ${ev.solved}/${ev.attempts} solved · cell ${(ev.cell_acc*100).toFixed(0)}%`);
    $('dash').src='/report?t='+Date.now();
  } else if(ev.ev==='model_stop'){
    const r=$('mtableWrap').querySelector(`[data-res="${cssEsc(ev.model)}"]`);
    // handled at model_done
  } else if(ev.ev==='model_done'){
    const r=$('mtableWrap').querySelector(`[data-res="${cssEsc(ev.model)}"]`);
    if(r){ if(ev.max_level){ r.textContent=ev.max_level; r.className='state-pass'; }
           else { r.textContent='none'; r.className='state-fail'; } }
    logLine(`<span class="dim">→ ${sn(ev.model)} highest cleared: ${ev.max_level||'none'}</span>`);
  } else if(ev.ev==='error'){
    logLine(`<span class="no">ERROR: ${esc(ev.message)}</span>`); setState('err','error');
  } else if(ev.ev==='done'){
    finish(ev.stopped);
  }
}
function passRatioNow(){ return +$('passRatio').value || 0.67; }

function finish(stopped){
  setState(stopped?'err':'done', stopped?'stopped':'done');
  $('runBtn').disabled=false; $('stopBtn').disabled=true;
  $('dash').src='/report?t='+Date.now();
  if(es){ es.close(); es=null; }
}

$('runBtn').onclick = async ()=>{
  $('runBtn').disabled=true;
  $('log').innerHTML=''; $('log').style.display='none';
  setState('run','starting…');
  let res;
  try{
    res = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(collectCfg())}).then(r=>r.json());
  }catch(e){ setState('err','error'); logLine('<span class="no">'+esc(e)+'</span>'); $('runBtn').disabled=false; return; }
  if(res.error){ setState('err','error'); $('mtableWrap').innerHTML=`<p class="hint" style="color:var(--bad);margin-top:12px">${esc(res.error)}</p>`; $('runBtn').disabled=false; return; }
  runId=res.id; setState('run','running'); $('stopBtn').disabled=false;
  es = new EventSource('/api/events?id='+encodeURIComponent(runId));
  es.onmessage = e => onEvent(JSON.parse(e.data));
  es.onerror = ()=>{ /* dropped connections auto-retry and resume via Last-Event-ID */ };
};

$('stopBtn').onclick = async ()=>{
  if(!runId) return;
  $('stopBtn').disabled=true; setState('run','stopping…');
  await fetch('/api/stop?id='+encodeURIComponent(runId),{method:'POST'});
};

// ---------- manual mode ----------
let manualPuzzle = null;

async function postJson(url, body){
  try{
    return await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body||{})}).then(r=>r.json());
  }catch(e){ return {error:String(e)}; }
}

async function generatePuzzle(){
  const btn=$('genPuzzleBtn'); btn.disabled=true;
  const r=await postJson('/api/manual/puzzle',{
    level:$('manualLevel').value, difficulty:$('manualDifficulty').value, seed:$('manualSeed').value});
  btn.disabled=false;
  if(r.error){ $('manualVerdict').innerHTML=`<p class="hint" style="color:var(--bad);margin:0">${esc(r.error)}</p>`; return; }
  manualPuzzle=r;
  $('manualSeed').value=r.seed;
  $('manualMeta').innerHTML=`Puzzle <b>${esc(r.seed)}</b> · ${esc(r.level)} · ${esc(r.difficulty)} · ${r.clues} clues`;
  $('manualPrompt').value=r.prompt;
  $('manualExpected').textContent='grid: '+JSON.stringify(r.expected_grid,null,1)
    +'\nanswer: '+r.expected_answer;
  $('manualReply').value='';
  $('manualVerdict').innerHTML=''; $('manualVerdict').style.display='none';
  $('manualPuzzle').style.display='';
  $('manualActions').style.display='';
  $('manualGradeRow').style.display='';
  $('manualGradeBtns').style.display='';
  $('manualVerdict').style.display='';
}

$('copyPromptBtn').onclick = async ()=>{
  const t=$('manualPrompt');
  try{ await navigator.clipboard.writeText(t.value); }
  catch(e){ t.select(); document.execCommand('copy'); }
  const b=$('copyPromptBtn'), old=b.textContent;
  b.textContent='Copied ✓'; setTimeout(()=>b.textContent=old,1200);
};

$('newSeedBtn').onclick = ()=>{ $('manualSeed').value=''; generatePuzzle(); };

$('genPuzzleBtn').onclick = generatePuzzle;

async function gradeMan(save){
  const g=$('gradeBtn'), s=$('saveBtn');
  g.disabled=true; s.disabled=true;
  const r=await postJson('/api/manual/submit',{
    model:$('manualModel').value, seed:manualPuzzle && manualPuzzle.seed,
    reply:$('manualReply').value, strict_no_code:$('manualStrict').checked, save});
  g.disabled=false; s.disabled=false;
  if(r.error){ $('manualVerdict').innerHTML=`<p class="hint" style="color:var(--bad);margin:0">${esc(r.error)}</p>`; return; }
  const rec=r.record;
  const pct=(rec.cell_acc*100).toFixed(0);
  let badge, detail;
  if(rec.disqualified){ badge='<span class="pill err">disqualified</span>';
    detail='the reply contains code — counted as a fail (strict no-code)'; }
  else if(rec.correct){ badge='<span class="pill done">correct</span>';
    detail='full grid and the answer are right'; }
  else { badge='<span class="pill err">incorrect</span>';
    detail=`cells ${pct}% · answer ${rec.answer_correct?'right':'wrong'}`
      +(rec.parsed?'':' · reply had no valid JSON'); }
  const tail = r.saved
    ? `<br><span class="hint">saved: ${esc(rec.model)} @ ${esc(rec.level)} — seed ${esc(rec.seed)} · expected answer: ${esc(r.expected_answer)}</span>`
    : `<br><span class="hint">graded, not saved — press “Save result” to record it</span>`;
  $('manualVerdict').innerHTML=
    `${badge} <span class="hint" style="display:inline">${esc(detail)}</span>${tail}`;
  if(r.saved){ loadManualHistory(); $('dash').src='/report?t='+Date.now(); }
}

$('gradeBtn').onclick = ()=>gradeMan(false);
$('saveBtn').onclick = ()=>gradeMan(true);

async function loadManualHistory(){
  let rows=[];
  try{ rows=await fetch('/api/manual/history').then(r=>r.json()); }catch(e){}
  if(!rows.length){ $('manualHistory').innerHTML='<p class="hint" style="margin:4px 0 0">Nothing sent yet.</p>'; return; }
  let h='<div style="max-height:260px;overflow:auto"><table class="mtable"><thead><tr>'
    +'<th>When</th><th>Model</th><th>Level</th><th>Seed</th><th>Result</th></tr></thead><tbody>';
  rows.forEach(r=>{
    const pct=((r.cell_acc||0)*100).toFixed(0);
    let res;
    if(r.disqualified) res=`<span class="state-fail">⚠ dq (code)</span>`;
    else if(r.correct) res=`<span class="state-pass">✓ ${pct}%</span>`;
    else res=`<span class="state-fail">✗ ${pct}%${r.parsed?'':' · unparsed'}</span>`;
    h+=`<tr><td>${esc(r.ts||'')}</td><td>${esc(r.model||'')}</td><td>${esc(r.level||'')}</td>`
      +`<td><code style="font-size:11px">${esc(r.seed||'')}</code></td><td>${res}</td></tr>`;
  });
  $('manualHistory').innerHTML=h+'</tbody></table></div>';
}
loadManualHistory();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; keep 127.0.0.1 unless you understand the exposure")
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--db", default=db.DEFAULT_DB,
                    help="SQLite file for puzzles/results/providers ('' disables)")
    a = ap.parse_args()

    global RESULTS_FILE
    RESULTS_FILE = a.results

    if a.db:
        db.init(a.db)
        # first start with an empty DB: pull the existing JSONL history in, so
        # the dashboard and manual history don't start from zero
        if not db.load_results() and os.path.exists(RESULTS_FILE):
            n = db.import_jsonl(RESULTS_FILE)
            if n:
                print(f"  imported {n} rows from {RESULTS_FILE} into {a.db}")

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"Zebra Bench runner on {url}")
    print(f"  results file: {RESULTS_FILE}" + (f"; db: {a.db}" if a.db else ""))
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: bound to {a.host}, not loopback — anyone who can reach this port can\n"
              "           submit runs (API keys typed into the page) and set the target base URL\n"
              "           the server calls. Only do this on a network you trust.")
    print("  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
