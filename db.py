#!/usr/bin/env python3
"""SQLite storage for zebra-bench: puzzles, results, providers & models.

    python serve.py                 # runs use bench.db automatically
    python bench.py --model ...     # ditto from the CLI
    python db.py stats              # what's inside the DB
    python db.py import results.jsonl

Pure stdlib (`sqlite3`), one file, safe to call from several threads — writes
run behind a lock over short-lived connections, and the DB is opened in WAL
mode so readers never block the benchmark writer threads.

Tables
------
providers  one row per base URL ever run against: label, API key, tokens_param
models     model names seen per provider (+ use counts)
puzzles    every generated puzzle, keyed by its seed code: the prompt, the
           clue texts, the question, and the full solution — a stored puzzle
           can be replayed/graded without regenerating it
runs       one row per benchmark invocation (CLI or web UI) with its config;
           never stores the API key itself (that lives in `providers`)
results    one row per graded attempt; queryable columns for the hot fields
           (model, level, seed, correct, …) plus `payload` holding the exact
           record dict, so rows read back are identical to results.jsonl lines

The API key is stored in plain text in a local, git-ignored file — same trust
level as an .env. Don't commit bench.db.
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
import re
import sqlite3
import threading
import time
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import zebra

DEFAULT_DB = "bench.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS providers (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  base_url      TEXT NOT NULL UNIQUE,
  label         TEXT,
  api_key       TEXT NOT NULL DEFAULT '',
  tokens_param  TEXT NOT NULL DEFAULT 'max_tokens',
  updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  provider_id  INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  uses         INTEGER NOT NULL DEFAULT 0,
  last_used    TEXT,
  UNIQUE (provider_id, name)
);
CREATE TABLE IF NOT EXISTS puzzles (
  seed          TEXT PRIMARY KEY,
  N             INTEGER NOT NULL,
  M             INTEGER NOT NULL,
  difficulty    TEXT NOT NULL,
  num           INTEGER NOT NULL,
  clue_count    INTEGER NOT NULL,
  question      TEXT NOT NULL,
  answer        TEXT NOT NULL,
  solution_grid TEXT NOT NULL,
  clue_texts    TEXT NOT NULL,
  prompt        TEXT NOT NULL,
  payload       TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  ts      TEXT NOT NULL,
  source  TEXT NOT NULL,
  config  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id   INTEGER REFERENCES runs(id) ON DELETE SET NULL,
  ts       TEXT NOT NULL,
  model    TEXT NOT NULL,
  level    TEXT NOT NULL,
  seed     TEXT,
  attempt  INTEGER,
  manual   INTEGER NOT NULL DEFAULT 0,
  correct  INTEGER NOT NULL DEFAULT 0,
  cell_acc REAL NOT NULL DEFAULT 0,
  error    TEXT,
  payload  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_model_level ON results(model, level);
CREATE INDEX IF NOT EXISTS idx_results_manual ON results(manual);
"""

_path: Optional[str] = None
_lock = threading.Lock()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def init(path: str = DEFAULT_DB) -> None:
    """Enable storage in `path`; creates the file and schema if needed."""
    global _path
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        with con:
            con.executescript(_SCHEMA)
    finally:
        con.close()
    _path = path


def is_enabled() -> bool:
    return _path is not None


def _run(fn: Callable[[sqlite3.Connection], object], write: bool) -> object:
    """Open a fresh connection, run fn inside a transaction, always close.

    Writes take the module lock so parallel benchmark threads queue up; reads
    don't (WAL mode lets them run alongside writers).
    """
    with (_lock if write else nullcontext()):
        con = sqlite3.connect(_path, timeout=30)
        try:
            con.execute("PRAGMA foreign_keys=ON")
            with con:
                return fn(con)
        finally:
            con.close()


# ------------------------------------------------------------- puzzles ------
def save_puzzle(p: Dict) -> None:
    """Store a built puzzle dict (zebra.build_puzzle output), idempotent by seed."""
    if not is_enabled():
        return
    row = (p["seed"], p["N"], p["M"], p["difficulty"],
           int(p["seed"].rsplit("-", 1)[1], 16),
           len(p["clue_texts"]), p["question"]["text"], p["question"]["answer"],
           json.dumps(p["solution_grid"]), json.dumps(p["clue_texts"]),
           zebra.render_prompt(p), json.dumps(p), now())

    def fn(con):
        con.execute("INSERT OR REPLACE INTO puzzles"
                    " (seed, N, M, difficulty, num, clue_count, question, answer,"
                    "  solution_grid, clue_texts, prompt, payload, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    _run(fn, write=True)


def get_puzzle(seed: str) -> Optional[Dict]:
    """Full puzzle dict for a seed code, or None if it isn't stored."""
    if not is_enabled():
        return None
    seed = (seed or "").strip()
    if not seed:
        return None

    def fn(con):
        return con.execute("SELECT payload FROM puzzles WHERE seed=?", (seed,)).fetchone()
    row = _run(fn, write=False)
    return json.loads(row[0]) if row else None


def list_puzzle_seeds(N: int, M: int, difficulty: str) -> List[str]:
    """Stored seed codes for a level+difficulty, sorted (stable sample order)."""
    if not is_enabled():
        return []

    def fn(con):
        return [r[0] for r in con.execute(
            "SELECT seed FROM puzzles WHERE N=? AND M=? AND difficulty=? ORDER BY seed",
            (N, M, difficulty))]
    return _run(fn, write=False)


# --------------------------------------------------------------- runs -------
def start_run(source: str, config: Dict) -> Optional[int]:
    """Record a benchmark invocation; returns its id for the result rows.

    `config` must not contain the API key — the key belongs to `providers`.
    """
    if not is_enabled():
        return None
    clean = {k: v for k, v in config.items() if "key" not in k.lower()}
    data = (now(), source, json.dumps(clean, default=str))

    def fn(con):
        return con.execute("INSERT INTO runs (ts, source, config) VALUES (?,?,?)",
                           data).lastrowid
    return _run(fn, write=True)


def finish_run(run_id: Optional[int], summary: Dict) -> None:
    """Attach the run's outcome (e.g. max level per model) to its config row."""
    if not is_enabled() or not run_id:
        return

    def fn(con):
        row = con.execute("SELECT config FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return
        cfg = json.loads(row[0])
        cfg.setdefault("summary", []).append(summary)
        con.execute("UPDATE runs SET config=? WHERE id=?",
                    (json.dumps(cfg, default=str), run_id))
    _run(fn, write=True)


# ------------------------------------------------------------- results ------
def save_result(rec: Dict, run_id: Optional[int] = None) -> None:
    """Store one graded attempt; `rec` is the same dict that goes to results.jsonl."""
    if not is_enabled():
        return
    ts = rec.get("ts") or now()
    row = (run_id, ts, rec.get("model", ""), rec.get("level", ""), rec.get("seed"),
           rec.get("attempt"), 1 if rec.get("manual") else 0,
           1 if rec.get("correct") else 0, rec.get("cell_acc") or 0.0,
           rec.get("error"), json.dumps(rec, default=str))

    def fn(con):
        con.execute("INSERT INTO results (run_id, ts, model, level, seed, attempt,"
                    " manual, correct, cell_acc, error, payload)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
    _run(fn, write=True)


def load_results() -> List[Dict]:
    """All attempt records in insertion order, each identical to its JSONL line."""
    if not is_enabled():
        return []

    def fn(con):
        return con.execute("SELECT payload FROM results ORDER BY id").fetchall()
    return [json.loads(payload) for (payload,) in _run(fn, write=False)]


def count_attempts(model: str, level: str) -> int:
    if not is_enabled():
        return 0

    def fn(con):
        return con.execute("SELECT COUNT(*) FROM results WHERE model=? AND level=?",
                           (model, level)).fetchone()
    return _run(fn, write=False)[0]


def manual_history(limit: int = 60) -> List[Dict]:
    """Recent manual-mode records, newest first, in the /api/manual/history shape."""
    if not is_enabled():
        return []

    def fn(con):
        return con.execute("SELECT payload FROM results WHERE manual=1"
                           " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    rows = _run(fn, write=False)
    keys = ("ts", "model", "level", "seed", "clues", "correct", "cell_acc",
            "parsed", "code_flag", "disqualified", "given_answer", "expected_answer")
    out = []
    for (payload,) in rows:
        r = json.loads(payload)
        out.append({k: r.get(k) for k in keys})
        out[-1]["answer_correct"] = r.get("answer_correct")
    return out


def manual_used_seeds() -> set:
    """Seeds already answered through manual mode, so the picker can avoid them."""
    if not is_enabled():
        return set()

    def fn(con):
        return {r[0] for r in con.execute(
            "SELECT DISTINCT seed FROM results WHERE manual=1 AND seed IS NOT NULL")}
    return _run(fn, write=False)


def clear_results() -> None:
    if not is_enabled():
        return

    def fn(con):
        con.execute("DELETE FROM results")
        con.execute("DELETE FROM runs")
    _run(fn, write=True)


# ------------------------------------------- providers & models -------------
def _host_label(base_url: str) -> str:
    host = (urlparse(base_url).hostname or base_url or "").lower()
    return re.sub(r"^www\.", "", host) or base_url


def upsert_provider(base_url: str, api_key: str = "", label: Optional[str] = None,
                    tokens_param: Optional[str] = None) -> int:
    """Create/update the provider row for a base URL; returns its id."""
    if not is_enabled():
        return 0

    def fn(con):
        row = con.execute("SELECT id, api_key FROM providers WHERE base_url=?",
                          (base_url,)).fetchone()
        if row:
            provider_id, old_key = row
            # a run with no key (e.g. env fallback) must not wipe a stored one
            con.execute("UPDATE providers SET label=?, api_key=?, tokens_param=?,"
                        " updated_at=? WHERE id=?",
                        (label or _host_label(base_url), api_key or old_key,
                         tokens_param or "max_tokens", now(), provider_id))
            return provider_id
        return con.execute("INSERT INTO providers (base_url, label, api_key,"
                           " tokens_param, updated_at) VALUES (?,?,?,?,?)",
                           (base_url, label or _host_label(base_url), api_key,
                            tokens_param or "max_tokens", now())).lastrowid
    return _run(fn, write=True)


def remember_models(provider_id: int, models: List[str]) -> None:
    """Bump (or create) a use counter for each model name under a provider."""
    if not is_enabled() or not models:
        return

    def fn(con):
        for m in models:
            con.execute("UPDATE models SET uses=uses+1, last_used=?"
                        " WHERE provider_id=? AND name=?", (now(), provider_id, m))
            con.execute("INSERT OR IGNORE INTO models (provider_id, name) VALUES (?,?)",
                        (provider_id, m))
    _run(fn, write=True)


def get_provider(base_url: str) -> Optional[Dict]:
    if not is_enabled():
        return None

    def fn(con):
        return con.execute("SELECT base_url, label, api_key, tokens_param, updated_at"
                           " FROM providers WHERE base_url=?", (base_url,)).fetchone()
    row = _run(fn, write=False)
    return dict(zip(("base_url", "label", "api_key", "tokens_param", "updated_at"), row)) if row else None


def list_providers() -> List[Dict]:
    """Saved providers with their model names, most recently used first."""
    if not is_enabled():
        return []

    def fn(con):
        provs = con.execute("SELECT id, base_url, label, api_key, tokens_param, updated_at"
                            " FROM providers ORDER BY updated_at DESC").fetchall()
        out = []
        for pid, base_url, label, key, tp, updated in provs:
            names = [r[0] for r in con.execute(
                "SELECT name FROM models WHERE provider_id=? ORDER BY uses DESC, name",
                (pid,))]
            out.append({"id": pid, "base_url": base_url, "label": label,
                        "has_key": bool(key), "tokens_param": tp,
                        "models": names, "updated_at": updated})
        return out
    return _run(fn, write=False)


# ------------------------------------------------------------- import -------
def import_jsonl(path: str, run_id: Optional[int] = None) -> int:
    """Load a results.jsonl file into the DB; returns how many rows were added."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            save_result(json.loads(line), run_id=run_id)
            n += 1
    return n


def stats() -> Dict:
    if not is_enabled():
        return {"enabled": False}

    def fn(con):
        one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
        return {
            "enabled": True, "path": _path,
            "providers": one("SELECT COUNT(*) FROM providers"),
            "models": one("SELECT COUNT(*) FROM models"),
            "puzzles": one("SELECT COUNT(*) FROM puzzles"),
            "runs": one("SELECT COUNT(*) FROM runs"),
            "results": one("SELECT COUNT(*) FROM results"),
            "manual_results": one("SELECT COUNT(*) FROM results WHERE manual=1"),
        }
    return _run(fn, write=False)


# ------------------------------------------------------------------ CLI -----
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB, help="database file (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd")

    p_imp = sub.add_parser("import", help="load a results.jsonl into the DB")
    p_imp.add_argument("file", nargs="?", default="results.jsonl")

    sub.add_parser("stats", help="row counts per table")
    sub.add_parser("providers", help="saved providers and their models")
    a = ap.parse_args()

    init(a.db)
    if a.cmd == "import":
        n = import_jsonl(a.file)
        print(f"imported {n} rows from {a.file} -> {a.db}")
    elif a.cmd == "providers":
        for p in list_providers():
            key = "key saved" if p["has_key"] else "no key"
            print(f"{p['label']:<28} {p['base_url']:<42} {key}")
            for m in p["models"]:
                print(f"    {m}")
    else:
        s = stats()
        if not s.get("enabled"):
            print("storage disabled")
            return
        print(f"{s['path']}: {s['providers']} providers, {s['models']} models, "
              f"{s['puzzles']} puzzles, {s['runs']} runs, "
              f"{s['results']} results ({s['manual_results']} manual)")


if __name__ == "__main__":
    main()
