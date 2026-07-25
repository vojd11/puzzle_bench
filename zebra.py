"""Zebra puzzle generator + solver (Python port of zebra-puzzle.html).

Uses the same mulberry32 PRNG and the same algorithm order as the HTML page,
so a seed string like "5h4p-m-b50737b4" produces the identical puzzle in both.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_CATS: List[Dict] = [
    {"name": "Color",       "values": ["Red", "Green", "Blue", "Yellow", "White", "Purple", "Orange"]},
    {"name": "Nationality", "values": ["Brit", "Swede", "Dane", "German", "Norwegian", "Spaniard", "Italian"]},
    {"name": "Drink",       "values": ["Tea", "Coffee", "Milk", "Beer", "Water", "Juice", "Wine"]},
    {"name": "Pet",         "values": ["Dog", "Bird", "Cat", "Horse", "Fish", "Rabbit", "Turtle"]},
    {"name": "Snack",       "values": ["Chips", "Cookies", "Popcorn", "Pretzels", "Nuts", "Candy", "Fruit"]},
    {"name": "Hobby",       "values": ["Chess", "Painting", "Running", "Gardening", "Reading", "Cooking", "Photography"]},
    {"name": "Job",         "values": ["Doctor", "Teacher", "Chef", "Pilot", "Artist", "Farmer", "Engineer"]},
    {"name": "Flower",      "values": ["Rose", "Tulip", "Lily", "Daisy", "Orchid", "Iris", "Poppy"]},
]

DIFFS = {"e": "easy", "m": "medium", "h": "hard"}
M32 = 0xFFFFFFFF


# ----------------------------------------------------------------- PRNG ----
def make_rng(seed: int) -> Callable[[], float]:
    """mulberry32 — bit-exact port of the JS PRNG used by the HTML page."""
    state = seed & M32

    def rnd() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & M32
        s = state
        t = ((s ^ (s >> 15)) * (1 | s)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    return rnd


def shuffle(arr: List, rnd: Callable[[], float]) -> List:
    for i in range(len(arr) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# --------------------------------------------------------------- solver ----
def popcount(x: int) -> int:
    return bin(x).count("1")


def low_idx(x: int) -> int:
    return (x & -x).bit_length() - 1


def high_idx(x: int) -> int:
    return x.bit_length() - 1


def init_masks(M: int, N: int) -> List[List[int]]:
    all_ = (1 << N) - 1
    return [[all_] * N for _ in range(M)]


def propagate(m: List[List[int]], clues: List[Dict], N: int) -> bool:
    all_ = (1 << N) - 1
    M = len(m)
    changed = True
    while changed:
        changed = False
        for cl in clues:
            ac, av = cl["a"]
            A = m[ac][av]
            B = None
            bc = bv = None
            if cl.get("b") is not None:
                bc, bv = cl["b"]
                B = m[bc][bv]
            nA, nB = A, B
            t = cl["t"]
            if t == "pos":
                nA = A & (1 << cl["h"])
            elif t == "same":
                nA = A & B
                nB = nA
            elif t == "diff":
                if popcount(A) == 1:
                    nB = B & ~A
                if popcount(nB) == 1:
                    nA = A & ~nB
            elif t == "imm":
                nB = B & ((A << 1) & all_)
                nA = A & (nB >> 1)
            elif t == "adj":
                nB = B & (((A << 1) | (A >> 1)) & all_)
                nA = A & (((nB << 1) | (nB >> 1)) & all_)
            elif t == "ord":
                if A == 0 or B == 0:
                    return False
                nA = A & ((1 << high_idx(B)) - 1)
                if nA == 0:
                    return False
                nB = B & ~((1 << (low_idx(nA) + 1)) - 1)
            if nA == 0 or nB == 0:
                return False
            if nA != A:
                m[ac][av] = nA
                changed = True
            if nB is not None and nB != B:
                m[bc][bv] = nB
                changed = True

        for c in range(M):
            row = m[c]
            for v in range(N):
                if row[v] == 0:
                    return False
                if popcount(row[v]) == 1:
                    for w in range(N):
                        if w != v and (row[w] & row[v]):
                            row[w] &= ~row[v]
                            if row[w] == 0:
                                return False
                            changed = True
            for h in range(N):
                bit = 1 << h
                cnt = 0
                last = -1
                for v in range(N):
                    if row[v] & bit:
                        cnt += 1
                        last = v
                if cnt == 0:
                    return False
                if cnt == 1 and row[last] != bit:
                    row[last] = bit
                    changed = True
    return True


def solve(M: int, N: int, clues: List[Dict], limit: int = 2) -> List[List[List[int]]]:
    """Returns up to `limit` solutions; sol[cat][house] = value index."""
    out: List[List[List[int]]] = []

    def rec(m: List[List[int]]) -> None:
        if len(out) >= limit:
            return
        if not propagate(m, clues, N):
            return
        bc = bv = -1
        best = 99
        for c in range(M):
            for v in range(N):
                p = popcount(m[c][v])
                if 1 < p < best:
                    best, bc, bv = p, c, v
        if bc < 0:
            sol = []
            for c in range(M):
                r = [0] * N
                for v in range(N):
                    r[low_idx(m[c][v])] = v
                sol.append(r)
            out.append(sol)
            return
        bits = m[bc][bv]
        while bits and len(out) < limit:
            b = bits & -bits
            bits ^= b
            m2 = [row[:] for row in m]
            m2[bc][bv] = b
            rec(m2)

    rec(init_masks(M, N))
    return out


# ------------------------------------------------------------ generator ----
def house_of(sol, c, v):
    return sol[c].index(v)


WEIGHTS = {
    "easy":   {"pos": 1.0,  "same": 1.0, "imm": 1.0, "adj": 0.8, "ord": 0.6, "diff": 0.15},
    "medium": {"pos": 0.35, "same": 0.8, "imm": 0.8, "adj": 0.8, "ord": 0.7, "diff": 0.3},
    "hard":   {"pos": 0.10, "same": 0.5, "imm": 0.5, "adj": 0.7, "ord": 0.8, "diff": 0.5},
}


def build_pool(sol, M, N, rnd, difficulty):
    pool: List[Dict] = []
    pos_clues: List[Dict] = []
    for c in range(M):
        for v in range(N):
            pos_clues.append({"t": "pos", "a": [c, v], "b": None, "h": house_of(sol, c, v)})
    for c1 in range(M):
        for v1 in range(N):
            h1 = house_of(sol, c1, v1)
            for c2 in range(M):
                if c2 == c1:
                    continue
                for v2 in range(N):
                    h2 = house_of(sol, c2, v2)
                    a, b = [c1, v1], [c2, v2]
                    if h1 == h2:
                        if c1 < c2:
                            pool.append({"t": "same", "a": a, "b": b})
                    else:
                        if h2 == h1 + 1:
                            pool.append({"t": "imm", "a": a, "b": b})
                        if abs(h1 - h2) == 1 and c1 < c2:
                            pool.append({"t": "adj", "a": a, "b": b})
                        if h1 < h2:
                            pool.append({"t": "ord", "a": a, "b": b})
                        if c1 < c2:
                            pool.append({"t": "diff", "a": a, "b": b})
    w = WEIGHTS.get(difficulty, WEIGHTS["medium"])
    picked = [c for c in pool if rnd() < w[c["t"]]]
    picked_pos = [c for c in pos_clues if rnd() < w["pos"]]
    main = shuffle(picked + picked_pos, rnd)
    return main + shuffle(list(pos_clues), rnd)


def generate(M: int, N: int, difficulty: str, rnd: Callable[[], float]) -> Optional[Dict]:
    sol = [shuffle(list(range(N)), rnd) for _ in range(M)]  # sol[cat][house] = value idx
    pool = build_pool(sol, M, N, rnd, difficulty)
    clues: List[Dict] = []
    for cl in pool:
        clues.append(cl)
        if len(clues) < max(3, M):
            continue
        if len(solve(M, N, clues, 2)) == 1:
            break
    if len(solve(M, N, clues, 2)) != 1:
        return None
    order = shuffle(list(range(len(clues))), rnd)
    dead = set()
    for i in order:
        trial = [c for j, c in enumerate(clues) if j != i and j not in dead]
        if len(solve(M, N, trial, 2)) == 1:
            dead.add(i)
    final = [c for j, c in enumerate(clues) if j not in dead]
    return {"sol": sol, "clues": shuffle(final, rnd)}


def make_question(sol, cats, N, rnd) -> Dict:
    M = len(cats)
    bi = int(rnd() * M)
    ai = int(rnd() * M)
    while ai == bi:
        ai = int(rnd() * M)
    house = int(rnd() * N)
    av, bv = sol[ai][house], sol[bi][house]
    A, B = cats[ai], cats[bi]
    person = re.search(r"nation|people|person|name|who", B["name"], re.I)
    text = (f"Who has {A['values'][av]} ({A['name']})?" if person
            else f"Which {B['name']} belongs to the house with {A['values'][av]} ({A['name']})?")
    return {"text": text, "cat": B["name"], "cat_idx": bi, "val_idx": bv, "answer": B["values"][bv]}


def clue_text(cl: Dict, cats) -> str:
    def it(x):
        c, v = x
        return f"{cats[c]['values'][v]} ({cats[c]['name']})"
    A = it(cl["a"])
    B = it(cl["b"]) if cl.get("b") else ""
    t = cl["t"]
    return {
        "pos":  f"{A} is in house {cl.get('h', -1) + 1}.",
        "same": f"{A} is in the same house as {B}.",
        "diff": f"{A} is not in the same house as {B}.",
        "imm":  f"{A} is directly to the left of {B}.",
        "adj":  f"{A} is next to {B}.",
        "ord":  f"{A} is somewhere to the left of {B}.",
    }[t]


# ------------------------------------------------------------ seed codes ---
def seed_code(N: int, M: int, difficulty: str, num: int) -> str:
    return f"{N}h{M}p-{difficulty[0]}-{num & M32:08x}"


def parse_seed(code: str) -> Optional[Dict]:
    m = re.fullmatch(r"(\d+)h(\d+)p-([emh])-([0-9a-fA-F]{1,8})", code.strip())
    if not m:
        return None
    return {"N": int(m.group(1)), "M": int(m.group(2)),
            "difficulty": DIFFS[m.group(3).lower()], "num": int(m.group(4), 16)}


def build_puzzle(N: int, M: int, difficulty: str = "medium", num: int = 0,
                 cats: Optional[List[Dict]] = None) -> Dict:
    """Full puzzle for the given seed number. Mirrors the HTML page exactly."""
    cats = cats or DEFAULT_CATS
    cats = [{"name": c["name"], "values": c["values"][:N]} for c in cats[:M]]
    rnd = make_rng(num)
    res = None
    tries = 0
    while res is None and tries < 5:
        res = generate(M, N, difficulty, rnd)
        tries += 1
    if res is None:
        raise RuntimeError("generation failed")
    question = make_question(res["sol"], cats, N, rnd)
    return {
        "seed": seed_code(N, M, difficulty, num),
        "N": N, "M": M, "difficulty": difficulty,
        "cats": cats,
        "sol": res["sol"],
        "clues": res["clues"],
        "clue_texts": [clue_text(c, cats) for c in res["clues"]],
        "question": question,
        "solution_grid": {c["name"]: [c["values"][res["sol"][ci][h]] for h in range(N)]
                          for ci, c in enumerate(cats)},
    }


# ------------------------------------------------------------- prompting ---
PROMPT_TEMPLATE = """You are solving a logic puzzle (a "zebra puzzle").

There are {N} houses in a row, numbered 1 to {N} from left to right.
Each house has exactly one value of each of the following {M} properties, and \
every value is used exactly once:

{categories}

Clues:
{clues}

Question: {question}

RULES — read carefully:
- Solve this by reasoning only, in your head / in your written reasoning.
- You must NOT write, generate, or execute any code, script, program, solver, \
SAT/CSP encoding, or pseudo-code implementation of a solver. No tool use.
- "to the left of" means a strictly smaller house number; "directly to the left of" \
means house number exactly one smaller; "next to" means the house numbers differ by 1.
- The puzzle has exactly one consistent solution.

Answer format — end your reply with a single JSON object and nothing after it:

{{"grid": {{{grid_example}}}, "answer": "<answer to the question>"}}

Each property maps to a list of {N} values ordered by house number (house 1 first).
"""


def render_prompt(p: Dict) -> str:
    cats = "\n".join(f"- {c['name']}: " + ", ".join(c["values"]) for c in p["cats"])
    clues = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(p["clue_texts"]))
    grid_example = ", ".join(
        f'"{c["name"]}": [' + ", ".join(f'"<house {h + 1}>"' for h in range(p["N"])) + "]"
        for c in p["cats"]
    )
    return PROMPT_TEMPLATE.format(N=p["N"], M=p["M"], categories=cats, clues=clues,
                                  question=p["question"]["text"], grid_example=grid_example)


# --------------------------------------------------------------- grading ---
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def extract_json(text: str) -> Optional[Dict]:
    """Last complete top-level JSON object in the text."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for i in reversed(starts):
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j + 1])
                        if isinstance(obj, dict) and "grid" in obj:
                            return obj
                    except Exception:
                        pass
                    break
    return None


def grade(p: Dict, reply: str) -> Dict:
    obj = extract_json(reply)
    N = p["N"]
    total = N * p["M"]
    if not obj or not isinstance(obj.get("grid"), dict):
        return {"parsed": False, "cells_correct": 0, "cells_total": total,
                "cell_acc": 0.0, "grid_correct": False, "answer_correct": False,
                "correct": False}
    truth = p["solution_grid"]
    got = {_norm(k): v for k, v in obj["grid"].items()}
    cells = 0
    for name, vals in truth.items():
        row = got.get(_norm(name))
        if not isinstance(row, list):
            continue
        for h, want in enumerate(vals):
            if h < len(row) and _norm(row[h]) == _norm(want):
                cells += 1
    grid_ok = cells == total
    ans_ok = _norm(obj.get("answer", "")) == _norm(p["question"]["answer"])
    return {"parsed": True, "cells_correct": cells, "cells_total": total,
            "cell_acc": cells / total, "grid_correct": grid_ok,
            "answer_correct": ans_ok, "correct": grid_ok and ans_ok}


CODE_PATTERNS = re.compile(
    r"```(?:python|py|js|javascript|c\+\+|java|prolog)|"
    r"\bimport itertools\b|\bfrom itertools\b|\bdef solve\b|\bfor perm in\b|"
    r"\bpython_repl\b|\bexec\(|\bprint\(",
    re.I)


def looks_like_code(reply: str) -> bool:
    return bool(CODE_PATTERNS.search(reply))


def solve_puzzle_answer(p: Dict) -> str:
    """Reference answer produced by the built-in solver (used by mock models)."""
    sols = solve(p["M"], p["N"], p["clues"], 2)
    sol = sols[0]
    grid = {c["name"]: [c["values"][sol[ci][h]] for h in range(p["N"])]
            for ci, c in enumerate(p["cats"])}
    q = p["question"]
    return json.dumps({"grid": grid, "answer": grid[q["cat"]][sol[q["cat_idx"]].index(q["val_idx"])]})
