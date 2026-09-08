"""Zebra puzzle generator + solver (Python port of zebra-puzzle.html).

Uses the same mulberry32 PRNG and the same algorithm order as the HTML page,
so a seed string like "5h4p-m-b50737b4" produces the identical puzzle in both.

Representation
--------------
A puzzle has N houses in a row and M property categories, each category holding
N distinct values (one per house). Internally:

    sol[c][h]        value index of category c in house h (the full solution)
    masks[c][v]      bitmask of houses value v of category c could still occupy;
                     bit h set means "house h+1" (bit 0 = house 1)

Clue kinds (each constrains one or two category values):
    pos   "X is in house h"                - fix one value to a house
    same  "X is in the same house as Y"
    diff  "X is not in the same house as Y"
    imm   "X is directly to the left of Y" - exactly one house apart
    adj   "X is next to Y"                 - one house apart, either side
    ord   "X is somewhere to the left of Y"

A clue dict looks like {"t": <kind>, "a": [cat, val], "b": [cat, val] | None,
"h": <house>} where "h" is only present on "pos" clues.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

DEFAULT_CATS: List[Dict] = [
    {"name": "Color",       "values": ["Red", "Green", "Blue", "Yellow", "White", "Purple", "Orange",
                                       "Black", "Pink", "Brown", "Gray", "Cyan", "Magenta", "Lime", "Teal", "Gold"]},
    {"name": "Nationality", "values": ["Brit", "Swede", "Dane", "German", "Norwegian", "Spaniard", "Italian",
                                       "Finn", "Pole", "Greek", "Russian", "Frenchman", "Brazilian", "Japanese", "Egyptian", "Mexican"]},
    {"name": "Drink",       "values": ["Tea", "Coffee", "Milk", "Beer", "Water", "Juice", "Wine",
                                       "Cola", "Cider", "Lemonade", "Smoothie", "Cocoa", "Brandy", "Rum", "Punch", "Shake"]},
    {"name": "Pet",         "values": ["Dog", "Bird", "Cat", "Horse", "Fish", "Rabbit", "Turtle",
                                       "Snake", "Hamster", "Parrot", "Goat", "Duck", "Ferret", "Canary", "Lizard", "Pig"]},
    {"name": "Snack",       "values": ["Chips", "Cookies", "Popcorn", "Pretzels", "Nuts", "Candy", "Fruit",
                                       "Crackers", "Donut", "Muffin", "Cheese", "Chocolate", "Granola", "Jerky", "Waffles", "Toast"]},
    {"name": "Hobby",       "values": ["Chess", "Painting", "Running", "Gardening", "Reading", "Cooking", "Photography",
                                       "Fishing", "Swimming", "Singing", "Knitting", "Writing", "Hiking", "Cycling", "Pottery", "Drawing"]},
    {"name": "Job",         "values": ["Doctor", "Teacher", "Chef", "Pilot", "Artist", "Farmer", "Engineer",
                                       "Lawyer", "Nurse", "Writer", "Baker", "Dentist", "Mechanic", "Astronomer", "Librarian", "Barber"]},
    {"name": "Flower",      "values": ["Rose", "Tulip", "Lily", "Daisy", "Orchid", "Iris", "Poppy",
                                       "Violet", "Sunflower", "Jasmine", "Lavender", "Marigold", "Peony", "Carnation", "Daffodil", "Azalea"]},
    {"name": "Sport",       "values": ["Soccer", "Tennis", "Golf", "Rugby", "Cricket", "Boxing", "Skiing", "Rowing",
                                       "Archery", "Volleyball", "Hockey", "Judo", "Curling", "Surfing", "Fencing", "Bowling"]},
    {"name": "Instrument",  "values": ["Piano", "Violin", "Guitar", "Drums", "Flute", "Trumpet", "Cello", "Clarinet",
                                       "Harp", "Saxophone", "Viola", "Banjo", "Accordion", "Harpsichord", "Organ", "Tuba"]},
    {"name": "Car",         "values": ["Ford", "Toyota", "Honda", "BMW", "Audi", "Volvo", "Mazda", "Kia",
                                       "Fiat", "Jeep", "Nissan", "Porsche", "Tesla", "Saab", "Renault", "Buick"]},
    {"name": "Fruit",       "values": ["Apple", "Pear", "Peach", "Plum", "Cherry", "Mango", "Kiwi", "Melon",
                                       "Apricot", "Fig", "Grape", "Lemon", "Banana", "Papaya", "Guava", "Olive"]},
    {"name": "Animal",      "values": ["Lion", "Tiger", "Bear", "Wolf", "Fox", "Deer", "Eagle", "Shark",
                                       "Whale", "Otter", "Badger", "Falcon", "Panther", "Moose", "Beaver", "Hawk"]},
    {"name": "Gemstone",    "values": ["Diamond", "Ruby", "Sapphire", "Emerald", "Opal", "Topaz", "Jade", "Pearl",
                                       "Amber", "Garnet", "Onyx", "Quartz", "Jasper", "Peridot", "Zircon", "Beryl"]},
    {"name": "Tree",        "values": ["Oak", "Pine", "Birch", "Maple", "Cedar", "Willow", "Elm", "Ash",
                                       "Beech", "Spruce", "Fir", "Aspen", "Redwood", "Sycamore", "Chestnut", "Walnut"]},
    {"name": "Dessert",     "values": ["Cake", "Pie", "Sundae", "Brownie", "Tart", "Pudding", "Custard", "Sorbet",
                                       "Macaron", "Cheesecake", "Cupcake", "Tiramisu", "Flan", "Strudel", "Baklava", "Fudge"]},
]

# Difficulty letter used in seed codes -> full difficulty name.
DIFFS = {"e": "easy", "m": "medium", "h": "hard"}

M32 = 0xFFFFFFFF  # keeps JS-number arithmetic in unsigned 32-bit range


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
    """Fisher-Yates shuffle in place (consumes rnd exactly like the JS code)."""
    for i in range(len(arr) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr


# --------------------------------------------------------------- solver ----
def popcount(x: int) -> int:
    """Number of set bits = number of remaining candidates in a mask."""
    return bin(x).count("1")


def low_idx(x: int) -> int:
    """Index of the lowest set bit."""
    return (x & -x).bit_length() - 1


def high_idx(x: int) -> int:
    """Index of the highest set bit."""
    return x.bit_length() - 1


def init_masks(num_cats: int, num_houses: int) -> List[List[int]]:
    """Initially every value of every category can be in any house."""
    all_houses = (1 << num_houses) - 1
    return [[all_houses] * num_houses for _ in range(num_cats)]


def apply_clue(masks: List[List[int]], clue: Dict, num_houses: int) -> int:
    """Narrow the candidate masks touched by one clue, in place.

    Returns the number of masks changed, or -1 on contradiction (an empty
    candidate set or an impossible clue relation).
    """
    all_houses = (1 << num_houses) - 1
    a_cat, a_val = clue["a"]
    a = masks[a_cat][a_val]
    b_cat = b_val = None
    b = None
    if clue.get("b") is not None:
        b_cat, b_val = clue["b"]
        b = masks[b_cat][b_val]
    new_a, new_b = a, b
    kind = clue["t"]

    if kind == "pos":
        new_a = a & (1 << clue["h"])
    elif kind == "same":
        new_a = a & b
        new_b = new_a
    elif kind == "diff":
        if popcount(a) == 1:
            new_b = b & ~a
        if popcount(new_b) == 1:
            new_a = a & ~new_b
    elif kind == "imm":
        # b sits exactly one house to the right of a
        new_b = b & ((a << 1) & all_houses)
        new_a = a & (new_b >> 1)
    elif kind == "adj":
        # b sits one house to the left or right of a
        new_b = b & (((a << 1) | (a >> 1)) & all_houses)
        new_a = a & (((new_b << 1) | (new_b >> 1)) & all_houses)
    elif kind == "ord":
        if a == 0 or b == 0:
            return -1
        # a must sit below b's best case; b must sit above a's worst case
        new_a = a & ((1 << high_idx(b)) - 1)
        if new_a == 0:
            return -1
        new_b = b & ~((1 << (low_idx(new_a) + 1)) - 1)

    if new_a == 0 or new_b == 0:
        return -1
    changed = 0
    if new_a != a:
        masks[a_cat][a_val] = new_a
        changed += 1
    if new_b is not None and new_b != b:
        masks[b_cat][b_val] = new_b
        changed += 1
    return changed


def apply_row_rules(masks: List[List[int]], cat: int, num_houses: int) -> int:
    """Permutation-row reductions for one category's row of masks, in place.

    Returns the number of masks changed, or -1 on contradiction. Three classic
    rules for "each house holds exactly one value of this category":
      - naked single: a value with one candidate house loses it elsewhere
      - hidden single: a house only one value can occupy belongs to that value
      - Hall interval: k values confined to a k-house interval claim it whole,
        so every other value must avoid the interval
    """
    row = masks[cat]
    changed = 0

    for val in range(num_houses):
        if row[val] == 0:
            return -1
        if popcount(row[val]) != 1:
            continue
        for other in range(num_houses):
            if other != val and (row[other] & row[val]):
                row[other] &= ~row[val]
                if row[other] == 0:
                    return -1
                changed += 1

    for house in range(num_houses):
        bit = 1 << house
        fits = [val for val in range(num_houses) if row[val] & bit]
        if not fits:
            return -1
        if len(fits) == 1 and row[fits[0]] != bit:
            row[fits[0]] = bit
            changed += 1

    if num_houses > 3:
        for lo in range(num_houses):
            span = 0
            for hi in range(lo, num_houses):
                span |= 1 << hi
                size = hi - lo + 1
                confined = sum(1 for val in range(num_houses)
                               if (row[val] & ~span) == 0)
                if confined == size and size < num_houses:
                    for val in range(num_houses):
                        if row[val] & ~span:  # not one of the confined values
                            trimmed = row[val] & ~span
                            if trimmed and trimmed != row[val]:
                                row[val] = trimmed
                                changed += 1
    return changed


def propagate(masks: List[List[int]], clues: List[Dict], num_houses: int) -> bool:
    """Run all clue and row rules until a fixed point.

    Returns False on contradiction; otherwise masks hold the narrowed candidates.
    """
    num_cats = len(masks)
    changed = True
    while changed:
        changed = False
        for clue in clues:
            n = apply_clue(masks, clue, num_houses)
            if n < 0:
                return False
            changed = changed or n > 0
        for cat in range(num_cats):
            n = apply_row_rules(masks, cat, num_houses)
            if n < 0:
                return False
            changed = changed or n > 0
    return True


def propagation_unique(num_cats: int, num_houses: int, clues: List[Dict]) -> Optional[bool]:
    """Cheap uniqueness pre-check.

    Returns True if constraint propagation alone fully determines the grid
    (sound: propagation only eliminates impossible values, so a complete
    singleton fixed point is the one and only solution). Returns None when
    propagation is inconclusive and a real search is needed; False means the
    clues are contradictory.
    """
    masks = init_masks(num_cats, num_houses)
    if not propagate(masks, clues, num_houses):
        return False
    if all(popcount(mask) == 1 for row in masks for mask in row):
        return True
    return None


def solve(num_cats: int, num_houses: int, clues: List[Dict], limit: int = 2) -> List[List[List[int]]]:
    """Search for solutions; returns up to `limit` of them as sol[cat][house]."""
    solutions: List[List[List[int]]] = []

    def search(masks: List[List[int]]) -> None:
        if len(solutions) >= limit:
            return
        if not propagate(masks, clues, num_houses):
            return
        # branch on the cell with the fewest remaining candidates
        best_cat = best_val = -1
        best_count = num_houses + 1
        for cat in range(num_cats):
            for val in range(num_houses):
                count = popcount(masks[cat][val])
                if 1 < count < best_count:
                    best_count, best_cat, best_val = count, cat, val
        if best_cat < 0:  # every cell is a singleton: record the solution
            solution = []
            for cat in range(num_cats):
                by_house = [0] * num_houses
                for val in range(num_houses):
                    by_house[low_idx(masks[cat][val])] = val
                solution.append(by_house)
            solutions.append(solution)
            return
        remaining = masks[best_cat][best_val]
        while remaining and len(solutions) < limit:
            bit = remaining & -remaining
            remaining ^= bit
            branch_masks = [row[:] for row in masks]
            branch_masks[best_cat][best_val] = bit
            search(branch_masks)

    search(init_masks(num_cats, num_houses))
    return solutions


# ------------------------------------------------------------ generator ----
def house_of(sol, cat: int, val: int) -> int:
    """House index where category `cat` has value `val` in solution `sol`."""
    return sol[cat].index(val)


# Per-clue-kind probability of offering that kind to the puzzle, per difficulty.
WEIGHTS = {
    "easy":   {"pos": 1.0,  "same": 1.0, "imm": 1.0, "adj": 0.8, "ord": 0.6, "diff": 0.15},
    "medium": {"pos": 0.35, "same": 0.8, "imm": 0.8, "adj": 0.8, "ord": 0.7, "diff": 0.3},
    "hard":   {"pos": 0.10, "same": 0.5, "imm": 0.5, "adj": 0.7, "ord": 0.8, "diff": 0.5},
}


def build_pool(sol, num_cats: int, num_houses: int, rnd, difficulty: str) -> List[Dict]:
    """Every possible clue about `sol`, kept with per-difficulty probability.

    Positional clues are kept separately and appended after the shuffled main
    pool (this order matters: the generator adds clues from the front until the
    puzzle is unique, so positional clues act as a guaranteed backfill).
    """
    pool: List[Dict] = []
    positional_clues: List[Dict] = []
    for cat in range(num_cats):
        for val in range(num_houses):
            positional_clues.append({"t": "pos", "a": [cat, val], "b": None,
                                     "h": house_of(sol, cat, val)})
    for cat_a in range(num_cats):
        for val_a in range(num_houses):
            house_a = house_of(sol, cat_a, val_a)
            for cat_b in range(num_cats):
                if cat_b == cat_a:
                    continue
                for val_b in range(num_houses):
                    house_b = house_of(sol, cat_b, val_b)
                    a, b = [cat_a, val_a], [cat_b, val_b]
                    if house_a == house_b:
                        if cat_a < cat_b:  # keep each unordered pair once
                            pool.append({"t": "same", "a": a, "b": b})
                    else:
                        if house_b == house_a + 1:
                            pool.append({"t": "imm", "a": a, "b": b})
                        if abs(house_a - house_b) == 1 and cat_a < cat_b:
                            pool.append({"t": "adj", "a": a, "b": b})
                        if house_a < house_b:
                            pool.append({"t": "ord", "a": a, "b": b})
                        if cat_a < cat_b:
                            pool.append({"t": "diff", "a": a, "b": b})
    weights = WEIGHTS.get(difficulty, WEIGHTS["medium"])
    selected = [clue for clue in pool if rnd() < weights[clue["t"]]]
    selected_positional = [c for c in positional_clues if rnd() < weights["pos"]]
    pair_clues = shuffle(selected + selected_positional, rnd)
    return pair_clues + shuffle(list(positional_clues), rnd)


def is_unique(num_cats: int, num_houses: int, clues: List[Dict]) -> bool:
    """True when the clues admit exactly one solution."""
    fast = propagation_unique(num_cats, num_houses, clues)
    if fast is not None:
        return fast
    return len(solve(num_cats, num_houses, clues, 2)) == 1


def generate(num_cats: int, num_houses: int, difficulty: str,
             rnd: Callable[[], float]) -> Optional[Dict]:
    """Draw a random solution, then add clues until it is the only one.

    Returns None if the pool runs out before the clues become unique.
    """
    sol = [shuffle(list(range(num_houses)), rnd) for _ in range(num_cats)]
    pool = build_pool(sol, num_cats, num_houses, rnd, difficulty)
    clues: List[Dict] = []
    for clue in pool:
        clues.append(clue)
        if len(clues) < max(3, num_cats):
            continue
        if is_unique(num_cats, num_houses, clues):
            break
    if not is_unique(num_cats, num_houses, clues):
        return None
    # greedily drop clues that the puzzle still solves without
    removal_order = shuffle(list(range(len(clues))), rnd)
    redundant = set()
    for i in removal_order:
        trial_clues = [c for j, c in enumerate(clues) if j != i and j not in redundant]
        if is_unique(num_cats, num_houses, trial_clues):
            redundant.add(i)
    kept_clues = [c for j, c in enumerate(clues) if j not in redundant]
    return {"sol": sol, "clues": shuffle(kept_clues, rnd)}


def make_question(sol, cats, num_houses: int, rnd) -> Dict:
    """Ask for one house's value in one category via another category."""
    num_cats = len(cats)
    asked_cat = int(rnd() * num_cats)
    given_cat = int(rnd() * num_cats)
    while given_cat == asked_cat:
        given_cat = int(rnd() * num_cats)
    house_idx = int(rnd() * num_houses)
    given_val = sol[given_cat][house_idx]
    answer_val = sol[asked_cat][house_idx]
    given, asked = cats[given_cat], cats[asked_cat]
    about_people = re.search(r"nation|people|person|name|who", asked["name"], re.I)
    if about_people:
        text = f"Who has {given['values'][given_val]} ({given['name']})?"
    else:
        text = (f"Which {asked['name']} belongs to the house with "
                f"{given['values'][given_val]} ({given['name']})?")
    return {"text": text, "cat": asked["name"], "cat_idx": asked_cat,
            "val_idx": answer_val, "answer": asked["values"][answer_val]}


def clue_text(clue: Dict, cats) -> str:
    """Human-readable sentence for a clue dict."""
    def item(x):
        cat, val = x
        return f"{cats[cat]['values'][val]} ({cats[cat]['name']})"

    a = item(clue["a"])
    b = item(clue["b"]) if clue.get("b") else ""
    return {
        "pos":  f"{a} is in house {clue.get('h', -1) + 1}.",
        "same": f"{a} is in the same house as {b}.",
        "diff": f"{a} is not in the same house as {b}.",
        "imm":  f"{a} is directly to the left of {b}.",
        "adj":  f"{a} is next to {b}.",
        "ord":  f"{a} is somewhere to the left of {b}.",
    }[clue["t"]]


# ------------------------------------------------------------ seed codes ---
def seed_code(num_houses: int, num_cats: int, difficulty: str, num: int) -> str:
    """Seed string like "5h4p-m-b50737b4": N houses, M properties, difficulty, RNG seed."""
    return f"{num_houses}h{num_cats}p-{difficulty[0]}-{num & M32:08x}"


def parse_seed(code: str) -> Optional[Dict]:
    """Inverse of seed_code; returns None for malformed codes."""
    match = re.fullmatch(r"(\d+)h(\d+)p-([emh])-([0-9a-fA-F]{1,8})", code.strip())
    if not match:
        return None
    return {"N": int(match.group(1)), "M": int(match.group(2)),
            "difficulty": DIFFS[match.group(3).lower()],
            "num": int(match.group(4), 16)}


def _pad_values(name: str, values: List[str], n: int) -> List[str]:
    """Extend a value list to length n with deterministic placeholder names."""
    return (values + [f"{name} {i + 1}" for i in range(len(values), n)])[:n]


def build_puzzle(N: int, M: int, difficulty: str = "medium", num: int = 0,
                 cats: Optional[List[Dict]] = None) -> Dict:
    """Full puzzle for the given seed number. Mirrors the HTML page exactly.

    N = number of houses, M = number of property categories.
    """
    cats = cats or DEFAULT_CATS
    cats = [{"name": c["name"], "values": _pad_values(c["name"], c["values"], N)}
            for c in cats[:M]]
    cats += [{"name": f"Property {i + 1}",
              "values": [f"Value {j + 1}" for j in range(N)]}
             for i in range(len(cats), M)]
    rnd = make_rng(num)
    result = None
    attempts = 0
    while result is None and attempts < 5:
        result = generate(M, N, difficulty, rnd)
        attempts += 1
    if result is None:
        raise RuntimeError("generation failed")
    question = make_question(result["sol"], cats, N, rnd)
    return {
        "seed": seed_code(N, M, difficulty, num),
        "N": N, "M": M, "difficulty": difficulty,
        "cats": cats,
        "sol": result["sol"],
        "clues": result["clues"],
        "clue_texts": [clue_text(c, cats) for c in result["clues"]],
        "question": question,
        "solution_grid": {c["name"]: [c["values"][result["sol"][ci][h]] for h in range(N)]
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
    """The exact prompt sent to the model for puzzle `p`."""
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
    """Case/punctuation-insensitive key for comparing answers."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def extract_json(text: str) -> Optional[Dict]:
    """Last complete top-level JSON object in the text containing a "grid" key."""
    text = re.sub(r"```(?:json)?", "", text)
    brace_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(brace_positions):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:end + 1])
                        if isinstance(obj, dict) and "grid" in obj:
                            return obj
                    except Exception:
                        pass
                    break
    return None


def grade(p: Dict, reply: str) -> Dict:
    """Score a model reply against the puzzle's known solution."""
    reply_json = extract_json(reply)
    num_cells = p["N"] * p["M"]
    if not reply_json or not isinstance(reply_json.get("grid"), dict):
        return {"parsed": False, "cells_correct": 0, "cells_total": num_cells,
                "cell_acc": 0.0, "grid_correct": False, "answer_correct": False,
                "correct": False}
    expected_grid = p["solution_grid"]
    submitted_grid = {_norm(k): v for k, v in reply_json["grid"].items()}
    correct_cells = 0
    for name, expected_vals in expected_grid.items():
        submitted_row = submitted_grid.get(_norm(name))
        if not isinstance(submitted_row, list):
            continue
        for house, expected in enumerate(expected_vals):
            if house < len(submitted_row) and _norm(submitted_row[house]) == _norm(expected):
                correct_cells += 1
    grid_ok = correct_cells == num_cells
    answer_ok = _norm(reply_json.get("answer", "")) == _norm(p["question"]["answer"])
    return {"parsed": True, "cells_correct": correct_cells, "cells_total": num_cells,
            "cell_acc": correct_cells / num_cells, "grid_correct": grid_ok,
            "answer_correct": answer_ok, "correct": grid_ok and answer_ok}


CODE_PATTERNS = re.compile(
    r"```(?:python|py|js|javascript|c\+\+|java|prolog)|"
    r"\bimport itertools\b|\bfrom itertools\b|\bdef solve\b|\bfor perm in\b|"
    r"\bpython_repl\b|\bexec\(|\bprint\(",
    re.I)


def looks_like_code(reply: str) -> bool:
    """Heuristic: did the model answer by writing code despite the rules?"""
    return bool(CODE_PATTERNS.search(reply))


def solve_puzzle_answer(p: Dict) -> str:
    """Reference answer produced by the built-in solver (used by mock models)."""
    solutions = solve(p["M"], p["N"], p["clues"], 2)
    sol = solutions[0]
    grid = {c["name"]: [c["values"][sol[ci][h]] for h in range(p["N"])]
            for ci, c in enumerate(p["cats"])}
    question = p["question"]
    return json.dumps({"grid": grid,
                       "answer": grid[question["cat"]][sol[question["cat_idx"]].index(question["val_idx"])]})
