# Zebra Bench

Zebra Bench is a logic-reasoning benchmark for large language models. It generates zebra puzzles with exactly one solution, sends each puzzle in a fresh context, and measures how far a model progresses through this ladder:

**3x3 → 4x4 → 5x5 → 6x6 → 7x7**

Each level is written as **houses x property categories**. For example, a 7x7 puzzle has seven houses, seven property categories, 42 grid cells, and roughly 35–47 clues.

> This repository is purely **"vibecoded"**.

## Features

- Deterministic, uniquely solvable zebra puzzles.
- Compatible seed generation with the companion `zebra-puzzle.html` implementation.
- Fresh context for every attempt: no conversation history or few-shot examples.
- OpenAI-compatible `/chat/completions` API support.
- Local web runner and manual grading mode.
- Offline mock modes for testing the pipeline.
- SQLite caching of puzzles and results.
- An interactive, standalone HTML report and optional static charts.

The model prompt forbids code, scripts, tools, and solver encodings. The optional `--strict-no-code` setting disqualifies replies that match the project's code-detection heuristic.

## Requirements

Python 3.9+ is enough for the benchmark, local runner, SQLite database, and interactive report; those parts use only the standard library.

Static PNG charts require `matplotlib`:

```bash
python -m pip install matplotlib
```

The repository includes `pyproject.toml` and `uv.lock` for reproducible `uv` environments:

```bash
uv sync
uv sync --extra plots
```

## Repository layout

| File | Purpose |
| --- | --- |
| `zebra.py` | Puzzle generator, reference solver, prompt renderer, response parser, and grader. |
| `bench.py` | Command-line benchmark runner. |
| `serve.py` | Local web UI for running benchmarks and grading manual replies. |
| `db.py` | SQLite storage for puzzles, results, runs, providers, and models. |
| `report.py` | Dependency-free, self-contained interactive HTML dashboard. |
| `plots.py` | Static PNG charts and a CSV summary; requires `matplotlib`. |
| `pyproject.toml` / `uv.lock` | Project metadata and locked `uv` environment. |

## Quick start

Test the pipeline without network access or an API key:

```bash
python bench.py --mock perfect
python report.py results.jsonl
```

Run a real model with the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
python bench.py --model gpt-4o-mini --attempts 3
```

For newer OpenAI reasoning models:

```bash
python bench.py --model YOUR_MODEL \
  --tokens-param max_completion_tokens --no-temperature
```

Use another OpenAI-compatible provider or local endpoint:

```bash
export OPENROUTER_API_KEY=...
python bench.py \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --model anthropic/claude-sonnet-4.5 \
  --model openai/gpt-5 \
  --attempts 5 --parallel 3
```

Print a puzzle prompt and its known answer without calling an API:

```bash
python bench.py --dry-run --levels 3x3
```

## Local web runner

Start the local UI:

```bash
python serve.py
```

Open `http://127.0.0.1:8000`, select a provider, enter an API key and model names, and choose **Run benchmark**. Progress streams as attempts complete, and the dashboard refreshes after each level.

Manual mode supports models without an API:

1. Choose the model name, level, and difficulty, or provide a seed code.
2. Generate and copy the prompt.
3. Submit it in any chat interface.
4. Paste the reply into the UI to grade and save it.

The server listens on loopback by default. API keys are sent only to the local Python process, which makes the provider request; the browser never contacts the provider directly. Do not bind this server to an untrusted network.

Use the **Mock** provider to exercise the UI without a key or network connection. **Start fresh** clears previous results, and **Stop** requests that an active UI run halt.

## Benchmark behavior

By default, Zebra Bench runs three puzzles per level and requires a 0.67 fully-correct pass rate to advance. The ladder stops at the first level a model fails.

The puzzle plan is derived from the base seed only, so every model in the same run receives the identical puzzle for each level and attempt. Stored puzzles are reused deterministically when available, while fresh puzzles are generated in the background and added to the local cache.

| Option | Meaning |
| --- | --- |
| `--model NAME` | Model to test; repeat to test multiple models. |
| `--levels 3x3,4x4,5x5` | Override the default ladder. |
| `--attempts N` | Puzzles per level; default: `3`. |
| `--pass-ratio R` | Fully-correct pass rate needed to advance; default: `0.67`. |
| `--difficulty easy\|medium\|hard` | Puzzle clue density; default: `medium`. |
| `--seed VALUE` | Base seed for reproducible, comparable puzzle plans. |
| `--parallel N` | Concurrent API requests per model and level. |
| `--timeout SECONDS` | Request timeout; default: `600`. |
| `--max-tokens N` | Completion-token limit; default: `16000`. |
| `--tokens-param NAME` | Provider token field, such as `max_completion_tokens`. |
| `--no-temperature` | Omit the temperature field. |
| `--strict-no-code` | Disqualify code-flagged replies. |
| `--save-replies` | Save full replies in `results.jsonl`. |
| `--db PATH` | SQLite database path; use `--db ""` to disable storage. |
| `--mock perfect\|noisy\|dumb` | Offline pipeline test mode. |

## Answer format and grading

The model must end its reply with one JSON object:

```json
{
  "grid": {
    "Color": ["Red", "..."],
    "Nationality": ["...", "..."]
  },
  "answer": "Swede"
}
```

Every property list is ordered by house number, beginning with house 1. The grader normalizes property names, capitalization, whitespace, and punctuation.

| Field | Meaning |
| --- | --- |
| `correct` | The full grid and final answer are correct; this decides pass/fail. |
| `cell_acc` | Fraction of grid cells that are correct. |
| `answer_correct` | The answer to the final question is correct. |
| `parsed` | A valid JSON object containing `grid` was found. |
| `code_flag` | The reply matched the code-like response heuristic. |
| `disqualified` | Strict no-code mode turned a flagged reply into a failure. |

Code detection is heuristic, not proof. Disable provider-level tool use or code interpreters when you intend to test the no-code constraint.

## Results and storage

Each attempt is appended to `results.jsonl`. Records include model, level, seed, grading data, timing, token counts when available, and request errors. Full replies are included only with `--save-replies`.

Unless disabled, the project also stores data in `bench.db`:

| Table | Contents |
| --- | --- |
| `puzzles` | Generated puzzles, prompts, clues, questions, and solutions. |
| `results` | Graded attempts and their JSONL-compatible payloads. |
| `runs` | CLI and web-run configuration, excluding API keys. |
| `providers` | Provider URLs, settings, and saved API keys. |
| `models` | Provider-associated model names and use counts. |

API keys are stored in plain text in `bench.db`. Treat it like an `.env` file: keep it local and never commit it.

```bash
python db.py stats
python db.py providers
python db.py import results.jsonl
```

On first start with an empty database, `serve.py` imports existing JSONL history automatically.

## Reports

Build a standalone interactive dashboard:

```bash
python report.py results.jsonl --open
```

The default output is `charts/report.html`. It includes a leaderboard, per-level solve rate and cell accuracy, latency and token views, sortable results, model filters, tooltips, and light/dark themes. It uses no CDN or runtime dependency.

Build static charts and a CSV summary:

```bash
python plots.py results.jsonl
```

This creates `01_pass_rate.png`, `02_cell_accuracy.png`, `03_max_level.png`, `04_cost.png`, and `summary.csv` in `charts/`.

Generated results and reports are intentionally not included in the repository. Run the benchmark to produce your own data.

## Reproducibility

Puzzle seed codes look like `5h4p-m-b50737b4`: houses, properties, difficulty, and a hexadecimal seed. The generator is deterministic, and the same code recreates the same puzzle. Use `--seed` to reproduce a full benchmark plan and compare models fairly.

Mock modes are for verifying the benchmark pipeline and reports; they are not model evaluations.
