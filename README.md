# Data Agent Orchestration

A multi-agent analyst that does the job of a weekly business review: when a marketplace KPI moves, it works out why.

> Status: in progress. This README is a placeholder until the first replay runs are done.

## What it does

Each run replays one past week of a two-sided healthcare booking marketplace (patients searching and booking, providers offering appointment slots). A SQL detector flags unusual metric movements. A LangGraph team of agents then decomposes each movement through the KPI tree, checks supply, funnel, data quality and outside factors, and writes a WBR-style memo. A skeptic agent rejects any claim that isn't backed by the data before the memo is written.

## Data

- **Outside world (real):** regional flu activity from CDC ILINet.
- **Inside the company (synthetic):** daily searches, funnel steps, bookings and provider supply for 5 markets × 6 specialties × 3 platforms over 2 years.
- **Planted scenarios:** supply loss, a tracking bug, a product change, a marketing campaign, plus decoys and normal seasonality. Each has a known answer, so every run can be scored automatically. The evaluation design follows InsightBench's planted-insight method.

Table and column definitions: [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

![Real CDC flu activity vs synthetic urgent care demand](docs/story_check.png)

Synthetic urgent care demand follows each market's real flu curve (weekly correlation 0.89 to 0.95). The 2025–26 season peaked about six weeks earlier than the one before: more severe in NYC and Atlanta, milder in Chicago, Houston and Phoenix.

### Rebuild the data

```bash
conda create -n dao python=3.12 && conda activate dao
pip install -r requirements.txt
python data_gen/fetch_flu.py   # download CDC flu data (snapshot already in data/)
python -m data_gen.generate    # write the dev world's tables to data/
python data_gen/checks.py      # verify the data tells the intended story; redraw the chart
python -m data_gen.generate --world validation   # held-out worlds in worlds/<name>/
python -m data_gen.generate --world test
```

The generator is deterministic (fixed seed). Every planted effect is measured against a counterfactual world with identical noise and only that effect removed, so the answer key records each effect's true size (`evals/effect_sizes.csv`).

### Three worlds: train / validation / test

The detector and the agents are designed while looking at the dev world, so dev scores can't be trusted on their own. Two held-out worlds keep the evaluation honest. Each has the same assumptions, calendar and real flu data as dev, but new noise and about 30 randomly injected events (random type, size, scope, timing; `data_gen/inject.py`):

| World | Folder | Used for |
|---|---|---|
| dev | `data/`, `evals/` | Design and debugging; looked at freely |
| validation | `worlds/validation/` | Comparing versions; aggregate scores only |
| test | `worlds/test/` | Sealed; scored once, at the end (`FINAL_SCORING=1`) |

Thresholds are picked without answers, from the false-alarm rate in quiet weeks (`evals/alert_budget.py`).

The detector runs weekly, like the review it serves. On validation it catches most supply, demand and tracking events but none of the small (2–3%) or short (a few days) funnel bugs: averaged over a week they sit below normal noise. That floor is by design. Catching them is the job of daily engineering monitoring, not a weekly business review.

## Stack

LangGraph · Claude API (Haiku for routing, Sonnet for judgment) · TruLens · DuckDB · Python

## Repo layout

| Folder | Contents |
|---|---|
| `config/` | World assumptions: seasonality, flu sensitivity, market sizes |
| `data_gen/` | Synthetic data generator and CDC download |
| `data/` | Generated tables and the CDC snapshot |
| `worlds/` | Held-out worlds (validation, test): data and sealed answers |
| `agents/` | Data layer, SQL detector, one module per graph node |
| `skills/` | Instructions each agent reads |
| `evals/` | Dev answer key (`scenarios.yaml`) and scoring |
| `runs/` | Memos and exported traces from each replay |
| `docs/` | Portfolio page |

## License

MIT
