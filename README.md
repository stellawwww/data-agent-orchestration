# Data Agent Orchestration

A multi-agent analyst that does the job of a weekly business review: when a marketplace KPI moves, it works out why.

> Status: in progress. This README is a placeholder until the first replay runs are done.

## What it does

Each run replays one past week of a two-sided healthcare booking marketplace (patients searching and booking, providers offering appointment slots). A SQL detector flags unusual metric movements. A LangGraph team of agents then decomposes each movement through the KPI tree, checks supply, funnel, data quality and outside factors, and writes a WBR-style memo. A skeptic agent rejects any claim that isn't backed by the data before the memo is written.

## Data

- **Outside world (real):** regional flu activity from CDC ILINet.
- **Inside the company (synthetic):** daily searches, funnel steps, bookings and provider supply for 5 markets × 6 specialties × 3 platforms over 2 years.
- **Planted scenarios:** supply loss, a tracking bug, a product change, a marketing campaign, plus decoys and normal seasonality. Each has a known answer, so every run can be scored automatically. The evaluation design follows InsightBench's planted-insight method.

## Stack

LangGraph · Claude API (Haiku for routing, Sonnet for judgment) · TruLens · Snowflake · Python

## Repo layout

| Folder | Contents |
|---|---|
| `config/` | World assumptions: seasonality, flu sensitivity, market sizes |
| `data_gen/` | Synthetic data generator and CDC download |
| `data/` | Generated tables and the CDC snapshot |
| `sql/` | Detector queries |
| `agents/` | One module per graph node |
| `skills/` | Instructions each agent reads |
| `evals/` | Answer key (`scenarios.yaml`) and scoring |
| `runs/` | Memos and exported traces from each replay |
| `docs/` | Portfolio page |

## License

MIT
