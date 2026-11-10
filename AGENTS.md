# AGENTS.md

Routing table for coding agents working in this repo. It holds no project details; it says where to look.

| Task | Read first |
|---|---|
| Change world assumptions (seasonality, flu sensitivity, market sizes) | `config/` |
| Regenerate the synthetic data | `data_gen/` (`fetch_flu.py` once, then `generate.py`, then `checks.py`) |
| Understand the tables | `data/DATA_DICTIONARY.md` |
| Add or edit a planted scenario | `evals/scenarios.yaml`, then regenerate data |
| Change what gets flagged as an anomaly | `sql/` |
| Change how an agent reasons | that agent's file in `skills/` |
| Change the graph (nodes, edges, retries) | `agents/` |
| Score runs | `evals/` |
| Rebuild the portfolio page | `docs/` |

## Rules

- Never read anything in `evals/` or `config/` from inside an agent. `evals/` is the answer key
  (`scenarios.yaml`, `effect_sizes.csv`, `noise_cases.csv`) and `config/` holds the generator's
  hidden assumptions. Agents may read `data/` only, starting with `data/DATA_DICTIONARY.md`.
- Never commit `.env` or any key file.
- Use no real company data. Only CDC flu data is real; everything else is synthetic.
