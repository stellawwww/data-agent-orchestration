# AGENTS.md

Routing table for coding agents working in this repo. It holds no project details; it says where to look.

| Task | Read first |
|---|---|
| Change world assumptions (seasonality, flu sensitivity, market sizes) | `config/` |
| Regenerate the synthetic data | `data_gen/` (`fetch_flu.py` once, then `python -m data_gen.generate [--world validation\|test]`, then `checks.py` for dev) |
| Understand the tables | `data/DATA_DICTIONARY.md` |
| Add or edit a planted scenario | dev: `evals/scenarios.yaml`; held-out worlds: `data_gen/inject.py`; then regenerate |
| Change what gets flagged as an anomaly | `agents/detector.py` (rule numbers: `evals/alert_budget.py`) |
| Change how an agent reasons | that agent's file in `skills/` |
| Change the graph (nodes, edges, retries) | `agents/` |
| Score runs | `evals/` |
| Rebuild the portfolio page | `docs/` |

## Rules

- Never read anything in `evals/`, `worlds/*/answers/` or `config/` from inside an agent. Those are
  answer keys (`scenarios.yaml`, `effect_sizes.csv`, `noise_cases.csv`) and the generator's hidden
  assumptions. Agents read one world's data folder only (`data/` or `worlds/<name>/data/`),
  starting with its `DATA_DICTIONARY.md`.
- Three worlds, like train / validation / test:
  - **dev** (`data/`, `evals/`): look freely; design, debug, demo.
  - **validation** (`worlds/validation/`): compare versions; read aggregate scores only, don't study
    individual injected events. If a validation number drives a design change, log it.
  - **test** (`worlds/test/`): sealed. Nobody, human or coding agent, opens `worlds/test/answers/` or
    prints its events before the final scoring run. The scoring code refuses unless `FINAL_SCORING=1`.
- Never commit `.env` or any key file.
- Use no real company data. Only CDC flu data is real; everything else is synthetic.
