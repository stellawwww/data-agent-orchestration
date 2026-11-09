# AGENTS.md

Routing table for coding agents working in this repo. It holds no project details; it says where to look.

| Task | Read first |
|---|---|
| Change world assumptions (seasonality, flu sensitivity, market sizes) | `config/` |
| Regenerate the synthetic data | `data_gen/` |
| Add or edit a planted scenario | `evals/scenarios.yaml`, then regenerate data |
| Change what gets flagged as an anomaly | `sql/` |
| Change how an agent reasons | that agent's file in `skills/` |
| Change the graph (nodes, edges, retries) | `agents/` |
| Score runs | `evals/` |
| Rebuild the portfolio page | `docs/` |

## Rules

- Never read `evals/scenarios.yaml` from inside an agent. It is the answer key.
- Never commit `.env` or any key file.
- Use no real company data. Only CDC flu data is real; everything else is synthetic.
