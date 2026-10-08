# AdPilot — Autonomous D2C Advertising Intelligence & Decision Engine (Python)

Closed-loop decision system for D2C brands: **detect → diagnose → score → decide → approve → simulate → measure → learn**.
No paid APIs or credentials: a seeded simulation layer (`app/data.py`) stands in for Meta/Google/Amazon/TikTok, orders and inventory.

## Architecture
```
app/data.py       seed data: 10 products, 20 campaigns, 4 platforms, 10 creatives, 30 days (swap for real connectors)
app/analytics.py  KPI formulas, anomaly detection, root cause, opportunity score, budget optimizer
app/engine.py     state, demo scenarios, recommendations, approve/execute, outcome + learning, assistant
app/main.py       FastAPI routes + serves the UI
static/index.html UI (Decision Center, Overview, Budget Optimizer, Data Explorer, Decision History)
tests/test_flow.py  end-to-end check without a web server
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m tests.test_flow                               # optional: verify the full loop in the terminal
uvicorn app.main:app --reload                           # open http://localhost:8000
```

## Demo (judges' flow)
1. Open **Decision Center** (healthy; only a low-level Yoga Mat inventory watch).
2. Demo Mode → **1. Meta campaign ROAS drops** → anomaly (ROAS ≈ -42%), root cause (conversion rate ≈ -31% HIGH, CPC MEDIUM).
3. Recommendation: move budget from the failing campaign to the best-scoring one; click **Why?** for the explanation.
4. **Approve & Execute** → budgets change, next day simulated, positive outcome recorded, confidence rises (72% → 76%).
5. **Decision History** shows expected vs actual profit and model learning. Other scenarios: Demo Mode 2–5.

## Methodology
- **Metrics**: spend, revenue, profit = conv x (price - cost) - spend, ROAS, CTR, CPC, CPA, CVR, AOV, cover days = stock / 3-day units per day.
- **Anomaly detection**: today vs rolling 14-day mean/std per campaign and metric; flag |z| > 3 (std floored at 3% of mean). An Isolation Forest on relative deviations adds corroboration. Each record has metric, previous, current, % change, severity, entity, timestamp. Inventory anomaly when cover < 8 days.
- **Root cause**: ROAS = CVR x price / CPC, so log-changes of CVR and CPC give each one's share of the ROAS move. CTR, inventory cover and margin are checked as related dimensions. Ranked HIGH/MEDIUM/LOW from actual data.
- **Opportunity score (0-100)**: ROAS 25%, margin 20%, inventory 15%, 7-day trend 15%, conversion 10%, CPA 10%, stability 5%, plus a learned per-platform boost.
- **Budget optimizer**: response curve revenue(s) = ROAS x s0 x (s/s0)^0.7 (diminishing returns). Greedy Rs1,000 steps by marginal PROFIT (not revenue), risk-discounted by score, capped by a 5-day inventory limit, 30% floor spend. Budget not profitably usable is "held back".
- **Closed loop**: approval → budgets updated → next day simulated (CPC rises with spend, scenario shocks half-decay) → profit before/after → positive/negative outcome → model confidence ±, platform score boost → future recommendation confidence and scores change.

## API
`GET /api/dashboard|campaigns|products|anomalies|recommendations|opportunities|decisions|feedback`,
`GET /api/anomalies/{cid}/root-cause`, `POST /api/recommendations/{id}/approve|execute`,
`POST /api/demo/scenario {"scenario":0-4}`, `POST /api/demo/reset`, `POST /api/optimizer {"total":n}`, `POST /api/assistant {"question":"..."}`.
Interactive docs: `/docs`.

## Limitations / future work
Execution is simulated only; state is in memory (one process); real ad-API connectors, a learned conversion model (e.g. gradient boosting), per-SKU shared-inventory optimization, persistent storage and an optional LLM layer for narration are natural next steps.
