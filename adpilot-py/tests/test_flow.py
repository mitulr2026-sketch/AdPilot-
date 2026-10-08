"""End-to-end check of the closed loop (no web server needed): python -m tests.test_flow"""
from app.engine import Engine

e = Engine()
print("baseline anomalies:", [(a["label"], a.get("cid") or a["name"]) for a in e.anomalies])
for n in range(5):
    e.reset(); e.scenario(n)
    print(n, e.SCENARIOS[n], "->", len(e.anomalies), "anomalies;", [r["id"] for r in e.recs])
e.reset(); e.scenario(0)
top = e.anomalies[0]; print("TOP:", top["cid"], top["label"], f"{top['pct']:+.0%}", "iforest:", top["iforest"])
print("RCA:", [(x["cause"], x["impact"]) for x in e.rca(top["cid"])[:3]])
r = e.recs[0]; print("REC:", r["title"], round(r["expected_profit"]), f"conf={r['confidence']:.0%}")
e.approve(r["id"]); fb = e.execute(r["id"])
print("OUTCOME:", fb["result"], fb["before"], fb["after"], f"conf {fb['confidence_before']:.0%}->{fb['confidence_after']:.0%}")
print("optimizer:", {k: v for k, v in e.optimizer().items() if k != "allocation"})
print("ask:", e.ask("why did roas drop"))
d = e.dashboard(); print("dashboard ok", round(d["kpis"]["roas"], 2), len(e.campaigns()), len(e.products()))
