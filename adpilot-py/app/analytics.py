"""Deterministic analytics: metrics, anomaly detection, root cause, opportunity score, budget optimizer."""
import numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest

METRICS = ["roas", "cpc", "ctr", "cvr"]
LABEL = {"roas": "ROAS", "cpc": "CPC", "ctr": "CTR", "cvr": "Conversion rate"}
WEIGHTS = {"roas": .25, "margin": .20, "inv": .15, "trend": .15, "cvr": .10, "cpa": .10, "stab": .05}


def enrich(daily, camps, prod):
    """Unified table: campaign -> platform -> SKU -> spend/sales/margin, with all KPI formulas."""
    df = daily.merge(camps.reset_index()[["cid", "pid", "platform"]], on="cid").merge(
        prod.reset_index()[["pid", "price", "cost"]], on="pid")
    df["clicks"] = df.spend / df.cpc
    df["impr"] = df.clicks / df.ctr
    df["conv"] = df.clicks * df.cvr
    df["revenue"] = df.conv * df.price
    df["profit"] = df.conv * (df.price - df.cost) - df.spend
    df["roas"] = df.revenue / df.spend
    return df


def window(df, start, end=0, by="cid"):
    """Aggregate days in (last-start, last-end]. window(df,3)=last 3 days, window(df,15,1)=14 days before today."""
    last = df.day.max()
    s = df[(df.day > last - start) & (df.day <= last - end)]
    g = s.groupby(by)[["spend", "revenue", "profit", "clicks", "impr", "conv"]].sum()
    g["roas"] = g.revenue / g.spend
    g["cpc"] = g.spend / g.clicks
    g["ctr"] = g.clicks / g.impr
    g["cvr"] = g.conv / g.clicks
    g["cpa"] = g.spend / g.conv
    return g


def cover(E):
    vel = window(E.df, 3, by="pid").conv / 3
    return E.prod.stock / vel.reindex(E.prod.index).clip(lower=1)


def detect(E):
    """Rolling 14-day mean/std z-score (|z|>3) + Isolation Forest corroboration on relative deviations."""
    df, out, now = E.df, [], datetime.now().strftime("%H:%M:%S")
    last = df.day.max()
    base = df[(df.day < last) & (df.day >= last - 14)]
    cur = df[df.day == last].set_index("cid")
    mu = base.groupby("cid")[METRICS].mean()
    sd = base.groupby("cid")[METRICS].std(ddof=0)
    rel = base[METRICS].values / mu.loc[base.cid, METRICS].values - 1
    iso = IsolationForest(n_estimators=100, random_state=0).fit(rel)
    flag = iso.predict(cur[METRICS].loc[mu.index].values / mu.values - 1) == -1
    iso_flag = dict(zip(mu.index, flag))
    for cid in mu.index:
        for m in METRICS:
            m0, s0 = mu.loc[cid, m], max(sd.loc[cid, m], .03 * mu.loc[cid, m])
            z, v = (cur.loc[cid, m] - m0) / s0, cur.loc[cid, m]
            if abs(z) > 3:
                pct = v / m0 - 1
                out.append(dict(cid=cid, name=E.camps.name[cid], metric=m, label=LABEL[m], previous=float(m0),
                                current=float(v), pct=float(pct), z=float(z), status="NORMAL → ANOMALY",
                                good=bool(z * (-1 if m == "cpc" else 1) > 0), iforest=bool(iso_flag[cid]),
                                severity="High" if abs(pct) > .3 else "Medium" if abs(pct) > .15 else "Low", time=now))
    cv = cover(E)
    for pid, c in cv.items():
        if c < 8:
            out.append(dict(pid=int(pid), name=E.prod.name[pid], metric="inventory", label="Inventory days",
                            previous=float(E.prod.cover0[pid]), current=float(c), pct=float(c / E.prod.cover0[pid] - 1),
                            z=0.0, status="NORMAL → ANOMALY", good=False, iforest=False,
                            severity="High" if c < 5 else "Medium", time=now))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(out, key=lambda a: (order[a["severity"]], -abs(a["pct"])))


def rca(E, cid):
    """ROAS = CVR x price / CPC -> log-decomposition into conversion vs CPC; CTR/inventory/margin checked as related dims."""
    b, n = window(E.df, 15, 1).loc[cid], window(E.df, 1).loc[cid]
    pid = E.camps.pid[cid]
    dc, dp = np.log(n.cvr / b.cvr), -np.log(n.cpc / b.cpc)
    tot = abs(dc) + abs(dp) or 1e-9
    ct, cv = n.ctr / b.ctr - 1, cover(E)[pid]
    items = [("Conversion rate", n.cvr / b.cvr - 1, abs(dc) / tot), ("CPC", n.cpc / b.cpc - 1, abs(dp) / tot),
             ("Creative CTR", ct, abs(ct) * .5), ("Inventory cover", None, .1 if cv < 7 else .02),
             ("Price / margin", 0.0, .01)]
    out = []
    for name, ch, w in items:
        txt = (f"{name} {'increased' if ch > 0 else 'decreased'} {abs(ch):.0%}" if ch not in (None, 0.0) else
               f"Inventory cover {cv:.1f} days for {E.prod.name[pid]}" if ch is None else
               f"Price/margin unchanged ({E.prod.margin[pid]:.0%} margin)")
        out.append(dict(cause=txt, weight=float(w), impact="HIGH" if w >= .45 else "MEDIUM" if w >= .15 else "LOW"))
    return sorted(out, key=lambda x: -x["weight"])


def opportunities(E):
    """Weighted 0-100 score (+ learned per-platform boost from feedback)."""
    df, c = E.df, E.camps
    w3, w14 = window(df, 3).reindex(c.index), window(df, 17, 3).reindex(c.index)
    last = df.day.max()
    r = df[(df.day < last) & (df.day >= last - 14)].groupby("cid").roas.agg(["mean", "std"]).reindex(c.index)
    cov = cover(E).loc[c.pid].values
    f = pd.DataFrame({"roas": w3.roas / 5, "margin": E.prod.margin.loc[c.pid].values / .7, "inv": cov / 14,
                      "trend": (w3.roas / w14.roas - .6) / .8, "cvr": w3.cvr / .05, "cpa": 1 - (w3.cpa - 150) / 700,
                      "stab": 1 - r["std"] / r["mean"] * 5}, index=c.index).clip(0, 1)
    score = sum(f[k] * w for k, w in WEIGHTS.items()) * 100 + c.platform.map(E.boost)
    o = f.copy()
    o["score"] = score.clip(0, 100).round().astype(int)
    o["roas3"], o["cvr3"], o["cpc3"], o["cpc14"] = w3.roas, w3.cvr, w3.cpc, w14.cpc
    o["cover"] = cov
    return o


def tier(s):
    return "High" if s >= 62 else "Medium" if s >= 45 else "Low"


def params(E, bad):
    """Per-campaign response curve: revenue(s) = roas * s0 * (s/s0)^0.7 (diminishing returns)."""
    c, df = E.camps, E.df
    o, w3, w1 = opportunities(E), window(df, 3).reindex(c.index), window(df, 1).reindex(c.index)
    r = np.where(c.index.isin(bad), w1.roas, w3.roas)
    n = c.groupby("pid").size().loc[c.pid].values
    stk = E.prod.stock.loc[c.pid].values / (5 * n)          # 5-day inventory cover cap, shared by campaigns on SKU
    conv = np.maximum(w3.conv.values / 3, 1)
    s0 = c.budget.values
    return dict(ids=list(c.index), s0=s0, r=r, m=E.prod.margin.loc[c.pid].values, score=o.score.values,
                f=.6 + o.score.values / 250, cap=np.minimum(s0 * 2.2, s0 * (stk / conv) ** (1 / .7)))


rev = lambda q, i, s: q["r"][i] * q["s0"][i] * (s / q["s0"][i]) ** .7
prof = lambda q, i, s: rev(q, i, s) * q["m"][i] - s


def optimize(E, total, bad=()):
    """Greedy marginal-profit allocation in Rs1000 steps; optimizes PROFIT, risk-discounted, inventory-capped."""
    q = params(E, bad)
    a = np.minimum(q["s0"] * .3, q["cap"])
    left = total - a.sum()
    while left >= 1000:
        g = .7 * q["r"] * q["m"] * q["f"] * ((a + 500) / q["s0"]) ** -.3 - 1
        g[a + 1000 > q["cap"]] = -1
        i = int(np.argmax(g))
        if g[i] <= 0:
            break
        a[i] += 1000
        left -= 1000
    return q, a, max(float(left), 0.0)
