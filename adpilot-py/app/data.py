"""Seed data layer: simulated Meta/Google/Amazon/TikTok ads + products/inventory.
Swap `generate()` for real API connectors later; everything downstream uses these 3 tables."""
import numpy as np, pandas as pd

PLATFORMS = ["Meta", "Google", "Amazon", "TikTok"]
BASE_CPC = [14, 22, 18, 11]
AUDIENCES = ["Lookalike 1%", "Retargeting", "Broad 25-40", "Interest: Wellness", "Cart Abandoners"]
CREATIVES = ["UGC Video A", "Carousel Bestsellers", "Founder Story", "Before/After", "Offer Banner",
             "Unboxing", "Testimonial", "Static Hero", "Reel Hook", "Comparison"]
# name, price, unit cost, days of stock cover at start
PRODUCTS = [("Glow Serum", 1299, 390, 30), ("Hydra Cream", 899, 520, 28), ("Sun Shield", 599, 390, 35),
            ("Hair Oil", 499, 150, 40), ("Protein Bar Box", 1199, 780, 26), ("Yoga Mat", 1599, 700, 7),
            ("Smart Bottle", 1999, 1300, 14), ("Face Wash", 349, 110, 33), ("Vitamin C Gummies", 799, 240, 30),
            ("Body Lotion", 649, 300, 22)]


def generate(seed: int = 7):
    rng = np.random.default_rng(seed)
    nz = lambda a: 1 + rng.uniform(-a, a)
    prod = pd.DataFrame(PRODUCTS, columns=["name", "price", "cost", "cover0"])
    prod.index.name = "pid"
    prod["sku"] = [f"SKU-{101 + i}" for i in prod.index]
    prod["margin"] = (prod.price - prod.cost) / prod.price
    rows = []
    for i in range(20):
        q = i % 4
        rows.append(dict(cid=f"C{i + 1:02d}", platform=PLATFORMS[q], pid=i % 10,
                         audience=AUDIENCES[(i * 3) % 5], creative=CREATIVES[(i * 3 + 1) % 10],
                         base_cpc=BASE_CPC[q] * rng.uniform(.8, 1.3), base_ctr=rng.uniform(.008, .025),
                         base_cvr=rng.uniform(.022, .067), budget=float((15 + rng.integers(0, 30)) * 1000),
                         trend=rng.uniform(-.004, .004)))
    camps = pd.DataFrame(rows).set_index("cid")
    camps["base_budget"] = camps.budget
    for k in ("cvr", "cpc", "ctr"):          # persistent scenario multipliers (decay after actions)
        camps["mod_" + k] = 1.0
    camps["name"] = [f"{c.platform} · {prod.name[c.pid]} · {c.audience}" for c in camps.itertuples()]
    d = []
    for cid, c in camps.iterrows():
        for day in range(30):
            d.append(dict(day=day, cid=cid, spend=c.budget * nz(.05), cpc=c.base_cpc * nz(.06),
                          ctr=c.base_ctr * nz(.06), cvr=c.base_cvr * (1 + c.trend * (day - 29)) * nz(.06)))
    return prod, camps, pd.DataFrame(d)
