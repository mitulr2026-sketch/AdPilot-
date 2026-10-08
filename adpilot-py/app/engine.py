"""Decision engine: state, scenarios, recommendations, approval, simulated execution, feedback learning."""
import numpy as np, pandas as pd
from datetime import datetime
from . import analytics as A
from .data import generate, PLATFORMS


class Engine:
    def __init__(self, seed=7):
        self.reset(seed)

    # ------------------------------------------------------------------ state
    def reset(self, seed=7):
        self.seed, self.rng = seed, np.random.default_rng(seed + 1)
        self.prod, self.camps, self.daily = generate(seed)
        self.fb, self.decisions, self.approved = [], [], {}
        self.conf, self.boost, self.message = .72, {p: 0 for p in PLATFORMS}, ""
        df = A.enrich(self.daily, self.camps, self.prod)
        self.prod["stock"] = (df.groupby("pid").conv.sum() / 30 * self.prod.cover0).round()
        self.refresh()

    @property
    def df(self):
        return A.enrich(self.daily, self.camps, self.prod)

    def refresh(self):
        self.anomalies = A.detect(self)
        self.bad = {a["cid"] for a in self.anomalies if a.get("cid") and not a["good"] and a["metric"] == "roas"}
        self.opps = A.opportunities(self)
        self.recs = self._recommend()

    # ------------------------------------------------------------- scenarios
    SCENARIOS = ["Meta campaign ROAS drops", "Google conversion rate surges", "Top product nears stockout",
                 "High-margin product outperforms", "CPC suddenly spikes (Amazon)"]

    def _mod(self, idx, **m):
        cid = self.camps.index[idx]
        last = self.daily.day.max()
        for k, v in m.items():
            self.camps.loc[cid, "mod_" + k] *= v
            self.daily.loc[(self.daily.cid == cid) & (self.daily.day == last), k] *= v

    def scenario(self, n):
        if n == 0: self._mod(4, cvr=.65, cpc=1.2)
        elif n == 1: self._mod(1, cvr=1.5)
        elif n == 2:
            vel = A.window(self.df, 3, by="pid").conv / 3
            self.prod.loc[3, "stock"] = round(vel[3] * 2.2)
        elif n == 3:
            for i in (7, 17): self._mod(i, cvr=1.4, ctr=1.35)
        elif n == 4: self._mod(2, cpc=1.6)
        else: raise ValueError("scenario must be 0-4")
        self.message = ""
        self.refresh()
        return self.SCENARIOS[n]

    # --------------------------------------------------------- recommendations
    def _recommend(self):
        c = self.camps
        total = float(c.budget.sum())
        q, a, _ = A.optimize(self, total, self.bad)
        d = a - q["s0"]
        ri = int(np.argmax(d))
        bad = sorted([x for x in self.anomalies if x.get("cid") in self.bad and x["metric"] == "roas"], key=lambda x: x["pct"])
        di = list(c.index).index(bad[0]["cid"]) if bad else int(np.argmin(d))
        out = []
        amt = 0 if di == ri else int(min(q["s0"][di] * .4, max(d[ri], 0), q["cap"][ri] - q["s0"][ri]) // 1000 * 1000)
        if amt >= 1000:
            dR = A.rev(q, ri, q["s0"][ri] + amt) - A.rev(q, ri, q["s0"][ri]) + A.rev(q, di, q["s0"][di] - amt) - A.rev(q, di, q["s0"][di])
            dP = A.prof(q, ri, q["s0"][ri] + amt) - A.prof(q, ri, q["s0"][ri]) + A.prof(q, di, q["s0"][di] - amt) - A.prof(q, di, q["s0"][di])
            f_id, t_id, o = c.index[di], c.index[ri], self.opps
            t = o.loc[t_id]
            pos = [f"{k}: {t[k]:.0%} of max" for k in A.WEIGHTS if t[k] > .6]
            neg = ["Diminishing returns: extra spend converts at a lower marginal ROAS"]
            if t.cpc3 / t.cpc14 > 1.05: neg.insert(0, f"CPC up {t.cpc3 / t.cpc14 - 1:.0%} recently")
            if t.cover < 10: neg.append(f"Inventory cover only {t.cover:.1f} days")
            out.append(dict(id=f"shift-{f_id}-{t_id}-{amt}", kind="shift", title=f"Move ₹{amt:,} from {f_id} → {t_id}",
                            from_cid=f_id, to_cid=t_id, amount=amt, from_name=c.name[f_id], to_name=c.name[t_id],
                            expected_revenue=float(dR), expected_profit=float(dP),
                            confidence=float(np.clip(.35 + .4 * t.score / 100 + .25 * self.conf, 0, .97)),
                            why=dict(data_used=["Ad platform metrics", "Orders/revenue", "Margins", "Inventory", "14-day history"],
                                     key_metrics=dict(target_roas=float(t.roas3), target_score=int(t.score), source_score=int(o.score[f_id]),
                                                      target_margin=float(self.prod.margin[c.pid[t_id]])),
                                     positives=pos, risks=neg,
                                     constraints=["Budget-neutral", "Min 30% of current spend kept", "5-day inventory cover cap",
                                                  "Optimizes profit, not revenue"],
                                     anomaly_driver=bad[0]["label"] + f" {bad[0]['pct']:+.0%}" if bad else "Lowest marginal return")))
        cv = A.cover(self)
        for pid, days in cv.items():
            if days < 5:
                out.append(dict(id=f"protect-{pid}", kind="protect", title=f"Protect inventory: reduce {self.prod.name[pid]} exposure 15%",
                                pid=int(pid), cids=list(c.index[c.pid == pid]), expected_revenue=0.0, expected_profit=0.0,
                                confidence=float(np.clip(.6 + .25 * self.conf, 0, .95)),
                                why=dict(data_used=["Inventory", "Sales velocity"], key_metrics=dict(days_left=float(days)),
                                         positives=[f"Only {days:.1f} days of stock at current velocity",
                                                    f"Cutting exposure 15% stretches cover to ~{days / .85 ** .7:.1f} days"],
                                         risks=["Small short-term revenue loss"], constraints=["Never advertise past supply"],
                                         anomaly_driver="Stockout risk")))
        return out

    # ------------------------------------------------------ approve / execute
    def approve(self, rid):
        rec = next((r for r in self.recs if r["id"] == rid), None)
        if not rec: raise KeyError(rid)
        self.approved[rid] = rec
        self.decisions.append(dict(id=rid, title=rec["title"], status="approved", time=datetime.now().strftime("%H:%M:%S")))
        return rec

    def execute(self, rid):
        rec = self.approved.get(rid)
        if not rec: raise PermissionError("Recommendation must be approved before execution")
        if rec["kind"] == "shift":
            changes = {rec["from_cid"]: self.camps.budget[rec["from_cid"]] - rec["amount"],
                       rec["to_cid"]: self.camps.budget[rec["to_cid"]] + rec["amount"]}
            inv, plat = list(changes), self.camps.platform[rec["to_cid"]]
        else:
            changes = {cid: self.camps.budget[cid] * .85 for cid in rec["cids"]}
            inv, plat = rec["cids"], None
        fb = self._run(rec, changes, inv, plat)
        for d in self.decisions:
            if d["id"] == rid and d["status"] == "approved": d["status"] = "executed"
        del self.approved[rid]
        self.refresh()
        return fb

    def _snap(self, inv):
        w = A.window(self.df, 1).loc[inv]
        return dict(roas=float(w.revenue.sum() / w.spend.sum()), profit=float(w.profit.sum()))

    def _run(self, rec, changes, inv, plat):
        before, c0 = self._snap(inv), self.conf
        for cid, v in changes.items(): self.camps.loc[cid, "budget"] = float(v)
        self._tick()
        after = self._snap(inv)
        d = after["profit"] - before["profit"]
        pos = d > 0
        self.conf = float(np.clip(c0 + (.04 if pos else -.05), .3, .97))   # closed-loop learning
        if plat: self.boost[plat] += 2 if pos else -3
        fb = dict(n=len(self.fb) + 1, time=datetime.now().strftime("%H:%M:%S"), decision=rec["title"], before=before, after=after,
                  expected_profit=rec["expected_profit"], actual_profit_delta=float(d), result="Positive" if pos else "Negative",
                  confidence_before=c0, confidence_after=self.conf)
        self.fb.insert(0, fb)
        self.message = f"Executed in simulation layer. {fb['result']} outcome recorded. Confidence {c0:.0%} → {self.conf:.0%}."
        return fb

    def _tick(self):
        """Simulate the next day: budgets drive spend; CPC rises with spend (elasticity .3); scenario shocks half-decay."""
        last, rows = int(self.daily.day.max()), []
        for cid in self.camps.index:
            for k in ("cvr", "cpc", "ctr"): self.camps.loc[cid, "mod_" + k] **= .5
            c, nz = self.camps.loc[cid], lambda a: 1 + self.rng.uniform(-a, a)
            rows.append(dict(day=last + 1, cid=cid, spend=c.budget * nz(.04),
                             cpc=c.base_cpc * c.mod_cpc * (c.budget / c.base_budget) ** .3 * nz(.06),
                             ctr=c.base_ctr * c.mod_ctr * nz(.06), cvr=c.base_cvr * c.mod_cvr * nz(.06)))
        self.daily = pd.concat([self.daily, pd.DataFrame(rows)], ignore_index=True)
        self.daily = self.daily[self.daily.day > last + 1 - 30].reset_index(drop=True)
        sold = self.df.query("day == @last + 1").groupby("pid").conv.sum()
        self.prod["stock"] = (self.prod.stock - sold.reindex(self.prod.index).fillna(0)).clip(lower=0)

    # ------------------------------------------------------------------ views
    def dashboard(self):
        df, cur, prev = self.df, A.window(self.df, 7), A.window(self.df, 14, 7)
        k = lambda g: dict(spend=g.spend.sum(), revenue=g.revenue.sum(), profit=g.profit.sum(),
                           roas=g.revenue.sum() / g.spend.sum(), cvr=g.conv.sum() / g.clicks.sum())
        kc, kp = k(cur), k(prev)
        series = df.groupby("day")[["spend", "revenue", "profit"]].sum().reset_index()
        plat = A.window(df, 7, by="platform")
        prof = A.window(df, 7, by="pid").profit.rename(index=self.prod.name)
        return dict(kpis=kc, previous=kp, series=series.to_dict("records"), roas_by_platform=plat.roas.round(2).to_dict(),
                    profit_by_product=prof.round(0).to_dict(),
                    inventory_risk=int((A.cover(self) < 8).sum()), monitoring=f"AI monitoring {len(self.camps)} campaigns",
                    anomalies=self.anomalies[:5], recommendations=self.recs, top_opportunities=self.opportunities()[:5],
                    confidence=self.conf, message=self.message,
                    pipeline=dict(anomalies=len(self.anomalies), recommendations=len(self.recs), decisions=len(self.fb)))

    def campaigns(self):
        g, o = A.window(self.df, 7), self.opps
        t = self.camps.join(g).join(o[["score"]])
        t["margin"] = self.prod.margin.loc[t.pid].values
        t["cover"] = A.cover(self).loc[t.pid].values
        t["tier"] = t.score.map(A.tier)
        return t.reset_index().replace([np.inf, -np.inf], 0).fillna(0).round(4).to_dict("records")

    def products(self):
        vel = (A.window(self.df, 3, by="pid").conv / 3).reindex(self.prod.index)
        p = self.prod.copy()
        p["velocity"], p["cover_days"] = vel, A.cover(self)
        p["profit_7d"] = A.window(self.df, 7, by="pid").profit
        p["stockout_risk"] = p.cover_days.map(lambda c: "High" if c < 5 else "Medium" if c < 8 else "Low")
        return p.reset_index().round(3).fillna(0).to_dict("records")

    def opportunities(self):
        o = self.opps.join(self.camps[["name", "platform"]]).reset_index().sort_values("score", ascending=False)
        o["tier"] = o.score.map(A.tier)
        return o.round(3).fillna(0).to_dict("records")

    def optimizer(self, total=None):
        total = total or float(self.camps.budget.sum())
        q, a, held = A.optimize(self, total, self.bad)
        n = len(a)
        p0, p1 = sum(A.prof(q, i, q["s0"][i]) for i in range(n)), sum(A.prof(q, i, a[i]) for i in range(n))
        rows = [dict(cid=q["ids"][i], name=self.camps.name[q["ids"][i]], current=float(q["s0"][i]), recommended=float(a[i]),
                     delta=float(a[i] - q["s0"][i])) for i in range(n)]
        return dict(total=total, held_back=held, profit_now=float(p0), profit_optimized=float(p1), allocation=rows)

    def rca(self, cid): return A.rca(self, cid)

    def ask(self, q):
        """
        Natural-language assistant for the AdPilot dashboard.

        The assistant uses the current Engine state and calculates
        answers dynamically from campaigns, products, anomalies,
        recommendations, inventory, and performance metrics.
        """

        question = str(q).strip()
        text = question.lower()

        if not question:
            return "Please ask me something about your campaigns, products, budget, sales, or performance."

        # -----------------------------------------------------
        # Helpers
        # -----------------------------------------------------

        def money(value):
            return f"₹{float(value):,.0f}"

        def percent(value):
            return f"{float(value):.1%}"

        def number(value):
            return f"{float(value):,.0f}"

        # -----------------------------------------------------
        # Current data
        # -----------------------------------------------------

        df = self.df

        today = A.window(df, 1)

        last_7 = A.window(df, 7)

        campaign_7 = A.window(
            df,
            7,
            by="cid"
        )

        product_7 = A.window(
            df,
            7,
            by="pid"
        )

        # -----------------------------------------------------
        # Greetings
        # -----------------------------------------------------

        greetings = {
            "hi",
            "hello",
            "hey",
            "yo",
            "hii",
            "helo",
            "good morning",
            "good afternoon",
            "good evening",
        }

        if text in greetings:
            return (
                "Hey! 👋 I'm your AdPilot assistant. "
                "Ask me about campaign performance, ROAS, profit, "
                "budget allocation, inventory, anomalies, or recommendations."
            )

        # -----------------------------------------------------
        # Help / capabilities
        # -----------------------------------------------------

        if (
            "what can you do" in text
            or "what do you do" in text
            or "help" in text
            or "capabilities" in text
        ):
            return (
                "I can analyze your current AdPilot data. You can ask me things like:\n"
                "• Which campaign is performing best?\n"
                "• Which campaign should I reduce?\n"
                "• What is my ROAS?\n"
                "• Which product has low inventory?\n"
                "• Which campaign is most profitable?\n"
                "• Why did performance drop?\n"
                "• Where should I invest more budget?\n"
                "• What are today's sales and spend?"
            )

        # -----------------------------------------------------
        # Specific campaign name detection
        # -----------------------------------------------------

        campaign_id = None

        for cid in self.camps.index:
            name = str(self.camps.loc[cid, "name"]).lower()

            if str(cid).lower() in text or name in text:
                campaign_id = cid
                break

        # -----------------------------------------------------
        # Specific campaign performance
        # -----------------------------------------------------

        if campaign_id is not None:

            campaign_name = self.camps.loc[
                campaign_id,
                "name"
            ]

            if campaign_id in campaign_7.index:

                row = campaign_7.loc[campaign_id]

                spend = float(row.spend)
                revenue = float(row.revenue)
                profit = float(row.profit)

                roas = (
                    revenue / spend
                    if spend
                    else 0
                )

                clicks = float(row.clicks)
                conversions = float(row.conv)

                cvr = (
                    conversions / clicks
                    if clicks
                    else 0
                )

                if any(
                    word in text
                    for word in (
                        "performance",
                        "doing",
                        "performing",
                        "metrics",
                        "stats",
                    )
                ):
                    return (
                        f"{campaign_name} — last 7 days:\n"
                        f"Spend: {money(spend)}\n"
                        f"Revenue: {money(revenue)}\n"
                        f"Profit: {money(profit)}\n"
                        f"ROAS: {roas:.2f}x\n"
                        f"Conversion rate: {percent(cvr)}"
                    )

                if "roas" in text:
                    return (
                        f"{campaign_name} has a 7-day ROAS of "
                        f"{roas:.2f}x."
                    )

                if "profit" in text:
                    return (
                        f"{campaign_name} generated "
                        f"{money(profit)} profit in the last 7 days."
                    )

                if (
                    "spend" in text
                    or "spent" in text
                ):
                    return (
                        f"{campaign_name} spent "
                        f"{money(spend)} in the last 7 days."
                    )

                if (
                    "revenue" in text
                    or "sales" in text
                ):
                    return (
                        f"{campaign_name} generated "
                        f"{money(revenue)} revenue in the last 7 days."
                    )

        # -----------------------------------------------------
        # Overall ROAS
        # -----------------------------------------------------

        if (
            "roas" in text
            or "return on ad spend" in text
        ):

            spend = float(last_7.spend.sum())
            revenue = float(last_7.revenue.sum())

            roas = revenue / spend if spend else 0

            return (
                f"Your overall ROAS for the last 7 days is "
                f"{roas:.2f}x.\n"
                f"Spend: {money(spend)}\n"
                f"Revenue: {money(revenue)}"
            )

        # -----------------------------------------------------
        # Today's numbers
        # -----------------------------------------------------

        if (
            "today" in text
            and any(
                word in text
                for word in (
                    "spend",
                    "revenue",
                    "sales",
                    "profit",
                    "performance",
                )
            )
        ):

            spend = float(today.spend.sum())
            revenue = float(today.revenue.sum())
            profit = float(today.profit.sum())

            roas = revenue / spend if spend else 0

            return (
                f"Today's performance:\n"
                f"Spend: {money(spend)}\n"
                f"Revenue: {money(revenue)}\n"
                f"Profit: {money(profit)}\n"
                f"ROAS: {roas:.2f}x"
            )

        # -----------------------------------------------------
        # Overall spend
        # -----------------------------------------------------

        if (
            "spend" in text
            or "spent" in text
            or "spending" in text
        ):

            spend = float(last_7.spend.sum())

            return (
                f"Your total ad spend over the last 7 days is "
                f"{money(spend)}."
            )

        # -----------------------------------------------------
        # Overall revenue
        # -----------------------------------------------------

        if (
            "revenue" in text
            or "sales" in text
            or "turnover" in text
        ):

            revenue = float(last_7.revenue.sum())

            return (
                f"Your total revenue over the last 7 days is "
                f"{money(revenue)}."
            )

        # -----------------------------------------------------
        # Overall profit
        # -----------------------------------------------------

        if (
            "profit" in text
            or "profitable" in text
            or "earnings" in text
        ):

            total_profit = float(last_7.profit.sum())

            return (
                f"Your total profit over the last 7 days is "
                f"{money(total_profit)}."
            )

        # -----------------------------------------------------
        # Best campaign
        # -----------------------------------------------------

        if (
            "best campaign" in text
            or "top campaign" in text
            or "best performing campaign" in text
            or "performing best" in text
        ):

            if campaign_7.empty:
                return "I don't have enough campaign data to determine the best campaign."

            scores = campaign_7.copy()

            scores["roas"] = (
                scores.revenue
                / scores.spend.replace(0, np.nan)
            )

            best_id = scores.roas.idxmax()

            best = scores.loc[best_id]

            return (
                f"The best-performing campaign by 7-day ROAS is "
                f"{self.camps.loc[best_id, 'name']}.\n"
                f"ROAS: {best.roas:.2f}x\n"
                f"Revenue: {money(best.revenue)}\n"
                f"Profit: {money(best.profit)}"
            )

        # -----------------------------------------------------
        # Worst campaign
        # -----------------------------------------------------

        if (
            "worst campaign" in text
            or "weakest campaign" in text
            or "bad campaign" in text
            or "underperforming campaign" in text
        ):

            if campaign_7.empty:
                return "I don't have enough campaign data to determine the weakest campaign."

            scores = campaign_7.copy()

            scores["roas"] = (
                scores.revenue
                / scores.spend.replace(0, np.nan)
            )

            worst_id = scores.roas.idxmin()

            worst = scores.loc[worst_id]

            return (
                f"The weakest campaign by 7-day ROAS is "
                f"{self.camps.loc[worst_id, 'name']}.\n"
                f"ROAS: {worst.roas:.2f}x\n"
                f"Revenue: {money(worst.revenue)}\n"
                f"Profit: {money(worst.profit)}"
            )

        # -----------------------------------------------------
        # Inventory / stock
        # -----------------------------------------------------

        if (
            "stock" in text
            or "inventory" in text
            or "stockout" in text
            or "stock out" in text
        ):

            cover = A.cover(self)

            risky = [
                (pid, days)
                for pid, days in cover.items()
                if days < 8
            ]

            risky.sort(key=lambda x: x[1])

            if not risky:
                return (
                    "Inventory looks healthy. "
                    "No product currently has less than 8 days of cover."
                )

            lines = []

            for pid, days in risky[:5]:
                product_name = self.prod.name.loc[pid]

                risk = (
                    "HIGH"
                    if days < 5
                    else "MEDIUM"
                )

                lines.append(
                    f"{product_name}: {days:.1f} days ({risk} risk)"
                )

            return (
                "These products need inventory attention:\n"
                + "\n".join(lines)
            )

        # -----------------------------------------------------
        # Best product
        # -----------------------------------------------------

        if (
            "best product" in text
            or "top product" in text
            or "most profitable product" in text
        ):

            if product_7.empty:
                return "I don't have enough product data."

            best_id = product_7.profit.idxmax()

            return (
                f"The most profitable product over the last 7 days is "
                f"{self.prod.name.loc[best_id]} "
                f"with {money(product_7.loc[best_id, 'profit'])} profit."
            )

        # -----------------------------------------------------
        # Recommendations
        # -----------------------------------------------------

        if (
            "recommend" in text
            or "recommendation" in text
            or "what should i do" in text
            or "what should we do" in text
            or "action" in text
        ):

            if not self.recs:
                return "There are no active recommendations right now."

            rec = self.recs[0]

            response = (
                f"My top recommendation is: {rec['title']}.\n"
                f"Confidence: {rec['confidence']:.0%}."
            )

            if "expected_profit" in rec:
                response += (
                    f"\nExpected profit impact: "
                    f"{money(rec['expected_profit'])}/day."
                )

            return response

        # -----------------------------------------------------
        # Budget / investment opportunities
        # -----------------------------------------------------

        if (
            "budget" in text
            or "invest" in text
            or "scale" in text
            or "increase budget" in text
            or "where should i put" in text
        ):

            opportunities = self.opportunities()[:3]

            if not opportunities:
                return "I don't currently see any strong budget opportunities."

            lines = []

            for opportunity in opportunities:
                lines.append(
                    f"{opportunity['name']} "
                    f"({opportunity['cid']}) — "
                    f"score {opportunity['score']:.1f}"
                )

            return (
                "My strongest budget opportunities are:\n"
                + "\n".join(lines)
            )

        # -----------------------------------------------------
        # Anomalies
        # -----------------------------------------------------

        if (
            "anomal" in text
            or "problem" in text
            or "issue" in text
            or "wrong" in text
        ):

            active = [
                a
                for a in self.anomalies
                if not a.get("good")
            ]

            if not active:
                return "I don't currently detect any major performance anomalies."

            lines = []

            for anomaly in active[:5]:
                name = anomaly.get(
                    "name",
                    anomaly.get("cid", "Unknown")
                )

                label = anomaly.get(
                    "label",
                    anomaly.get("metric", "performance change")
                )

                pct = anomaly.get("pct", 0)

                lines.append(
                    f"{name}: {label} ({pct:+.0%})"
                )

            return (
                "I found these performance issues:\n"
                + "\n".join(lines)
            )

        # -----------------------------------------------------
        # Why / root cause
        # -----------------------------------------------------

        if (
            "why" in text
            or "reason" in text
            or "cause" in text
            or "explain" in text
        ):

            anomaly = next(
                (
                    a
                    for a in self.anomalies
                    if a.get("cid")
                    and not a.get("good")
                ),
                None,
            )

            if anomaly:

                cid = anomaly["cid"]

                causes = A.rca(
                    self,
                    cid
                )

                if causes:

                    details = "; ".join(
                        f"{item['cause']} [{item['impact']}]"
                        for item in causes[:3]
                    )

                    return (
                        f"{anomaly.get('name', cid)} is showing "
                        f"{anomaly.get('label', 'a performance issue')}.\n"
                        f"Likely causes: {details}"
                    )

            return (
                "I don't currently have a strong anomaly-based "
                "root cause to explain."
            )

        # -----------------------------------------------------
        # Conversion rate
        # -----------------------------------------------------

        if (
            "conversion" in text
            or "cvr" in text
        ):

            clicks = float(last_7.clicks.sum())
            conversions = float(last_7.conv.sum())

            cvr = (
                conversions / clicks
                if clicks
                else 0
            )

            return (
                f"Your overall 7-day conversion rate is "
                f"{percent(cvr)} "
                f"({number(conversions)} conversions from "
                f"{number(clicks)} clicks)."
            )

        # -----------------------------------------------------
        # CTR
        # -----------------------------------------------------

        if (
            "ctr" in text
            or "click through" in text
        ):

            impressions = float(
                last_7.impressions.sum()
            )

            clicks = float(
                last_7.clicks.sum()
            )

            ctr = (
                clicks / impressions
                if impressions
                else 0
            )

            return (
                f"Your overall 7-day CTR is "
                f"{percent(ctr)}."
            )

        # -----------------------------------------------------
        # General dashboard summary
        # -----------------------------------------------------

        if (
            "summary" in text
            or "overview" in text
            or "dashboard" in text
            or "how are we doing" in text
        ):

            spend = float(last_7.spend.sum())
            revenue = float(last_7.revenue.sum())
            profit = float(last_7.profit.sum())

            roas = (
                revenue / spend
                if spend
                else 0
            )

            return (
                "Here's your current 7-day overview:\n"
                f"Spend: {money(spend)}\n"
                f"Revenue: {money(revenue)}\n"
                f"Profit: {money(profit)}\n"
                f"ROAS: {roas:.2f}x\n"
                f"Active campaigns: {len(self.camps)}\n"
                f"Anomalies: {len([a for a in self.anomalies if not a.get('good')])}\n"
                f"Recommendations: {len(self.recs)}"
            )

        # -----------------------------------------------------
        # Unknown question
        # -----------------------------------------------------

        return (
            "I'm not sure what you're asking yet. 🤔\n\n"
            "Try asking me something like:\n"
            "• Which campaign is performing best?\n"
            "• What is our ROAS?\n"
            "• Which product has low inventory?\n"
            "• Which campaign is most profitable?\n"
            "• Why is performance dropping?\n"
            "• Where should we increase budget?\n"
            "• Give me a summary of today's performance."
        )