"""
analytics.py
============
Pure, framework-free computation for the Retail Loyalty & Promo dashboard.

Every function here takes pandas DataFrames in and returns plain Python / pandas
objects out -- no Streamlit, no plotting. That separation keeps the business
logic unit-testable and lets reviewers run the numbers without a browser:

    python -c "import analytics, data_gen; ..."   # see __main__ smoke test below
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# 1. Headline KPIs
# --------------------------------------------------------------------------- #
def compute_kpis(orders: pd.DataFrame) -> dict[str, float]:
    """Revenue, orders, AOV, unique customers, and loyalty share of sales."""
    revenue = float(orders["revenue"].sum())
    n_orders = int(len(orders))
    aov = revenue / n_orders if n_orders else 0.0
    n_customers = int(orders["customer_id"].nunique())
    loyalty_rev = float(orders.loc[orders["is_loyalty"], "revenue"].sum())
    loyalty_share = (loyalty_rev / revenue) if revenue else 0.0
    promo_rev = float(orders.loc[orders["is_promo"], "revenue"].sum())
    promo_share = (promo_rev / revenue) if revenue else 0.0
    return {
        "revenue": revenue,
        "orders": n_orders,
        "aov": aov,
        "customers": n_customers,
        "loyalty_share": loyalty_share,
        "promo_share": promo_share,
    }


def revenue_by_month(orders: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue split by loyalty vs non-loyalty for the trend chart."""
    df = orders.copy()
    df["month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").dt.to_timestamp()
    df["segment"] = np.where(df["is_loyalty"], "Loyalty", "Non-loyalty")
    out = (
        df.groupby(["month", "segment"], as_index=False)["revenue"]
        .sum()
        .sort_values("month")
    )
    return out


# --------------------------------------------------------------------------- #
# 2. A/B promo-lift readout (two-proportion z-test)
# --------------------------------------------------------------------------- #
@dataclass
class ABResult:
    n_treatment: int
    n_control: int
    conv_treatment: int
    conv_control: int
    rate_treatment: float
    rate_control: float
    abs_lift: float          # treatment rate - control rate (percentage points)
    rel_lift: float          # abs_lift / control rate
    z_stat: float
    p_value: float
    ci_low: float            # 95% CI on the absolute difference
    ci_high: float
    significant: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def ab_two_proportion(experiment: pd.DataFrame, alpha: float = 0.05) -> ABResult:
    """
    Two-proportion z-test comparing treatment vs control conversion.

    Pooled standard error is used for the test statistic / p-value; the
    unpooled standard error is used for the 95% CI on the difference, which
    is the conventional pairing taught for A/B conversion tests.
    """
    g = experiment.groupby("group")["converted"]
    n_t = int(g.count().get("treatment", 0))
    n_c = int(g.count().get("control", 0))
    x_t = int(g.sum().get("treatment", 0))
    x_c = int(g.sum().get("control", 0))

    p_t = x_t / n_t if n_t else 0.0
    p_c = x_c / n_c if n_c else 0.0

    # Pooled proportion for the z statistic.
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    z = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval on the difference.
    se_unpool = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_low = diff - z_crit * se_unpool
    ci_high = diff + z_crit * se_unpool

    return ABResult(
        n_treatment=n_t,
        n_control=n_c,
        conv_treatment=x_t,
        conv_control=x_c,
        rate_treatment=p_t,
        rate_control=p_c,
        abs_lift=diff,
        rel_lift=(diff / p_c) if p_c else float("nan"),
        z_stat=float(z),
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(p_value < alpha),
    )


# --------------------------------------------------------------------------- #
# 3. RFM customer segmentation
# --------------------------------------------------------------------------- #
def _score_1_to_5(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Quintile score 1..5. reverse=True means small values score high
    (used for Recency: fewer days since last order is better).

    Falls back gracefully when a column has too few distinct values to
    cut into five bins.
    """
    try:
        ranks = series.rank(method="first")
        q = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        score = q.astype(int)
    except ValueError:
        # Not enough distinct values -> rank-scale into 1..5.
        r = series.rank(pct=True)
        score = np.ceil(r * 5).clip(1, 5).astype(int)
    if reverse:
        score = 6 - score
    return score


def _segment(r: int, fm: int) -> str:
    """Map an (R, FM) score pair to a named segment (standard RFM grid)."""
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3:
        return "Loyal Customers"
    if r >= 4 and fm <= 2:
        return "New / Promising"
    if r == 3 and fm <= 2:
        return "Potential Loyalists"
    if r <= 2 and fm >= 4:
        return "Can't Lose Them"
    if r <= 2 and fm == 3:
        return "At Risk"
    return "Hibernating"


def compute_rfm(orders: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build a per-customer RFM table with 1..5 scores and a named segment.

    Recency  = days since last order (lower is better)
    Frequency= number of orders
    Monetary = total revenue
    """
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o["order_date"])
    snapshot = pd.Timestamp(snapshot_date)

    agg = o.groupby("customer_id").agg(
        recency_days=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_id", "count"),
        monetary=("revenue", "sum"),
    ).reset_index()

    agg["R"] = _score_1_to_5(agg["recency_days"], reverse=True)
    agg["F"] = _score_1_to_5(agg["frequency"])
    agg["M"] = _score_1_to_5(agg["monetary"])
    agg["FM"] = np.ceil((agg["F"] + agg["M"]) / 2).astype(int)
    agg["segment"] = [_segment(int(r), int(fm)) for r, fm in zip(agg["R"], agg["FM"])]
    return agg


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Per-segment customer counts and revenue -- the storytelling table."""
    out = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            total_monetary=("monetary", "sum"),
        )
        .reset_index()
    )
    out["pct_customers"] = out["customers"] / out["customers"].sum()
    out["pct_revenue"] = out["total_monetary"] / out["total_monetary"].sum()
    return out.sort_values("total_monetary", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Data-quality checks (framed like dbt tests)
# --------------------------------------------------------------------------- #
@dataclass
class QualityCheck:
    test: str          # dbt-style test name
    target: str        # table.column it applies to
    status: str        # "pass" or "fail"
    detail: str        # human-readable result


def run_data_quality(tables: dict[str, pd.DataFrame]) -> list[QualityCheck]:
    """
    A compact battery of validation tests, modelled on the kinds of
    assertions a dbt project ships in CI: not-null, unique, accepted range,
    and relationships (referential integrity).
    """
    orders = tables["orders"]
    customers = tables["customers"]
    stores = tables["stores"]
    experiment = tables["experiment"]
    checks: list[QualityCheck] = []

    def add(test, target, ok, detail):
        checks.append(QualityCheck(test, target, "pass" if ok else "fail", detail))

    # --- not_null ---------------------------------------------------------
    for col in ["order_id", "customer_id", "store_id", "order_date", "revenue"]:
        n_null = int(orders[col].isna().sum())
        add("not_null", f"orders.{col}", n_null == 0, f"{n_null} null(s)")

    # --- unique (primary keys) -------------------------------------------
    dup_orders = int(orders["order_id"].duplicated().sum())
    add("unique", "orders.order_id", dup_orders == 0, f"{dup_orders} duplicate PK(s)")
    dup_cust = int(customers["customer_id"].duplicated().sum())
    add("unique", "customers.customer_id", dup_cust == 0, f"{dup_cust} duplicate PK(s)")

    # --- accepted_range ---------------------------------------------------
    n_bad_rev = int((orders["revenue"] <= 0).sum())
    add("accepted_range", "orders.revenue > 0", n_bad_rev == 0, f"{n_bad_rev} non-positive")

    # --- relationships (referential integrity) ---------------------------
    orphan_cust = int((~orders["customer_id"].isin(customers["customer_id"])).sum())
    add("relationships", "orders.customer_id -> customers",
        orphan_cust == 0, f"{orphan_cust} orphan row(s)")
    orphan_store = int((~orders["store_id"].isin(stores["store_id"])).sum())
    add("relationships", "orders.store_id -> stores",
        orphan_store == 0, f"{orphan_store} orphan row(s)")
    orphan_exp = int((~experiment["customer_id"].isin(customers["customer_id"])).sum())
    add("relationships", "experiment.customer_id -> customers",
        orphan_exp == 0, f"{orphan_exp} orphan row(s)")

    # --- accepted_values --------------------------------------------------
    bad_groups = int((~experiment["group"].isin(["treatment", "control"])).sum())
    add("accepted_values", "experiment.group", bad_groups == 0,
        f"{bad_groups} unexpected value(s)")

    return checks


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import data_gen

    t = data_gen.generate_all()
    kpis = compute_kpis(t["orders"])
    print("KPIs:", {k: round(v, 2) for k, v in kpis.items()})

    ab = ab_two_proportion(t["experiment"])
    print(
        f"A/B: control={ab.rate_control:.3%} treatment={ab.rate_treatment:.3%} "
        f"abs_lift={ab.abs_lift:+.3%} p={ab.p_value:.4f} "
        f"CI=[{ab.ci_low:+.3%}, {ab.ci_high:+.3%}] sig={ab.significant}"
    )

    rfm = compute_rfm(t["orders"], data_gen.SNAPSHOT_DATE)
    print("\nSegment summary:")
    print(segment_summary(rfm).to_string(index=False))

    checks = run_data_quality(t)
    n_pass = sum(c.status == "pass" for c in checks)
    print(f"\nData quality: {n_pass}/{len(checks)} tests passed")
    for c in checks:
        print(f"  [{c.status.upper():4}] {c.test:16} {c.target:38} {c.detail}")
