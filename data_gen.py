"""
data_gen.py
===========
Generates a SYNTHETIC multi-store retail dataset for the
"Retail Loyalty & Promo Performance" portfolio dashboard.

  >>> ALL DATA PRODUCED HERE IS FAKE / SYNTHETIC <<<
No real customers, stores, transactions, or campaigns are represented.
The data is created procedurally from a fixed random seed so that every
run (and every reviewer) sees identical numbers.

Tables produced
---------------
    stores      one row per physical store
    customers   one row per customer (loyalty flag + signup cohort)
    orders      one row per order (header-level revenue, promo/loyalty flags)
    experiment  one row per customer in the promo A/B test (group + converted)

Run directly to write CSVs into ./data for inspection:
    python data_gen.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
N_CUSTOMERS = 6_000
N_DAYS = 365
START_DATE = pd.Timestamp("2024-09-01")
SNAPSHOT_DATE = START_DATE + pd.Timedelta(days=N_DAYS)  # "today" for RFM recency

STORES = [
    # store_id, store_name, region, opened_year
    (1, "Downtown Flagship", "West", 2009),
    (2, "Harbor Center", "West", 2013),
    (3, "Prairie Commons", "Midwest", 2011),
    (4, "Lakeside Mall", "Midwest", 2016),
    (5, "Riverwalk Plaza", "South", 2014),
    (6, "Sunbelt Crossing", "South", 2018),
    (7, "Granite Square", "Northeast", 2010),
    (8, "Online", "Ecommerce", 2007),
]

# Loyalty members spend a bit more and buy a bit more often. Promo-exposed
# orders carry a modest discount but drive incremental volume.
LOYALTY_SHARE = 0.42
BASE_ORDER_VALUE = 58.0


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #
def make_stores() -> pd.DataFrame:
    df = pd.DataFrame(STORES, columns=["store_id", "store_name", "region", "opened_year"])
    return df


def make_customers(rng: np.random.Generator) -> pd.DataFrame:
    customer_id = np.arange(1, N_CUSTOMERS + 1)
    is_loyalty = rng.random(N_CUSTOMERS) < LOYALTY_SHARE
    # Signup cohort month (0 = 11 months ago ... 11 = this month)
    signup_offset = rng.integers(0, N_DAYS, size=N_CUSTOMERS)
    signup_date = START_DATE + pd.to_timedelta(signup_offset, unit="D")
    # Per-customer latent "activity" level drives how many orders they place.
    activity = rng.gamma(shape=2.0, scale=1.0, size=N_CUSTOMERS)
    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "is_loyalty": is_loyalty,
            "signup_date": signup_date,
            "activity": activity,
        }
    )


def make_orders(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    """One row per order. Loyalty members order more often and spend more."""
    rows = []
    store_ids = [s[0] for s in STORES]
    # Online store gets a larger share of orders.
    store_weights = np.array([0.14, 0.11, 0.11, 0.10, 0.11, 0.09, 0.10, 0.24])
    store_weights = store_weights / store_weights.sum()

    cust_ids = customers["customer_id"].to_numpy()
    cust_loyal = customers["is_loyalty"].to_numpy()
    cust_activity = customers["activity"].to_numpy()

    # Expected order count per customer.
    base_lambda = 2.4
    lam = base_lambda * cust_activity * np.where(cust_loyal, 1.5, 1.0)
    n_orders = rng.poisson(lam)

    order_id = 1
    for i, cid in enumerate(cust_ids):
        k = int(n_orders[i])
        if k == 0:
            continue
        loyal = bool(cust_loyal[i])
        day_offsets = rng.integers(0, N_DAYS, size=k)
        stores_for = rng.choice(store_ids, size=k, p=store_weights)
        # Promo exposure ~28% of orders; loyalty members slightly more targeted.
        promo_prob = 0.33 if loyal else 0.24
        promo_flags = rng.random(k) < promo_prob
        for j in range(k):
            # Revenue: lognormal around base, lifted for loyalty, discounted on promo.
            base = BASE_ORDER_VALUE * (1.18 if loyal else 1.0)
            rev = rng.lognormal(mean=np.log(base), sigma=0.45)
            if promo_flags[j]:
                rev *= 0.90  # promo discount
            rows.append(
                (
                    order_id,
                    int(cid),
                    int(stores_for[j]),
                    START_DATE + pd.Timedelta(days=int(day_offsets[j])),
                    round(float(rev), 2),
                    bool(promo_flags[j]),
                    loyal,
                )
            )
            order_id += 1

    df = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "customer_id",
            "store_id",
            "order_date",
            "revenue",
            "is_promo",
            "is_loyalty",
        ],
    )
    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def make_experiment(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    """
    Promo A/B test: a "Loyalty Members: 15% Off Weekend" email campaign.
    A random half of eligible customers get the email (treatment); the rest
    are held out (control). We record whether each customer converted
    (made a qualifying purchase) during the campaign window.

    True effect baked in: control converts ~11%, treatment ~13.2%
    (a ~2.2pp absolute lift) so the readout shows a real, significant result.
    """
    # Whole-base holdout test: every customer is randomized to emailed vs held-out.
    eligible = customers.sample(frac=1.0, random_state=SEED)["customer_id"].to_numpy()
    rng.shuffle(eligible)
    half = len(eligible) // 2
    treatment_ids = set(eligible[:half].tolist())

    rows = []
    base_rate = 0.110
    treat_rate = 0.132
    for cid in eligible:
        group = "treatment" if cid in treatment_ids else "control"
        p = treat_rate if group == "treatment" else base_rate
        converted = int(rng.random() < p)
        rows.append((int(cid), group, converted))

    return pd.DataFrame(rows, columns=["customer_id", "group", "converted"])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_all() -> dict[str, pd.DataFrame]:
    """Build every table in memory and return them in a dict. Deterministic."""
    rng = _rng()
    stores = make_stores()
    customers = make_customers(rng)
    orders = make_orders(rng, customers)
    experiment = make_experiment(rng, customers)
    return {
        "stores": stores,
        "customers": customers,
        "orders": orders,
        "experiment": experiment,
    }


def write_csvs(out_dir: str = "data") -> None:
    os.makedirs(out_dir, exist_ok=True)
    tables = generate_all()
    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path:<28} {len(df):>7,} rows")


if __name__ == "__main__":
    print("Generating SYNTHETIC retail data (fixed seed = %d) ..." % SEED)
    write_csvs()
    print("Done. All data is synthetic and for portfolio demonstration only.")
