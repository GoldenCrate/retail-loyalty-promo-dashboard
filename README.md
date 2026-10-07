# Retail Loyalty & Promo Performance

An interactive **Streamlit** dashboard that walks through the core loop of an
enterprise retail analytics team: track the headline KPIs, measure a promotional
A/B test, segment customers with RFM, and validate the underlying data before
anyone trusts a number.

Built as a portfolio project by **Leo Chan - [github.com/GoldenCrate](https://github.com/GoldenCrate)**.

> **All data in this project is SYNTHETIC.** It is generated procedurally by
> `data_gen.py` from a fixed random seed. There are no real customers, stores,
> transactions, or marketing campaigns here - the dataset exists only to
> demonstrate the analysis and the dashboard.

---

## What the dashboard shows

The app has four views, selectable from the sidebar. Every view ends with a
plain-language **"So what"** line stating the single takeaway for a
non-technical stakeholder.

1. **KPI Overview** - Revenue, orders, average order value, loyalty share of
   sales, and promo share of sales as headline tiles, plus a monthly revenue
   trend (loyalty vs. non-loyalty) and revenue by store.
2. **Promo A/B Lift** - A whole-base holdout test of a "Loyalty 15% Off Weekend"
   email. Reports the control and treatment conversion rates, absolute and
   relative lift, a **two-proportion z-test** (z-stat and p-value), and the
   **95% confidence interval** on the lift, with a clear ship / don't-ship verdict.
3. **RFM Segmentation** - Scores every customer on **Recency, Frequency, and
   Monetary** value (1-5 quintiles), maps them to named segments
   (Champions, Loyal, At Risk, Can't Lose Them, Hibernating, ...), and shows
   how customer count and revenue concentrate across segments.
4. **Data Quality** - A dbt-style test suite run against the dataset:
   `not_null`, `unique`, `accepted_range`, `accepted_values`, and
   `relationships` (referential integrity) checks, each reported pass / fail.

## Charts

The visualizations follow Cole Nussbaumer Knaflic's *Storytelling with Data*
principles: every chart has a **takeaway title** (the insight, not just the
topic), is **decluttered** (no gridlines or chart junk), uses **direct labels**
instead of legends where possible, and relies on **restrained color** - grey for
context and a single accent for the one thing that matters.

## Project structure

| File             | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `data_gen.py`    | Generates the synthetic stores / customers / orders / experiment tables |
| `analytics.py`   | Pure, framework-free compute: KPIs, A/B z-test, RFM, data-quality tests |
| `app.py`         | The Streamlit UI (four views) that renders `analytics` outputs          |
| `requirements.txt` | Python dependencies                                                   |

Business logic lives in `analytics.py` with **no Streamlit imports**, so it can
be unit-tested or run headless, independent of the UI.

## Run it locally

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) write the synthetic CSVs to ./data for inspection
python data_gen.py

# 4. launch the dashboard
streamlit run app.py
```

The app regenerates the data in memory on startup, so step 3 is optional -
a fresh clone runs with no pre-built files. Because the seed is fixed, every
run produces identical numbers.

### Quick headless check (no browser)

```bash
python analytics.py     # prints KPIs, the A/B readout, segments, and the DQ suite
```

## Notes on the synthetic data

- `~6,000` customers, `~35,000` orders, `8` stores, 12 months of history.
- A real ~2.2 percentage-point promo effect is baked into the experiment so the
  A/B readout has something genuine to detect; the test correctly recovers it
  as significant (p < 0.01).
- Loyalty members are modeled to order more often and spend slightly more,
  which is why they account for the majority of revenue.

---

*Synthetic data, portfolio demonstration only. Author: Leo Chan -
[github.com/GoldenCrate](https://github.com/GoldenCrate).*
