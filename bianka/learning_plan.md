# Forecasting & Event Prediction — Learning Plan

> For an infrastructure engineer getting into probability theory and prediction markets as a hobby.
> Ordered in **ascending complexity**, alternating **theory** and **hands-on practice**.
> Legend: **[MH]** = must have · **[NTH]** = nice to have · **[PRACTICE]** · **[THEORY]**

---

## Phase 0 — Orientation (Week 0)

Goal: understand the landscape and set up a workflow you'll actually use.

- **[THEORY]** What is a prediction/event market? (Polymarket, Kalshi, Manifold, Metaculus). How a "market price" maps to a probability.
- **[THEORY]** Forecasting vs. trading vs. betting — what is the actual edge you're hunting?
- **[PRACTICE]** Open free/paper accounts: **Manifold** (play money, great for learning), **Metaculus** (pure forecasting, no money), read but don't yet trade on Polymarket/Kalshi.
- **[PRACTICE]** Make 10 manual forecasts on Metaculus/Manifold. Write down *why* for each. This is your baseline.

**Milestone:** you can articulate, in one paragraph, what "the price is 63%" means and what would make it mispriced.

---

## Phase 1 — Infrastructure Setup

You already have the skills here — this is your unfair advantage. Build the plumbing early so every later phase has a place to live.

### Must have [MH]
- **[PRACTICE]** Python data stack: `python`, `uv`/`poetry`, `numpy`, `pandas`, `polars`, `matplotlib`, `jupyter`.
- **[PRACTICE]** A git repo for notebooks + scripts. Treat experiments as reproducible (seed, requirements pinned).
- **[PRACTICE]** A small local database for time series & market snapshots: **DuckDB** (start here, zero-ops) or **TimescaleDB/Postgres**.
- **[PRACTICE]** A data-collection daemon: poll market APIs (Polymarket/Kalshi/Manifold) on a schedule, store order book & price snapshots. (`cron`/`systemd` timer, or a tiny scheduler.)
- **[PRACTICE]** Secrets handling: API keys in env/`.env` (gitignored) or a secrets manager — don't commit keys.

### Nice to have [NTH]
- **[PRACTICE]** Containerize the collector (Docker) and run it on a cheap VM/your homelab (you have OCI infra already — reuse it).
- **[PRACTICE]** Orchestration for pipelines: **Prefect** or **Dagster** (lighter than Airflow for a hobby).
- **[PRACTICE]** Dashboards: **Grafana** (you likely know it) over your time-series DB, or **Streamlit** for quick analytics UIs.
- **[PRACTICE]** Alerting (Telegram/Discord bot) for "market moved X%" or "edge detected".
- **[PRACTICE]** Experiment tracking: **MLflow** or **Weights & Biases** once you start modeling.
- **[PRACTICE]** CI for your strategy code; a backtest that runs on every push.

**Milestone:** a service that, unattended, captures market data into a queryable store, and a notebook that reads it.

---

## Phase 2 — Math & Probability Theory (Foundations)

The core of the hobby. Go slow; do exercises, not just reading.

### Must have [MH]
- **[THEORY]** Probability basics: sample spaces, events, conditional probability, **Bayes' theorem**.
- **[THEORY]** Random variables, expectation, variance, common distributions (Bernoulli, Binomial, Poisson, Normal, Beta).
- **[THEORY]** **Law of large numbers** & **central limit theorem** — why sample sizes matter.
- **[THEORY]** Odds, probabilities, and the relationship to prices/implied probability. Removing the "vig"/spread.
- **[PRACTICE]** Simulate everything: coin flips, dice, Monte Carlo estimation of probabilities in Python.

### Nice to have [NTH]
- **[THEORY]** Bayesian inference: priors, posteriors, conjugate priors (Beta-Binomial is *the* event-prediction workhorse).
- **[THEORY]** Markov chains, basic stochastic processes, random walks.
- **[THEORY]** Information theory lite: entropy, KL divergence (useful for scoring beliefs).

### Resources (pick, don't hoard)
- Book: *Introduction to Probability* (Blitzstein & Hwang) — has free Harvard Stat 110 lectures.
- Book/Practical: *Think Bayes* (Allen Downey) — Python-first, perfect for an engineer.

**Milestone:** you can update a probability with Bayes by hand and via code, and explain a Beta posterior.

---

## Phase 3 — Calibration & Scoring (How good is a forecast?)

This is the bridge between "doing math" and "being a good forecaster."

- **[THEORY]** Proper scoring rules: **Brier score**, **log loss**. Why proper rules reward honesty.
- **[THEORY]** **Calibration** (do your 70%s happen 70% of the time?) vs. **resolution/sharpness**.
- **[PRACTICE]** Build a calibration plot from your Phase-0 forecasts. Score them.
- **[PRACTICE]** Keep a forecasting journal/notebook auto-scored from your DB.
- **[NTH][THEORY]** Read the *Superforecasting* (Tetlock) ideas: base rates, reference classes, updating incrementally.

**Milestone:** a dashboard showing your personal Brier score and calibration curve over time.

---

## Phase 4 — Market Essentials

Now learn the mechanics of the venues you'll act in.

### Core concepts [MH]
- **[THEORY]** **Order book** anatomy: bids/asks, depth, spread, mid price, last trade.
- **[THEORY]** **Liquidity**: market depth, slippage, what "thin" vs "deep" means for fills.
- **[THEORY]** Order types: limit vs market; maker vs taker; fees and their impact on edge.
- **[THEORY]** **Market makers** vs **takers**; how spreads are a cost you pay or earn.

### Prediction-market specifics [MH]
- **[THEORY]** Binary (Yes/No) markets, payout at $0 or $1, price = implied probability.
- **[THEORY]** **CLOB vs CPMM/LMSR**: Polymarket/Kalshi use order books; Manifold-style and some AMMs use automated market makers (constant-product or **Logarithmic Market Scoring Rule**).
- **[THEORY]** **Resolution risk**: who resolves the market, oracle/source ambiguity, settlement timing.
- **[THEORY]** Fees, gas (Polymarket on Polygon), withdrawal frictions.

### Practice [PRACTICE]
- Reconstruct a live order book from your collector and visualize depth/spread.
- Compute implied probability + remove fee/spread to get a "fair" estimate.
- Track how a single market's price evolves around a news event.

### Nice to have [NTH]
- **[THEORY]** LMSR math (cost function, price function) — directly ties Phase 2 to market microstructure.
- **[THEORY]** Cross-venue price differences and why they persist (frictions, capital, KYC).

**Milestone:** given any market, you can state spread, depth, fee-adjusted fair price, and resolution source.

---

## Phase 5 — Backtesting

Don't risk money until you can lie to yourself less. Backtesting is mostly about *not fooling yourself*.

### Must have [MH]
- **[PRACTICE]** Replay engine: feed historical snapshots through a strategy, simulate fills against the *recorded* order book/liquidity (not just mid price).
- **[THEORY]** Account for **fees, slippage, and spread** — most naive backtests die here.
- **[THEORY]** Avoid the classic traps: **look-ahead bias**, **survivorship bias**, overfitting, data snooping.
- **[PRACTICE]** Metrics: ROI, hit rate, **Brier/log loss** on entered markets, drawdown, P&L distribution.

### Nice to have [NTH]
- **[THEORY]** Out-of-sample / walk-forward validation; train/test splits by *time*, never random.
- **[PRACTICE]** Monte Carlo over outcomes to get a P&L distribution, not a single number.
- **[PRACTICE]** A reusable backtest framework in your repo with config-driven strategies + a results report.

**Milestone:** one strategy backtested honestly, with fees/slippage, reported as a P&L distribution — and you can explain why it might be fooling you.

---

## Phase 6 — Strategies (ascending risk/complexity)

Start with the lowest-edge-required, lowest-skill ideas; graduate to model-driven ones.

### Tier A — Low risk / low skill [MH to attempt at least one]
- **[PRACTICE]** **Copy/consensus betting**: follow Metaculus community / sharp Manifold users; measure if copying actually beats baseline (often it doesn't after fees — that's a lesson).
- **[PRACTICE]** **Arbitrage**:
  - *Internal*: Yes + No priced so total < $1 (minus fees) → lock profit.
  - *Cross-venue*: same event priced differently on Kalshi vs Polymarket (watch resolution-rule differences — they break "free" arbs).
  - *Dutch-book* across mutually exclusive outcomes summing < 100%.
- **[THEORY]** Why most pure arbs are tiny, fleeting, or fake (different resolution criteria, fees, capital lockup).

### Tier B — Math-backed / edge-from-modeling [MH for the math hobby]
- **[THEORY+PRACTICE]** **Base-rate / reference-class** models: estimate true probability, bet when market deviates beyond fees.
- **[PRACTICE]** **Bayesian updating** strategy: maintain a posterior per market, trade the gap to market price.
- **[THEORY]** **Kelly criterion** & fractional Kelly for position sizing (this is where Phase 2 pays off). Bankroll management.
- **[PRACTICE]** Simple predictive models: logistic regression / gradient boosting on features (polls, sports stats, on-chain/news signals) → calibrated probability → bet vs market.
- **[THEORY]** Calibrate model outputs (Platt/isotonic) before trusting them as probabilities.

### Tier C — Higher complexity [NTH]
- **[PRACTICE]** **Market making**: quote both sides, earn the spread, manage inventory risk (LMSR/CPMM understanding required).
- **[PRACTICE]** Event-driven / news-latency strategies: react to information faster than the book.
- **[PRACTICE]** Ensemble of models + meta-calibration; portfolio of uncorrelated bets.

**Milestone:** a documented strategy with: thesis, edge source, sizing rule (Kelly), backtest, and a paper-trading track record.

---

## Phase 7 — From Paper to (Tiny) Real & Operations

- **[PRACTICE]** Paper-trade live for several weeks; compare realized vs backtested results (expect a gap).
- **[PRACTICE]** Go live with money you can lose entirely. Start absurdly small.
- **[PRACTICE][MH]** **Risk management**: max position, max daily loss, bankroll % per bet, kill-switch.
- **[PRACTICE]** Monitoring/ops (your home turf): uptime for the collector, alerting on stale data, position/P&L dashboard, automated reconciliation.
- **[THEORY]** **Legal/tax/KYC**: check what's allowed in your jurisdiction *before* funding anything. Keep records for taxes.
- **[THEORY]** Post-mortems: review losing trades, update process, watch for regime change (a working edge decays).

**Milestone:** a small live system with hard risk limits, monitoring, and a regular review ritual.

---

## GitHub Projects to Learn From

Curated, roughly by phase. **Read the code, run the examples, then adapt — don't blindly trust strategy claims.**

> ⚠️ **Safety note:** Trading-bot repos often ask for **API keys / private keys**. Never run an untrusted bot with a funded wallet. Use a fresh wallet with trivial funds, read the code first, and treat any repo promising profits with heavy skepticism. The official SDKs and educational libraries below are the safe starting points.

### Official SDKs & APIs (Phases 1, 4)
- **Polymarket — `Polymarket/py-clob-client`** ([github](https://github.com/Polymarket/py-clob-client)) — official Python client for Polymarket's CLOB. Start here for order-book data & order placement.
- **Kalshi — `Kalshi-Exchange/kalshi-starter-code-python`** ([github](https://github.com/Kalshi-Exchange/kalshi-starter-code-python)) — official starter: REST/WebSocket auth (RSA), market data, orders.
- **Manifold — `manifoldmarkets/manifold`** ([github](https://github.com/manifoldmarkets/manifold)) — open-source play-money market; documented public API, great low-risk sandbox.

### Probability, Bayes & Forecasting (Phases 2–3)
- **`AllenDowney/ThinkBayes2`** ([github](https://github.com/AllenDowney/ThinkBayes2)) — the *Think Bayes* book as runnable notebooks. Engineer-friendly, Python-first.
- **`pymc-devs/pymc`** ([github](https://github.com/pymc-devs/pymc)) — probabilistic programming / Bayesian modeling; tons of example notebooks.
- **`scikit-learn` calibration docs/examples** ([github](https://github.com/scikit-learn/scikit-learn)) — `CalibratedClassifierCV`, reliability curves, Brier score — directly the Phase-3 toolkit.

### Time-Series & Predictive Modeling (Phases 2, 6)
- **`unit8co/darts`** ([github](https://github.com/unit8co/darts)) — clean unified API for forecasting (classical → deep learning), with backtesting built in.
- **`sktime/sktime`** ([github](https://github.com/sktime/sktime)) — scikit-learn-style time-series framework.
- **`facebook/prophet`** ([github](https://github.com/facebook/prophet)) — simple, fast baseline forecasting; good "is my fancy model even better than this?" yardstick.

### Backtesting (Phase 5)
- **`kernc/backtesting.py`** ([github](https://github.com/kernc/backtesting.py)) — **start here.** Tiny, beginner-friendly, easy mental model.
- **`mementum/backtrader`** ([github](https://github.com/mementum/backtrader)) — event-driven, realistic commissions/slippage; classic retail workhorse.
- **`polakowo/vectorbt`** ([github](https://github.com/polakowo/vectorbt)) — vectorized, sweep thousands of strategy variants in seconds (research, not live).
- **`nautechsystems/nautilus_trader`** ([github](https://github.com/nautechsystems/nautilus_trader)) — production-grade, backtest/live parity. Steeper curve — revisit in Phase 7.

### Prediction-Market Specific (Phases 4, 6) — community, unvetted
- **`AKCodez/prediction-market-alpha-playbook`** ([github](https://github.com/AKCodez/prediction-market-alpha-playbook)) — docs-only reference: edge types, API gotchas, architecture patterns, antipatterns. Great map of the terrain (read before coding).
- **`gnosis/prediction-market-agent-tooling`** ([github](https://github.com/gnosis/prediction-market-agent-tooling)) — Python tooling for building/benchmarking prediction-market agents.
- **`suislanchez/polymarket-kalshi-weather-bot`** ([github](https://github.com/suislanchez/polymarket-kalshi-weather-bot)) — worked multi-venue example (ensemble weather forecasts + Kelly sizing + dashboard). Read as a *case study*, not a money printer.

### How to use these
1. **Phase 1–3:** clone `ThinkBayes2`, run notebooks; pull live data with `py-clob-client`/Kalshi starter into your DuckDB.
2. **Phase 5:** reproduce a `backtesting.py` example, then re-implement *your* prediction-market replay with fees/slippage.
3. **Phase 6:** study the playbook's edge taxonomy + the weather-bot's sizing/calibration, then build your own — small and audited.

---

## Suggested Pacing (flexible)

| Weeks | Focus |
|------:|-------|
| 0     | Orientation (Phase 0) |
| 1–2   | Infra setup (Phase 1) + start probability (Phase 2) |
| 3–6   | Probability/Bayes (Phase 2) + calibration (Phase 3) |
| 5–7   | Market essentials (Phase 4), in parallel with collector data piling up |
| 7–9   | Backtesting (Phase 5) |
| 9–13  | Strategies Tier A → B (Phase 6) |
| 13+   | Paper → tiny live + ops (Phase 7), then revisit Tier C |

---

## Guiding Principles

1. **Practice each theory immediately** — simulate it, plot it, or trade it on paper.
2. **Probabilities first, money last.** Be well-calibrated before you size bets.
3. **Fees and slippage kill naive edges** — model them from day one.
4. **Small, reproducible experiments.** Treat strategies like infra: versioned, tested, monitored.
5. **Your edge is operational discipline + clean math**, not secret signals.
