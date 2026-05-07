# Prosperity 3 — Team matchacita

> **IMC Prosperity 3** is an algorithmic trading competition where teams write bots to trade simulated financial instruments across 5 rounds of increasing complexity.

## Results

| Metric | Result |
|---|---|
| Overall rank | **849 / 12,620** — top 7% globally |
| United States rank | **Top 250** |
| Algorithmic trading rank | **495 / 12,620** — top 4% |

---

## Table of Contents

- [Competition Overview](#competition-overview)
- [Repository Structure](#repository-structure)
- [Tutorial Round](#tutorial-round)
- [Round 1 — Foundation](#round-1--foundation)
- [Round 2 — Basket Arbitrage](#round-2--basket-arbitrage)
- [Round 3 — Options Pricing](#round-3--options-pricing)
- [Round 4 — Conversion Arbitrage](#round-4--conversion-arbitrage)
- [Round 5 — Flow-Based Trading](#round-5--flow-based-trading)
- [Strategy Evolution](#strategy-evolution)
- [Mathematical Methods](#mathematical-methods)

---

## Competition Overview

IMC Prosperity 3 (2025) is a 15-day, 5-round global algorithmic trading competition with 12,620 participating teams. The competition is set in a fictional island economy where teams trade goods using **SeaShells** as currency. Each round introduces new products and mechanics that progressively mirror real financial markets — from basic market making all the way to derivatives, cross-exchange arbitrage, and information-based trading.

Participants submit a Python `Trader` class whose `run()` method is called at every timestep. The bot receives a live order book snapshot and emits buy/sell orders. Position limits are enforced per product, and cumulative PnL is tracked across all rounds.

| Round | Theme | New Products |
|---|---|---|
| Tutorial | Warm-up: stable vs. volatile assets | RAINFOREST_RESIN, KELP |
| 1 | Mean reversion & market making | SQUID_INK |
| 2 | ETF / index basket arbitrage | CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2 |
| 3 | Options pricing & derivatives | VOLCANIC_ROCK + 5 call option vouchers |
| 4 | Cross-exchange locational arbitrage | MAGNIFICENT_MACARONS |
| 5 | Flow-based / information trading | Counterparty IDs revealed (no new products) |

---

## Repository Structure

```
prosperity3/
├── tutorial/              # Tutorial warm-up implementations
├── round1/                # Round 1 — 3 products
│   ├── round1.py          # Submitted trader
│   └── datamodel.py       # Competition datamodel
├── round2/                # Round 2 — 8 products (baskets added)
│   ├── round2.py          # Primary submission
│   ├── emily.py           # Basket arbitrage / z-score variant
│   ├── meep.py            # Experimental variant
│   └── datamodel.py
├── round3/                # Round 3 — 13 products (options added)
│   ├── round3.py          # Submitted trader
│   └── datamodel.py
├── round4/                # Round 4 — 14 products (macarons added)
│   ├── round4.py          # Main composite trader
│   ├── squid_ink.py       # Cycle-detection specialist
│   ├── squid_ink_test.py  # Test variant
│   ├── macaron.py         # Conversion-arbitrage specialist
│   ├── macaron_fixed.py   # Refined macaron version
│   └── datamodel.py
└── big data/              # Execution logs and backtesting output
```

---

## Tutorial Round

> **Official description:** RAINFOREST_RESIN has been stable throughout the history of the archipelago. KELP has been going up and down over time. Algorithms are processed instantly in the tutorial round for quick experimentation.

**Products:**

| Product | Limit | Price Behavior |
|---|---|---|
| RAINFOREST_RESIN | 50 | Stable — consistent historical fair value |
| KELP | 50 | Volatile — oscillates over time |

**File:** `tutorial/`

This round was used to test the order submission framework and validate basic market-making logic before Round 1.

---

## Round 1 — Foundation

> **Official description:** RAINFOREST_RESIN remains stable, KELP continues to oscillate, and SQUID_INK is introduced as a highly volatile product. The official hint noted that SQUID_INK shows a strong tendency to revert large short-term price swings — tracking the size of deviation from a recent average is key.

**Products traded:**

| Product | Limit | Price Behavior |
|---|---|---|
| RAINFOREST_RESIN | 50 | Stable — fixed fair value strategy |
| KELP | 50 | Oscillating — volatility-adaptive market making |
| SQUID_INK | 50 | High volatility with mean reversion tendencies |

**File:** `round1/round1.py`

### Methods

**RAINFOREST_RESIN — Fixed Fair Value**
- Hard-coded fair value of **10,000** derived from its historically stable price
- Places limit orders symmetrically around fair value with a fixed spread width of 1
- Captures any quote that crosses fair value; clears residual positions at fair value

**KELP — Volatility-Adaptive Market Making**
- Fair value = `(best_bid + best_ask) / 2`
- Maintains a rolling 20-price window to compute price volatility (std / mean)
- Spread formula: `max(2, min(5, int(volatility × mid_price)))` — widens in volatile periods, tightens when calm
- Order size: 10% of remaining position capacity

**SQUID_INK — Bollinger Bands Mean Reversion**
- Motivated by the official hint: large price swings tend to revert
- Tracks a 20-period mid-price history; computes `mean ± 2σ` Bollinger Bands
- **Buy** signal: price crosses below lower band (oversold)
- **Sell** signal: price crosses above upper band (overbought)
- Fixed order size of 10 units per signal

**Order Execution Pattern (used throughout all rounds)**
1. **Take** — hit opposing quotes that beat fair value by more than a threshold
2. **Clear** — liquidate residual positions at or near fair value
3. **Make** — post resting limit orders around fair value to earn the spread

---

## Round 2 — Basket Arbitrage

> **Official description:** Two Picnic Baskets are introduced as composite tradable goods alongside their individual components. PICNIC_BASKET1 = 6× CROISSANTS + 3× JAMS + 1× DJEMBES. PICNIC_BASKET2 = 4× CROISSANTS + 2× JAMS. All five new products can also be traded individually on the exchange.

**New products:**

| Product | Limit | Composition |
|---|---|---|
| CROISSANTS | 250 | Component |
| JAMS | 350 | Component |
| DJEMBES | 60 | Component |
| PICNIC_BASKET1 | 60 | 6× CROISSANTS + 3× JAMS + 1× DJEMBES |
| PICNIC_BASKET2 | 100 | 4× CROISSANTS + 2× JAMS |

**Files:** `round2/round2.py`, `round2/emily.py`

### Methods

**Market-Maker Filtered Fair Value**
- Discards small "noise" orders below an `adverse_volume` threshold
- Fair value = mean price of remaining large orders (proxy for informed flow)
- Applies **mean reversion correction**: `fair_value = mid + mid × (returns × β)` where `β = −0.229`

**Component Trading (CROISSANTS, JAMS, DJEMBES)**
- Three-phase take/clear/make execution with per-product parameters (`take_width`, `clear_width`, `edge`)
- Soft position limits prevent over-commitment before the clear phase runs

**Basket Arbitrage (`emily.py`)**
- Continuously monitors the spread between basket price and the weighted sum of component prices
- **Leg 1:** basket ask < component bid-side value → buy basket, sell components
- **Leg 2:** basket bid > component ask-side value → sell basket, buy components
- Minimum profit threshold of 2 units filters out marginal trades

**Z-Score Statistical Arbitrage (`emily.py`)**
- Maintains a 25-period rolling window of the basket-vs-component spread
- Computes `z = (spread − mean) / std`
- `z > +2` → spread too wide → short basket / long components
- `z < −2` → spread too tight → long basket / short components
- Reverts to neutral when `|z| < 0.5`

---

## Round 3 — Options Pricing

> **Official description:** Volcanic Rock Vouchers are call options — they give the right but not the obligation to buy VOLCANIC_ROCK at a fixed strike price at expiry. Vouchers start with **7 trading days to expiry at the beginning of Round 1** (1 round = 1 day), leaving 2 days remaining by Round 5. The official hint introduced the following framework:
>
> - Compute moneyness: `m_t = log(K / S_t) / sqrt(TTE)`
> - Extract implied volatility: `v_t = BlackScholes_ImpliedVol(S_t, V_t, K, TTE)`
> - Plot `v_t` vs `m_t` across strikes and fit a parabolic curve to filter noise
> - The fitted curve is the **volatility smile**; `v_t(m_t = 0)` is the **base IV**

**New products:**

| Product | Strike | Limit | Time to Expiry (start of Round 1) |
|---|---|---|---|
| VOLCANIC_ROCK (underlying) | — | 400 | — |
| VOLCANIC_ROCK_VOUCHER_9500 | 9,500 | 200 | 7 days |
| VOLCANIC_ROCK_VOUCHER_9750 | 9,750 | 200 | 7 days |
| VOLCANIC_ROCK_VOUCHER_10000 | 10,000 | 200 | 7 days |
| VOLCANIC_ROCK_VOUCHER_10250 | 10,250 | 200 | 7 days |
| VOLCANIC_ROCK_VOUCHER_10500 | 10,500 | 200 | 7 days |

**File:** `round3/round3.py`

### Methods

**Black-Scholes Call Pricing**
- Implements the closed-form Black-Scholes formula for European call options
- Inputs: spot `S`, strike `K`, time to expiry `T`, risk-free rate `r`, volatility `σ`
- Output: theoretical call price used as fair value for each voucher

**Implied Volatility Extraction**
- Inverts Black-Scholes numerically to back out `σ_implied` from observed market prices
- One IV computed per strike per timestep

**Volatility Smile Fitting**
- Computes moneyness `m_t = log(K / S_t) / sqrt(TTE)` per the official hint
- Fits a **quadratic (parabolic) curve** to IV vs. `m_t` using `numpy.polyfit`
- Each strike is re-priced with its smile-adjusted IV, smoothing out noise in individual estimates

**Voucher Trading**
- `market_ask < 0.98 × theoretical` → buy (underpriced)
- `market_bid > 1.02 × theoretical` → sell (overpriced)
- 2% buffer avoids over-trading on small mispricings

**Volatility Scalping**
- Manages short-vega exposure on VOLCANIC_ROCK itself
- Sells 1% of holdings on price increases; buys 1.0125% on price decreases
- Exploits negative autocorrelation in large price moves

---

## Round 4 — Conversion Arbitrage

> **Official description:** MAGNIFICENT_MACARONS are a luxury product whose value depends on observable factors: sunlight hours, sugar prices, shipping costs, and import/export tariffs. They can be traded locally **or** converted with the foreign supplier "Pristine Cuisine" at a published fee schedule. A storage cost of **0.1 SeaShells per timestamp** applies to every unit held long. The official hint introduced the **Critical Sunlight Index (CSI)**: when sunlight drops below CSI for an extended period, macaron prices rise sharply; above CSI, prices follow normal supply/demand dynamics.

**New product:**

| Product | Position Limit | Conversion Limit | Storage Cost |
|---|---|---|---|
| MAGNIFICENT_MACARONS | 75 | 10 per timestep | 0.1 SeaShells / unit / timestamp (long only) |

**Conversion mechanics (trading with Pristine Cuisine):**
- **Buy via conversion:** pay `ask + transport_fees + import_tariff`
- **Sell via conversion:** receive `bid − transport_fees − export_tariff`
- Conversions are the **only** way to trade with the external market

**Files:** `round4/round4.py`, `round4/squid_ink.py`, `round4/macaron.py`

### Methods

**Main Trader (`round4.py`)**
- Orchestrates all sub-strategies in a single `run()` call
- Maintains shared `traderData` JSON state across timesteps
- Routes each product to its dedicated handler

**SQUID_INK Cycle Specialist (`squid_ink.py`)**

*Bootstrap phase (first 20 timesteps):*
- Adaptive market making with `width = std_dev × 2` (clamped to [1, 3])
- Order size grows gradually as price history accumulates

*Advanced phase:*
- **Dual-timeframe cycle detection** — 75-period primary cycle blended 70/30 with a 25-period short cycle
- Cycle position expressed as a scalar in [−1, +1] (−1 = trough, +1 = peak)
- **Average True Range (ATR)** over 14 periods for volatility-adjusted spreads
- **Momentum signal** = recent 10-period avg vs. prior 10-period avg
- Asymmetric bid/ask placement leans toward anticipated cycle direction
- Adaptive order sizing scales with cycle strength and ATR

**MAGNIFICENT_MACARONS Conversion Arbitrage (`macaron.py`)**
- **Buy signal:** `(bid − transport_fees − export_tariff) − local_ask ≥ 1.0`
- **Sell signal:** `local_bid − (ask + transport_fees + import_tariff) ≥ 1.0`
- Maximum 10 conversions per timestep; position capped at 75
- Storage cost of 0.1 SeaShells/unit/timestamp for long positions factors into hold decisions
- **CSI filter:** monitors sunlight index; increases position aggressiveness when sunlight drops below the critical threshold

---

## Round 5 — Flow-Based Trading

> **Official description:** No new products are introduced. Instead, the exchange now discloses the **counterparty ID** on every trade via the `counter_party` field of the `OwnTrade` object. Teams can use this to identify informed traders and leverage their order flow as a directional signal.

```python
class OwnTrade:
    def __init__(self, symbol, price, quantity, counter_party=None):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.counter_party = counter_party  # now populated in Round 5
```

**Strategy:**
- Track fill history per counterparty to identify consistently profitable participants
- **Informed-flow following** — shadow the direction of known profitable traders
- **Adverse selection filter** — avoid taking the other side of informed participants; only trade against known noise traders
- All Round 1–4 strategies remain active; the counterparty signal is layered on top

> *Note: Round 5 built directly on the Round 4 codebase with live parameter tuning; a separate file is not included in this repository.*

---

## Strategy Evolution

| Round | Theme | Key Techniques | Products |
|---|---|---|---|
| Tutorial | Stable vs. volatile assets | Basic market making | 2 |
| 1 | Mean reversion & MM | Fixed fair value · Bollinger Bands · Volatility-adaptive MM | 3 |
| 2 | ETF basket arbitrage | Mean reversion beta · Basket arbitrage · Z-score stat-arb | 8 |
| 3 | Options / derivatives | Black-Scholes · IV extraction · Volatility smile fitting | 14 |
| 4 | Cross-exchange arbitrage | Cycle detection · ATR spreads · Conversion arbitrage · CSI filter | 15 |
| 5 | Flow-based / info trading | Counterparty flow analysis · Informed-trader shadowing | 15 |

---

## Mathematical Methods

| Method | Used In |
|---|---|
| Rolling mean & standard deviation | All rounds |
| Bollinger Bands (mean ± 2σ) | Round 1 — SQUID_INK |
| Mean reversion with beta coefficient | Round 2 — KELP, SQUID_INK |
| Z-score normalization | Round 2 — basket arbitrage |
| Black-Scholes option pricing | Round 3 — vouchers |
| Moneyness normalization `log(K/S) / sqrt(TTE)` | Round 3 — volatility smile |
| Numerical implied-volatility inversion | Round 3 — vouchers |
| Polynomial regression (volatility smile) | Round 3 — vouchers |
| Average True Range (ATR) | Round 4 — SQUID_INK |
| Dual-timeframe cycle detection | Round 4 — SQUID_INK |
| Fee-adjusted conversion arbitrage | Round 4 — MACARONS |
| Critical Sunlight Index (CSI) threshold filter | Round 4 — MACARONS |
| Counterparty flow analysis | Round 5 |
