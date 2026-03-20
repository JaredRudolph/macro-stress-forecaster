# macro-stress-forecaster

This is the third project in a four-part macro stress series. It extends [macro-stress-optimizer](https://github.com/JaredRudolph/macro-stress-optimizer) with a forward-looking forecaster. The pipeline and optimizer packages are inherited unchanged. This repo adds `macro_stress_forecaster`, which re-optimizes indicator weights against forward SPY drawdown labels and trains a classifier to output drawdown probability.

The series:

1. **macro-stress-pipeline**: ingests yfinance and FRED data, computes a composite stress score from 16 leading indicators, writes `stress_score.parquet`
2. **macro-stress-optimizer**: reads `stress_score.parquet`, learns optimal per-indicator weights via SLSQP to maximize AUC against coincident SPY drawdown labels, writes `optimized_weights.json`
3. **macro-stress-forecaster** (this repo): reads `stress_score.parquet`, re-runs weight optimization against forward labels (60-day horizon), trains a classifier to output drawdown probability, writes `forecast.parquet`
4. **macro-stress-dashboard**: consumes all upstream outputs and visualizes stress score, optimized weights, and forecast probabilities

## Indicators

**Yield curve (FRED)**
| Series | Indicator |
|---|---|
| `T10Y2Y` | 10Y-2Y Treasury spread |
| `T10Y3M` | 10Y-3M Treasury spread |
| `T30Y10Y` | 30Y-10Y Treasury spread (computed: `DGS30 - DGS10`) |

**Macro leading (FRED)**
| Series | Indicator |
|---|---|
| `USALOLITOAASTSAM` | OECD composite leading indicator |
| `UMCSENT` | University of Michigan consumer sentiment |
| `PERMIT` | Building permits |
| `NEWORDER` | Manufacturers new orders |

**Labor and credit (FRED)**
| Series | Indicator |
|---|---|
| `ICSA` | Initial jobless claims |
| `DRCCLACBS` | Credit card delinquency rate |
| `BAMLH0A0HYM2` | ICE BofA HY OAS spread |

**Market (yfinance)**
| Ticker | Indicator |
|---|---|
| `XLK` / `XLV` | Tech vs defensive rotation |
| `TLT` | Long-duration Treasury ETF |
| `HG=F` | Copper futures (growth proxy) |
| `CL=F` | Crude oil futures (rate-of-change; dual stress regime) |
| `EEM` | Emerging markets ETF |
| `DX=F` | DXY dollar index |

## Architecture

```
Pipeline
  fetch_data.py      pulls yfinance + FRED
  process_data.py    merges, resamples, computes ratios
  features.py        rolling percentile rank, direction flip, composite score
  pipeline.py        orchestration, writes stress_score.parquet

                     data/processed/stress_score.parquet
                                      |
                                      v
Optimizer
  labels.py          coincident SPY drawdown labels
  optimizer.py       SLSQP weight optimization, alpha sweep, CV evaluation

                     data/processed/optimized_weights.json
                                      |
                                      v
Forecaster
  labels.py          forward SPY drawdown labels (lookahead=60 days)
  forecaster.py      re-optimizes weights against forward labels, trains classifier,
                     writes forecast.parquet and forecast_weights.json
```

All three packages are installed from `src/` via `pyproject.toml`. The forecaster imports `optimize_weights` from `macro_stress_optimizer`; neither imports from `macro_stress_pipeline`. The parquet file is the only interface between the pipeline and ML layers.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JaredRudolph/macro-stress-forecaster.git
cd macro-stress-forecaster
uv sync
```

FRED requires a free API key. Create a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

Get a key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

## Usage

```bash
uv run stress-pipeline   # fetch data, compute stress score
uv run stress-optimize   # read parquet, optimize weights (coincident labels)
uv run stress-forecast   # read parquet, optimize for forward labels, train classifier
```

Outputs:

- `data/raw/market_raw.csv`: raw yfinance closes
- `data/raw/fred_raw.csv`: raw FRED series
- `data/processed/stress_score.parquet`: stress score with all ranked indicators and SPY
- `data/processed/optimized_weights.json`: per-indicator weights optimized against coincident drawdown labels
- `data/processed/forecast_weights.json`: per-indicator weights optimized against forward labels, plus CV metrics
- `data/processed/forecast.parquet`: original parquet columns plus `FORWARD_LABEL`, `FORECAST_SCORE`, and `DRAWDOWN_PROB`

## Forecaster Design

**Forward labels**: a day is labeled 1 if SPY drops at least 10% at any point within the next 60 trading days. The last 60 rows are dropped (no complete forward window).

**Weight re-optimization**: `optimize_weights(X, y_forward)` from the optimizer is called with forward labels before classifier training. The resulting weights (`FORECAST_SCORE`) are tuned for predictive rather than coincident fit and serve as an interpretable single-feature baseline.

**Leakage prevention**: the classifier uses `TimeSeriesSplit(gap=60)`, which skips 60 samples between each train and test fold so no fold's training labels overlap with the test period's forward window.

**Metrics**: AUC and Brier score. Baselines are the equal-weight stress score evaluated against forward labels.

## Notebooks

`notebooks/forecaster.ipynb` covers the full forecaster workflow: forward label construction, weight re-optimization, classifier training and CV evaluation, and probability visualization.

## Development

```bash
uv run pytest          # run tests
uv run ruff format .   # format
uv run ruff check .    # lint
```
