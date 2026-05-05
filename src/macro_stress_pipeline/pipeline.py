from pathlib import Path

import pandas as pd
from loguru import logger

from macro_stress_pipeline.features import compute_stress_score
from macro_stress_pipeline.fetch_data import fetch_fred_data, fetch_market_data
from macro_stress_pipeline.process_data import merge_all

# START_DATE chosen based on availability of Index/Ticker data
# (EEM started on 2003-04-14)
START_DATE = "2003-04-14"


def get_end_date() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


MARKET_TICKERS = [
    "SPY",  # S&P 500 (overlay only)
    "XLK",  # Tech ETF (for ratio)
    "XLV",  # Healthcare ETF (for ratio)
    "TLT",  # 20Y Treasury ETF
    "HG=F",  # Copper futures
    "CL=F",  # Crude oil futures (used as 21-day ROC)
    "EEM",  # Emerging markets ETF
    "DX-Y.NYB",  # DXY dollar index
]

FRED_SERIES = [
    "T10Y2Y",  # yield curve spread (10Y-2Y)
    "T10Y3M",  # yield curve spread (10Y-3M)
    "DGS30",  # 30Y constant maturity (used to compute T30Y10Y spread)
    "DGS10",  # 10Y constant maturity (used to compute T30Y10Y spread)
    "ICSA",  # initial jobless claims
    "DRCCLACBS",  # credit card delinquency rate
    "USALOLITOAASTSAM",  # Leading Indicators OECD
    "UMCSENT",  # University of Michigan Consumer Sentiment
    "PERMIT",  # building permits
    "NEWORDER",  # manufacturers new orders
    "DBAA",  # Moody's Baa corporate bond yield (used to compute BAA_SPREAD)
]

RAW_PATH = Path("data/raw")

OUTPUT_PATH = Path("data/processed/stress_score.parquet")


def run(start: str = START_DATE, end: str = None) -> pd.DataFrame:
    end = end or get_end_date()

    logger.info("Fetching market data")
    df_market = fetch_market_data(MARKET_TICKERS, start, end)

    logger.info("Fetching FRED data")
    df_fred = fetch_fred_data(FRED_SERIES, start, end)

    RAW_PATH.mkdir(parents=True, exist_ok=True)
    df_market.to_csv(RAW_PATH / "market_raw.csv")
    df_fred.to_csv(RAW_PATH / "fred_raw.csv")
    logger.info("Saved raw data to data/raw/")

    logger.info("Merging and aligning data")
    df_merged = merge_all(df_market, df_fred)

    logger.info("Computing stress score")
    df_scored = compute_stress_score(df_merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_parquet(OUTPUT_PATH)
    logger.info(f"Saved to {OUTPUT_PATH}")

    return df_scored


def main():
    from dotenv import load_dotenv

    load_dotenv()
    run()


if __name__ == "__main__":
    main()
