import pandas as pd


def resample_fred(df_fred: pd.DataFrame) -> pd.DataFrame:
    return df_fred.resample("D").last().ffill()


RATIO_INPUTS = {"XLK", "XLV"}

# 21 trading days (~1 month): CL=F is used as rate of change, not price level
CL_ROC_WINDOW = 21


def compute_ratios(df_market: pd.DataFrame) -> pd.DataFrame:
    missing = RATIO_INPUTS - set(df_market.columns)
    if missing:
        raise KeyError(f"compute_ratios: missing columns: {sorted(missing)}")
    ratios = pd.DataFrame(index=df_market.index)
    ratios["XLK_XLV"] = df_market["XLK"] / df_market["XLV"]
    return ratios


def merge_all(df_market: pd.DataFrame, df_fred: pd.DataFrame) -> pd.DataFrame:
    df_fred_daily = resample_fred(df_fred)
    ratios = compute_ratios(df_market)

    combined = df_market.join(df_fred_daily, how="left")
    combined = combined.join(ratios, how="left", rsuffix="_ratio")
    combined = combined.ffill()
    combined["T30Y10Y"] = combined["DGS30"] - combined["DGS10"]
    combined["BAA_SPREAD"] = combined["DBAA"] - combined["DGS10"]
    combined["CL=F"] = combined["CL=F"].pct_change(CL_ROC_WINDOW)
    return combined


if __name__ == "__main__":
    from dotenv import load_dotenv

    from macro_stress_pipeline.fetch_data import fetch_fred_data, fetch_market_data
    from macro_stress_pipeline.pipeline import (
        FRED_SERIES,
        MARKET_TICKERS,
        START_DATE,
        get_end_date,
    )

    load_dotenv()

    df_market = fetch_market_data(MARKET_TICKERS, START_DATE, get_end_date())
    df_fred = fetch_fred_data(FRED_SERIES, START_DATE, get_end_date())
    df = merge_all(df_market, df_fred)

    print(df.head())
    print(df.dtypes)
    print(df.isna().sum())
