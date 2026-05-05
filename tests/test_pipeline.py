from unittest.mock import patch

import numpy as np
import pandas as pd

from macro_stress_pipeline.pipeline import run


def make_market_df(n=800):
    idx = pd.bdate_range("2017-01-01", periods=n)
    rng = np.random.default_rng(42)
    tickers = [
        "SPY",
        "XLK",
        "XLV",
        "TLT",
        "HG=F",
        "CL=F",
        "EEM",
        "DX-Y.NYB",
    ]
    return pd.DataFrame({t: rng.random(n) * 100 for t in tickers}, index=idx)


def make_fred_df(n=800):
    idx = pd.bdate_range("2017-01-01", periods=n)
    rng = np.random.default_rng(42)
    series = [
        "T10Y2Y",
        "T10Y3M",
        "DGS30",
        "DGS10",
        "ICSA",
        "DRCCLACBS",
        "USALOLITOAASTSAM",
        "UMCSENT",
        "PERMIT",
        "NEWORDER",
        "DBAA",
    ]
    return pd.DataFrame({s: rng.random(n) for s in series}, index=idx)


def test_run_returns_dataframe(tmp_path):
    with (
        patch(
            "macro_stress_pipeline.pipeline.fetch_market_data",
            return_value=make_market_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.fetch_fred_data",
            return_value=make_fred_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.OUTPUT_PATH",
            tmp_path / "stress_score.parquet",
        ),
        patch("macro_stress_pipeline.pipeline.RAW_PATH", tmp_path / "raw"),
    ):
        result = run()

    assert isinstance(result, pd.DataFrame)


def test_run_output_has_expected_columns(tmp_path):
    with (
        patch(
            "macro_stress_pipeline.pipeline.fetch_market_data",
            return_value=make_market_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.fetch_fred_data",
            return_value=make_fred_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.OUTPUT_PATH",
            tmp_path / "stress_score.parquet",
        ),
        patch("macro_stress_pipeline.pipeline.RAW_PATH", tmp_path / "raw"),
    ):
        result = run()

    assert "STRESS_SCORE" in result.columns
    assert "SPY" in result.columns


def test_run_writes_parquet(tmp_path):
    out = tmp_path / "stress_score.parquet"
    with (
        patch(
            "macro_stress_pipeline.pipeline.fetch_market_data",
            return_value=make_market_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.fetch_fred_data",
            return_value=make_fred_df(),
        ),
        patch("macro_stress_pipeline.pipeline.OUTPUT_PATH", out),
        patch("macro_stress_pipeline.pipeline.RAW_PATH", tmp_path / "raw"),
    ):
        run()

    assert out.exists()


def test_run_writes_raw_csvs(tmp_path):
    raw = tmp_path / "raw"
    with (
        patch(
            "macro_stress_pipeline.pipeline.fetch_market_data",
            return_value=make_market_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.fetch_fred_data",
            return_value=make_fred_df(),
        ),
        patch(
            "macro_stress_pipeline.pipeline.OUTPUT_PATH",
            tmp_path / "stress_score.parquet",
        ),
        patch("macro_stress_pipeline.pipeline.RAW_PATH", raw),
    ):
        run()

    assert (raw / "market_raw.csv").exists()
    assert (raw / "fred_raw.csv").exists()
