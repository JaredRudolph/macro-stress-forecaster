import pandas as pd
import pytest

from macro_stress_pipeline.process_data import compute_ratios, merge_all, resample_fred


def make_market_df(n=30):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "SPY": 300.0,
            "XLK": 100.0,
            "XLV": 80.0,
            "TLT": 90.0,
            "HG=F": 4.0,
            "CL=F": 70.0,
            "EEM": 45.0,
            "DX-Y.NYB": 103.0,
        },
        index=idx,
    )


def make_fred_df():
    idx = pd.date_range("2020-01-01", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "T10Y2Y": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "DGS30": [2.5, 2.4, 2.3, 2.2, 2.1, 2.0],
            "DGS10": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
            "DBAA": [5.5, 5.4, 5.3, 5.2, 5.1, 5.0],
        },
        index=idx,
    )


def test_resample_fred_produces_daily_index():
    result = resample_fred(make_fred_df())
    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) > 6


def test_resample_fred_no_nans_after_ffill():
    result = resample_fred(make_fred_df())
    assert result["T10Y2Y"].isna().sum() == 0


def test_resample_fred_preserves_values():
    result = resample_fred(make_fred_df())
    assert result["T10Y2Y"].iloc[0] == pytest.approx(0.5)


def test_compute_ratios_columns_present():
    result = compute_ratios(make_market_df())
    assert {"XLK_XLV"}.issubset(result.columns)


def test_compute_ratios_values_correct():
    df = make_market_df()
    result = compute_ratios(df)
    pd.testing.assert_series_equal(
        result["XLK_XLV"], df["XLK"] / df["XLV"], check_names=False
    )


def test_merge_all_preserves_market_index():
    df_market = make_market_df(30)
    result = merge_all(df_market, make_fred_df())
    assert list(result.index) == list(df_market.index)


def test_merge_all_contains_expected_columns():
    result = merge_all(make_market_df(30), make_fred_df())
    for col in [
        "SPY",
        "TLT",
        "HG=F",
        "CL=F",
        "EEM",
        "DX-Y.NYB",
        "XLK_XLV",
        "T10Y2Y",
        "T30Y10Y",
        "BAA_SPREAD",
    ]:
        assert col in result.columns, f"Missing column: {col}"


def test_merge_all_no_nans_after_first_fred_observation():
    # fred starts before market data so ffill covers the full index
    idx = pd.date_range("2019-12-01", periods=6, freq="ME")
    df_fred = pd.DataFrame(
        {
            "T10Y2Y": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "DGS30": [2.5, 2.4, 2.3, 2.2, 2.1, 2.0],
            "DGS10": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
            "DBAA": [5.5, 5.4, 5.3, 5.2, 5.1, 5.0],
        },
        index=idx,
    )
    result = merge_all(make_market_df(30), df_fred)
    assert result["T10Y2Y"].isna().sum() == 0
