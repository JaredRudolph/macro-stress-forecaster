import numpy as np
import pandas as pd
import pytest

from macro_stress_forecaster.forecaster import INDICATOR_COLS, build_features, run


def make_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    data = {col: rng.random(n) for col in INDICATOR_COLS}
    data["SPY"] = np.cumprod(1 + rng.normal(0.0003, 0.015, n)) * 300
    data["STRESS_SCORE"] = rng.random(n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def synthetic_parquet(tmp_path):
    df = make_df()
    path = tmp_path / "stress_score.parquet"
    df.to_parquet(path)
    return path


def test_build_features_shape():
    df = make_df()
    features = build_features(df)
    # 16 raw ranks + 16 * 2 momentum windows = 48
    assert features.shape[1] == 48


def test_build_features_columns():
    df = make_df()
    features = build_features(df)
    assert all(col in features.columns for col in INDICATOR_COLS)
    assert all(f"{col}_mom21" in features.columns for col in INDICATOR_COLS)
    assert all(f"{col}_mom63" in features.columns for col in INDICATOR_COLS)


def test_run_writes_parquet(synthetic_parquet, tmp_path):
    output_path = tmp_path / "forecast.parquet"
    run(parquet_path=synthetic_parquet, output_path=output_path)
    assert output_path.exists()


def test_run_output_has_required_columns(synthetic_parquet, tmp_path):
    output_path = tmp_path / "forecast.parquet"
    run(parquet_path=synthetic_parquet, output_path=output_path)
    out = pd.read_parquet(output_path)
    assert "FORWARD_LABEL" in out.columns
    assert "DRAWDOWN_PROB" in out.columns


def test_run_drawdown_prob_in_unit_interval(synthetic_parquet, tmp_path):
    output_path = tmp_path / "forecast.parquet"
    run(parquet_path=synthetic_parquet, output_path=output_path)
    out = pd.read_parquet(output_path)
    prob = out["DRAWDOWN_PROB"].dropna()
    assert (prob >= 0).all() and (prob <= 1).all()


def test_run_forward_label_is_binary(synthetic_parquet, tmp_path):
    output_path = tmp_path / "forecast.parquet"
    run(parquet_path=synthetic_parquet, output_path=output_path)
    out = pd.read_parquet(output_path)
    labels = out["FORWARD_LABEL"].dropna()
    assert set(labels.unique()).issubset({0, 1, 0.0, 1.0})


def test_run_preserves_input_columns(synthetic_parquet, tmp_path):
    output_path = tmp_path / "forecast.parquet"
    original = pd.read_parquet(synthetic_parquet)
    run(parquet_path=synthetic_parquet, output_path=output_path)
    out = pd.read_parquet(output_path)
    assert all(col in out.columns for col in original.columns)
