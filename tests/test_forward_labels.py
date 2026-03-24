import numpy as np
import pandas as pd
import pytest

from macro_stress_forecaster.labels import compute_forward_drawdown_labels


def make_spy(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_no_future_drawdown_returns_all_zeros():
    spy = make_spy([100, 101, 102, 103, 104, 105, 106])
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3)
    assert (labels == 0).all()


def test_future_drawdown_above_threshold_returns_one():
    # t=0: future min over next 3 days = min(100, 80, 100) = 80, drop = -20%
    spy = make_spy([100, 100, 80, 100, 100, 100])
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3)
    assert labels.iloc[0] == 1


def test_future_drawdown_below_threshold_returns_zero():
    # t=0: future min = min(98, 97, 96), drop = -4%, under 8% threshold
    spy = make_spy([100, 98, 97, 96, 100, 100])
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3)
    assert labels.iloc[0] == 0


def test_last_lookahead_rows_are_dropped():
    spy = make_spy(list(range(100, 120)))
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=5)
    assert len(labels) == len(spy) - 5


def test_output_index_is_subset_of_input_index():
    spy = make_spy([100] * 20)
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=5)
    assert labels.index.isin(spy.index).all()


def test_output_is_binary():
    rng = np.random.default_rng(0)
    spy = make_spy(rng.random(100) * 100 + 200)
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=10)
    assert set(labels.unique()).issubset({0, 1})


def test_custom_threshold():
    # t=0: drop = -6%, labeled 1 at 5% threshold, 0 at 8%
    spy = make_spy([100, 94, 94, 100, 100, 100])
    assert compute_forward_drawdown_labels(spy, threshold=0.05, lookahead=3).iloc[0] == 1
    assert compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3).iloc[0] == 0


def test_drawdown_only_within_window():
    # Drawdown occurs at t=5, outside the lookahead window from t=0 (lookahead=3)
    spy = make_spy([100, 100, 100, 100, 50, 50, 50])
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3)
    assert labels.iloc[0] == 0


def test_event_at_edge_of_window_is_captured():
    # Drawdown occurs exactly at the last day of the window
    spy = make_spy([100, 100, 100, 80, 100, 100, 100])
    labels = compute_forward_drawdown_labels(spy, threshold=0.08, lookahead=3)
    # t=0: future min over next 3 days = min(spy[1], spy[2], spy[3]) = min(100, 100, 80) = 80
    assert labels.iloc[0] == 1
