import numpy as np
import pandas as pd


def compute_forward_drawdown_labels(
    spy: pd.Series,
    threshold: float = 0.08,
    lookahead: int = 60,
) -> pd.Series:
    """Return binary series: 1 if SPY drops at least `threshold` at any point within
    the next `lookahead` trading days. The last `lookahead` rows have no complete
    forward window and are dropped from the output."""
    future_min = (
        spy[::-1].rolling(window=lookahead, min_periods=lookahead).min()[::-1].shift(-1)
    )
    future_drop = future_min / spy - 1

    labels = pd.Series(np.nan, index=spy.index)
    valid = future_min.notna()
    labels[valid] = (future_drop[valid] <= -threshold).astype(float)
    return labels.dropna().astype(int)
