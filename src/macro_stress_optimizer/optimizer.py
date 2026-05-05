import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from macro_stress_optimizer.labels import compute_drawdown_labels

INDICATOR_COLS = [
    "T10Y2Y",
    "T10Y3M",
    "T30Y10Y",
    "USALOLITOAASTSAM",
    "UMCSENT",
    "PERMIT",
    "NEWORDER",
    "ICSA",
    "DRCCLACBS",
    "BAA_SPREAD",
    "XLK_XLV",
    "TLT",
    "HG=F",
    "CL=F",
    "EEM",
    "DX-Y.NYB",
]

DRAWDOWN_THRESHOLD = 0.10
N_SPLITS = 5
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]

PARQUET_PATH = Path("data/processed/stress_score.parquet")
OUTPUT_PATH = Path("data/processed/optimized_weights.json")


def _soft_auc(weights: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
    """Differentiable AUC approximation using sigmoid on pairwise score differences.

    Steepness=10: with pre-normalized 0-1 features and weights summing to 1,
    pairwise score differences fall in [-1, 1]. At steepness=10 a difference
    of 0.1 maps to sigmoid(1.0) ~ 0.73, giving sharp enough discrimination
    without saturating the gradient on small differences.
    """
    score = (X * weights).sum(axis=1)
    pos = score[y == 1].values
    neg = score[y == 0].values
    diff = pos[:, None] - neg[None, :]
    return -np.mean(1 / (1 + np.exp(-diff * 10)))


def _fit_slsqp(
    X_train: pd.DataFrame, y_train: pd.Series, alpha: float, equal_weights: np.ndarray
) -> np.ndarray:
    n = len(equal_weights)
    bounds = [(-0.10, 0.25)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    def objective(w):
        return _soft_auc(w, X_train, y_train) + alpha * ((w - equal_weights) ** 2).sum()

    result = minimize(
        objective,
        equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x


def _cv_auc(
    X: pd.DataFrame, y: pd.Series, alpha: float, n_splits: int = N_SPLITS
) -> dict:
    """Run time series CV for a given alpha. Returns mean train AUC, test AUC, and
    gap."""
    n = len(X.columns)
    equal_weights = np.ones(n) / n
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_aucs, test_aucs = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        weights = _fit_slsqp(X_train, y_train, alpha, equal_weights)
        train_aucs.append(roc_auc_score(y_train, (X_train * weights).sum(axis=1)))
        test_aucs.append(roc_auc_score(y_test, (X_test * weights).sum(axis=1)))

    mean_test = float(np.mean(test_aucs))
    mean_gap = float(np.mean([tr - te for tr, te in zip(train_aucs, test_aucs)]))
    return {"alpha": alpha, "mean_test_auc": mean_test, "mean_gap": mean_gap}


def optimize_weights(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list[float] = None,
    n_jobs: int = None,
) -> dict:
    """Learn per-indicator weights by maximizing CV AUC against drawdown labels.

    Runs a parallel alpha sweep, picks the alpha that best balances test AUC and
    generalization gap, fits a final model on the full dataset, and returns a dict
    ready to serialize as optimized_weights.json.
    """
    if alphas is None:
        alphas = ALPHAS
    if n_jobs is None:
        n_jobs = os.cpu_count()

    n = len(X.columns)
    equal_weights = np.ones(n) / n

    # Equal-weight CV AUC: weights are fixed, so only score each test fold.
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    equal_aucs = []
    for _, test_idx in tscv.split(X):
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue
        equal_aucs.append(roc_auc_score(y_test, (X_test * equal_weights).sum(axis=1)))
    auc_equal = float(np.mean(equal_aucs))
    logger.info(f"equal-weight CV test AUC: {auc_equal:.4f}")

    # Alpha sweep: penalize the gap so we don't pick an alpha that scores well on
    # one lucky fold but still overfits badly on average.
    sweep = Parallel(n_jobs=n_jobs)(delayed(_cv_auc)(X, y, a) for a in alphas)
    best = max(sweep, key=lambda r: r["mean_test_auc"] - abs(r["mean_gap"]))
    best_alpha = best["alpha"]
    logger.info(
        f"best alpha: {best_alpha}  CV test AUC: {best['mean_test_auc']:.4f}"
        f"  gap: {best['mean_gap']:.4f}"
    )

    # Final model trained on the full dataset.
    final_weights = _fit_slsqp(X, y, best_alpha, equal_weights)
    auc_optimized = best["mean_test_auc"]

    return {
        "weights": dict(zip(X.columns.tolist(), final_weights.round(6).tolist())),
        "auc_equal": round(auc_equal, 4),
        "auc_optimized": round(auc_optimized, 4),
        "best_alpha": best_alpha,
    }


def run(
    parquet_path: Path = PARQUET_PATH,
    output_path: Path = OUTPUT_PATH,
    alphas: list[float] = None,
) -> dict:
    """Load stress_score.parquet, optimize weights, write optimized_weights.json."""
    logger.info(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)

    X = df[INDICATOR_COLS].dropna()
    y = compute_drawdown_labels(df.loc[X.index, "SPY"], threshold=DRAWDOWN_THRESHOLD)
    logger.info(
        f"{len(X)} rows after dropping warmup NaNs;"
        f" {y.sum()} drawdown days ({y.mean():.1%})"
    )

    result = optimize_weights(X, y, alphas=alphas)
    result["meta"] = {
        "data_start": str(X.index[0].date()),
        "data_end": str(X.index[-1].date()),
        "n_rows": len(X),
        "drawdown_threshold": DRAWDOWN_THRESHOLD,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved weights to {output_path}")

    return result
