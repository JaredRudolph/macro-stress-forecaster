import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from macro_stress_forecaster.labels import compute_forward_drawdown_labels

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
    "BAMLH0A0HYM2",
    "XLK_XLV",
    "TLT",
    "HG=F",
    "CL=F",
    "EEM",
    "DX-Y.NYB",
]

DRAWDOWN_THRESHOLD = 0.08
LOOKAHEAD = 60
MOM_WINDOWS = [21, 63]
N_SPLITS = 5

PARAM_GRID = {
    "n_estimators": [75, 100, 150, 200, 300],
    "max_depth": [2],
    "learning_rate": [0.005, 0.01, 0.02],
    "reg_alpha": [0.1, 1.0, 5.0],
    "reg_lambda": [1.0, 5.0],
    "half_life": [126, 252, 504, None],
}

FIXED_PARAMS = {
    "subsample": 0.6,
    "colsample_bytree": 0.4,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "verbosity": 0,
}

PARQUET_PATH = Path("data/processed/stress_score.parquet")
OUTPUT_PATH = Path("data/processed/forecast.parquet")


def compute_sample_weights(index: pd.DatetimeIndex, half_life: int) -> np.ndarray:
    """Exponential decay weights relative to the last date in index.
    Most recent sample = 1.0, weight halves every half_life calendar days."""
    days_back = (index[-1] - index).days
    return np.exp(-np.log(2) / half_life * days_back)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw ranks plus 21d and 63d momentum per indicator (48 features total)."""
    ranks = df[INDICATOR_COLS]
    frames = [ranks]
    for w in MOM_WINDOWS:
        frames.append(
            (ranks - ranks.shift(w)).rename(
                columns={c: f"{c}_mom{w}" for c in INDICATOR_COLS}
            )
        )
    return pd.concat(frames, axis=1)


def _run_cv(
    params: dict, X: pd.DataFrame, y: pd.Series, half_life: int | None = None
) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=LOOKAHEAD)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        sw = compute_sample_weights(X_train.index, half_life) if half_life else None
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sw)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        folds.append(
            {
                "fold": fold + 1,
                "test_start": X.index[test_idx[0]].date(),
                "test_end": X.index[test_idx[-1]].date(),
                "train_auc": roc_auc_score(y_train, train_proba),
                "auc": roc_auc_score(y_test, test_proba),
                "brier": brier_score_loss(y_test, test_proba),
            }
        )
    return folds


def _sweep_job(combo: dict, X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict:
    half_life = combo.get("half_life")
    xgb_combo = {k: v for k, v in combo.items() if k != "half_life"}
    params = {**xgb_combo, **FIXED_PARAMS, "scale_pos_weight": scale_pos_weight}
    folds = _run_cv(params, X, y, half_life=half_life)
    aucs = [f["auc"] for f in folds]
    return {
        **combo,
        "mean_train_auc": np.mean([f["train_auc"] for f in folds]),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "mean_brier": np.mean([f["brier"] for f in folds]),
    }


def tune_hyperparams(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict:
    """Parallel sweep over PARAM_GRID; returns best params by mean_auc - std_auc."""
    combos = [
        {
            "n_estimators": n,
            "max_depth": d,
            "learning_rate": lr,
            "reg_alpha": a,
            "reg_lambda": l,
            "half_life": hl,
        }
        for n, d, lr, a, l, hl in product(
            PARAM_GRID["n_estimators"],
            PARAM_GRID["max_depth"],
            PARAM_GRID["learning_rate"],
            PARAM_GRID["reg_alpha"],
            PARAM_GRID["reg_lambda"],
            PARAM_GRID["half_life"],
        )
    ]
    logger.info(f"Sweeping {len(combos)} param combinations on {os.cpu_count()} cores")
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(_sweep_job)(c, X, y, scale_pos_weight) for c in combos
    )
    best = max(results, key=lambda r: r["mean_auc"] - r["std_auc"])
    logger.info(
        f"Best params: n_estimators={best['n_estimators']}  max_depth={best['max_depth']}"
        f"  learning_rate={best['learning_rate']}  reg_alpha={best['reg_alpha']}"
        f"  reg_lambda={best['reg_lambda']}  half_life={best['half_life']}"
        f"  AUC={best['mean_auc']:.4f} +/- {best['std_auc']:.4f}"
        f"  Brier={best['mean_brier']:.4f}"
    )
    return {
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "learning_rate": best["learning_rate"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "half_life": best["half_life"],
    }


def run(
    parquet_path: Path = PARQUET_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load stress_score.parquet, tune XGBoost via CV, train on full dataset, write
    forecast.parquet."""
    logger.info(f"Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)

    labels = compute_forward_drawdown_labels(
        df["SPY"], threshold=DRAWDOWN_THRESHOLD, lookahead=LOOKAHEAD
    )
    logger.info(
        f"{len(labels)} labeled rows; {labels.sum()} forward drawdown events"
        f" ({labels.mean():.1%})"
    )

    features = build_features(df)
    X = features.loc[labels.index].dropna()
    y = labels.loc[X.index]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos

    # Baseline: equal-weight stress score evaluated against forward labels via CV.
    equal_score = df[INDICATOR_COLS].loc[X.index].mean(axis=1)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=LOOKAHEAD)
    baseline_aucs, baseline_briers = [], []
    for _, test_idx in tscv.split(X):
        y_test = y.iloc[test_idx]
        eq_test = equal_score.iloc[test_idx]
        if y_test.nunique() < 2:
            continue
        baseline_aucs.append(roc_auc_score(y_test, eq_test))
        baseline_briers.append(brier_score_loss(y_test, eq_test))
    logger.info(
        f"Baseline (equal weight)  AUC={np.mean(baseline_aucs):.4f}"
        f"  Brier={np.mean(baseline_briers):.4f}"
    )

    best_combo = tune_hyperparams(X, y, scale_pos_weight)
    half_life = best_combo["half_life"]
    xgb_combo = {k: v for k, v in best_combo.items() if k != "half_life"}
    best_params = {**xgb_combo, **FIXED_PARAMS, "scale_pos_weight": scale_pos_weight}

    # Final CV with best params for honest metrics.
    best_folds = _run_cv(best_params, X, y, half_life=half_life)
    for f in best_folds:
        logger.info(
            f"  fold {f['fold']} ({f['test_start']} to {f['test_end']})"
            f"  AUC={f['auc']:.4f}  Brier={f['brier']:.4f}"
        )
    cv_auc = np.mean([f["auc"] for f in best_folds])
    cv_brier = np.mean([f["brier"] for f in best_folds])
    logger.info(
        f"XGBoost (tuned)          AUC={cv_auc:.4f}  Brier={cv_brier:.4f}"
        f"  (vs baseline AUC={np.mean(baseline_aucs):.4f})"
    )

    sw = compute_sample_weights(X.index, half_life) if half_life else None
    model = XGBClassifier(**best_params)
    model.fit(X, y, sample_weight=sw)
    logger.info(f"Model trained on {len(X)} rows ({n_pos} positive, {n_neg} negative)")

    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="DRAWDOWN_PROB")

    out = df.copy()
    out["FORWARD_LABEL"] = labels
    out["DRAWDOWN_PROB"] = proba
    out.attrs["meta"] = {
        "data_start": str(X.index[0].date()),
        "data_end": str(X.index[-1].date()),
        "n_rows": len(X),
        "drawdown_threshold": DRAWDOWN_THRESHOLD,
        "lookahead": LOOKAHEAD,
        "cv_auc": round(cv_auc, 4),
        "cv_brier": round(cv_brier, 4),
        "baseline_auc": round(float(np.mean(baseline_aucs)), 4),
        "half_life": half_life,
        "best_params": xgb_combo,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    logger.info(f"Saved forecast to {output_path}")

    return out
