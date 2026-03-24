import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
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
    "DX=F",
]

DRAWDOWN_THRESHOLD = 0.08
LOOKAHEAD = 60
MOM_WINDOWS = [21, 63]

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "verbosity": 0,
}

PARQUET_PATH = Path("data/processed/stress_score.parquet")
OUTPUT_PATH = Path("data/processed/forecast.parquet")


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


def run(
    parquet_path: Path = PARQUET_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Load stress_score.parquet, train XGBoost on forward labels, write forecast.parquet."""
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

    params = {**MODEL_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**params)
    model.fit(X, y)
    logger.info(
        f"Model trained on {len(X)} rows ({n_pos} positive, {n_neg} negative)"
    )

    proba = pd.Series(
        model.predict_proba(X)[:, 1], index=X.index, name="DRAWDOWN_PROB"
    )

    out = df.copy()
    out["FORWARD_LABEL"] = labels
    out["DRAWDOWN_PROB"] = proba
    out.attrs["meta"] = {
        "data_start": str(X.index[0].date()),
        "data_end": str(X.index[-1].date()),
        "n_rows": len(X),
        "drawdown_threshold": DRAWDOWN_THRESHOLD,
        "lookahead": LOOKAHEAD,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    logger.info(f"Saved forecast to {output_path}")

    return out
