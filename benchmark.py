import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 16
LATENCY_REPEATS = 50
BATCH_REPEATS = 10


def measured_seconds(model, data, repeats):
    """Return median predict_proba time after the caller performs warm-up."""
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(data)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


def main():
    started = time.perf_counter()
    df = pd.read_csv("creditcard.csv")
    data_load_seconds = time.perf_counter() - started
    x, y = df.drop(columns="Class"), df["Class"]

    # 60% train, 20% validation, 20% final test; every split is stratified.
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_trainval,
        y_trainval,
        test_size=0.25,
        random_state=SEED,
        stratify=y_trainval,
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    training_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    one_row, batch = x_test.iloc[:1], x_test.iloc[:1000]
    model.predict_proba(one_row)  # Warm-up excluded from the timings.
    model.predict_proba(batch)

    single_seconds = measured_seconds(model, one_row, LATENCY_REPEATS)
    batch_seconds = measured_seconds(model, batch, BATCH_REPEATS)
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": platform.machine(),
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_rows": len(df),
        "fraud_rows": int(y.sum()),
        "seed": SEED,
        "split": {
            "train": len(x_train),
            "validation": len(x_valid),
            "test": len(x_test),
        },
        "n_jobs": 2,
        "decision_threshold": 0.5,
        "data_load_seconds": data_load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "latency_1_row_ms": single_seconds * 1000,
        "latency_repeats": LATENCY_REPEATS,
        "batch_rows": len(batch),
        "batch_repeats": BATCH_REPEATS,
        "batch_1000_rows_seconds": batch_seconds,
        "throughput_1000_rows_per_second": len(batch) / batch_seconds,
        "timing_summary": "median; warm-up excluded; predict_proba on pandas input",
    }
    Path("benchmark_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
