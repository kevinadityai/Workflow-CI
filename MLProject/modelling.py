"""Logistic Regression training for the MLflow Project (CI re-training).

Run through MLflow Project (the run is created by `mlflow run`, so no tracking URI / experiment is set here):
    mlflow run . --env-manager=local
"""

import argparse
import os

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

TARGET = "deposit"


def load_split(data_dir, name):
    df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
    return df.drop(columns=[TARGET]), df[TARGET]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="bankmarketing_preprocessing")
    parser.add_argument("--C", type=float, default=0.001)
    parser.add_argument("--penalty", default="l1")
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    X_train, y_train = load_split(args.data_dir, "train")
    X_test, y_test = load_split(args.data_dir, "test")

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = LogisticRegression(
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            max_iter=args.max_iter,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        })
        print(f"Run ID: {run.info.run_id} | test recall: {recall_score(y_test, pred):.4f}")
