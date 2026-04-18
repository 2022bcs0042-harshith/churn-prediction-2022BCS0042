import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import joblib
import os
import hashlib

from ml.features import (
    load_and_preprocess,
    simulate_ticket_features,
    get_feature_columns
)

MODEL_NAME = "ChurnPredictionModel"


def train_and_track(
    data_path: str,
    model_output_path: str,
    experiment_name: str = "churn-prediction",
    n_estimators: int = 100,
    max_depth: int = 10
):

    # ✅ FIX 1: Use local MLflow store (avoid DB issues in GitHub Actions)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        print("📦 Loading data...")
        df = load_and_preprocess(data_path)

        if df is None or df.empty:
            raise ValueError("❌ Dataset is empty after loading")

        df = simulate_ticket_features(df)

        # ✅ FIX 2: Clean Churn column safely
        if df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # Drop invalid rows
        df = df.dropna(subset=["Churn"])

        if df.empty:
            raise ValueError("❌ Dataset empty after cleaning Churn column")

        df["Churn"] = df["Churn"].astype(int)

        feature_cols = get_feature_columns(df)

        if len(feature_cols) == 0:
            raise ValueError("❌ No features found")

        X = df[feature_cols]
        y = df["Churn"]

        if len(X) == 0:
            raise ValueError("❌ No samples available for training")

        # ✅ FIX 3: Safe split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )

        print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

        # ✅ Data versioning
        data_hash = hashlib.md5(open(data_path, "rb").read()).hexdigest()[:8]

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_version", data_hash)

        print("🤖 Training model...")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        pipeline.fit(X_train, y_train)

        # ✅ Evaluation
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"✅ F1: {f1:.4f}")
        print(f"✅ ROC-AUC: {roc_auc:.4f}")

        # ✅ Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # ✅ Save model
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "features": feature_cols
        }, model_output_path)

        mlflow.log_artifact(model_output_path)

        print("💾 Model saved")

        # ✅ Register model (safe)
        try:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            registered = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME
            )

            version = registered.version

            client = MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Staging"
            )

            print(f"🚀 Model v{version} → Staging")

        except Exception as e:
            print(f"⚠️ Model registry skipped: {e}")

        return pipeline, X_test, y_test, {
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }


def promote_to_production():
    client = MlflowClient()

    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

    if not versions:
        print("❌ No model in staging")
        return

    version = versions[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production"
    )

    print(f"✅ Model v{version} → Production")


if __name__ == "__main__":

    pipeline, X_test, y_test, metrics = train_and_track(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )

    print("\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    promote_to_production()