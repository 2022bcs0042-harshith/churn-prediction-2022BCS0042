import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)
import pandas as pd
import numpy as np
from ml.features import (
    load_and_preprocess,
    simulate_ticket_features,
    get_feature_columns
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import hashlib


MODEL_NAME = "ChurnPredictionModel"


def train_and_track(
    data_path: str,
    model_output_path: str,
    experiment_name: str = "churn-prediction",
    n_estimators: int = 100,
    max_depth: int = 10
):
    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print("📦 Loading and preprocessing data...")
        df = load_and_preprocess(data_path)
        df = simulate_ticket_features(df)

        feature_cols = get_feature_columns(df)
        X = df[feature_cols]
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ── Data versioning: log MD5 hash of the CSV ──────────────
        data_hash = hashlib.md5(
            open(data_path, "rb").read()
        ).hexdigest()[:8]
        print(f"📋 Data version (hash): {data_hash}")

        # ── Log parameters ────────────────────────────────────────
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("data_version", data_hash)          # NEW
        mlflow.log_param("data_path", data_path)             # NEW

        # ── Save + log feature list as artifact ───────────────────
        feature_list_path = "feature_list.txt"
        with open(feature_list_path, "w") as f:
            f.write("\n".join(feature_cols))
        mlflow.log_artifact(feature_list_path)               # NEW
        print(f"📋 Feature list logged ({len(feature_cols)} features)")

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

        # ── Evaluate ──────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)

        # ── Log metrics ───────────────────────────────────────────
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)

        print(f"✅ F1 Score:      {f1:.4f}")
        print(f"✅ ROC-AUC:       {roc_auc:.4f}")
        print(f"✅ Precision:     {precision:.4f}")
        print(f"✅ Recall:        {recall:.4f}")

        # ── Log model to MLflow ───────────────────────────────────
        mlflow.sklearn.log_model(pipeline, "model")

        # ── Save model artifact locally ───────────────────────────
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "feature_columns": feature_cols
        }, model_output_path)
        mlflow.log_artifact(model_output_path)

        print("💾 Model saved and tracked with MLflow")

        # ── Model Registry: register model ────────────────────────
        run_id    = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        print(f"\n📝 Registering model as '{MODEL_NAME}'...")
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        version = registered.version
        print(f"✅ Model registered — Version: {version}")

        # ── Model Registry: transition to Staging ─────────────────
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False   # keep older versions visible
        )
        print(f"🚀 Model v{version} transitioned → Staging")

        return pipeline, X_test, y_test, {
            "f1_score":   round(f1, 4),
            "roc_auc":    round(roc_auc, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4)
        }


def promote_to_production(version: str = None):
    """
    Promotes a model version to Production.
    If version is None, promotes the latest Staging version.
    """
    client = MlflowClient()

    if version is None:
        staging_versions = client.get_latest_versions(
            MODEL_NAME, stages=["Staging"]
        )
        if not staging_versions:
            print("❌ No model in Staging to promote.")
            return
        version = staging_versions[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True   # archive old Production versions
    )
    print(f"✅ Model v{version} transitioned → Production")


def archive_model(version: str):
    """Archives a specific model version."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Archived"
    )
    print(f"🗄️  Model v{version} transitioned → Archived")


if __name__ == "__main__":
    # Step 1: Train, track, and register → Staging
    pipeline, X_test, y_test, metrics = train_and_track(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )

    print("\n" + "=" * 50)
    print("📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # Step 2: Promote to Production (after validation)
    print("\n" + "=" * 50)
    promote_to_production()

    print("\n✅ Done! Open MLflow UI to verify:")
    print("   mlflow ui  →  http://localhost:5000")
    print(f"   Models tab → {MODEL_NAME}")