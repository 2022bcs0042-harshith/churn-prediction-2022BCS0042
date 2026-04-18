import os
import hashlib
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ml.features import (
    load_and_preprocess,
    simulate_ticket_features,
    get_feature_columns
)

MODEL_NAME = "ChurnPredictionModel"

# ✅ FIX: Use file-based MLflow (no DB issues)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_and_track(
    data_path: str,
    model_output_path: str,
    experiment_name: str = "churn-prediction",
    n_estimators: int = 100,
    max_depth: int = 10
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print("📦 Loading and preprocessing data...")

        df = load_and_preprocess(data_path)
        df = simulate_ticket_features(df)

        # ✅ FIX: Handle TotalCharges safely (no chained assignment)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # ✅ FIX: Clean target column
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # Drop invalid rows
        df = df.dropna(subset=["Churn"])

        feature_cols = get_feature_columns(df)

        X = df[feature_cols]
        y = df["Churn"].astype(int)

        # ✅ Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # ✅ Data versioning
        data_hash = hashlib.md5(open(data_path, "rb").read()).hexdigest()[:8]
        print(f"📋 Data version: {data_hash}")

        # ✅ Log params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_version", data_hash)
        mlflow.log_param("n_features", len(feature_cols))

        # ✅ Log feature list
        feature_list_path = "feature_list.txt"
        with open(feature_list_path, "w") as f:
            f.write("\n".join(feature_cols))
        mlflow.log_artifact(feature_list_path)

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

        # ✅ Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Ensure correct types
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)

        # ✅ Metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # ✅ Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"✅ F1 Score:  {f1:.4f}")
        print(f"✅ ROC-AUC:   {roc_auc:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall:    {recall:.4f}")

        # ✅ Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # ✅ Save locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "feature_columns": feature_cols
        }, model_output_path)

        mlflow.log_artifact(model_output_path)

        print("💾 Model saved")

        # ✅ Register model
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        print(f"\n📝 Registering model as '{MODEL_NAME}'...")
        registered = mlflow.register_model(model_uri, MODEL_NAME)

        version = registered.version
        print(f"✅ Registered version: {version}")

        # ✅ Move to staging
        client = MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )

        print(f"🚀 Model v{version} → Staging")

        return pipeline, X_test, y_test, {
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }


def promote_to_production(version=None):
    client = MlflowClient()

    if version is None:
        staging = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not staging:
            print("❌ No model in staging")
            return
        version = staging[0].version

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"✅ Model v{version} → Production")


if __name__ == "__main__":
    pipeline, X_test, y_test, metrics = train_and_track(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )

    print("\n📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n🚀 Promoting model...")
    promote_to_production()

    print("\n✅ DONE")