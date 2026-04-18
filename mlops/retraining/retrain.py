import joblib
from datetime import datetime
from sklearn.metrics import f1_score

from mlops.experiment_tracking import train_and_track
from mlops.drift.detector import (
    detect_data_drift,
    simulate_production_drift
)
from ml.features import (
    load_and_preprocess,
    simulate_ticket_features
)

# ---------------- CONFIG ---------------- #
DRIFT_THRESHOLD = 0.3
F1_THRESHOLD = 0.70

DATA_PATH = "ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "ml/models/model.pkl"


# ---------------- MAIN FUNCTION ---------------- #
def check_and_retrain():
    print(f"\n{'=' * 50}")
    print(f"🔄 Running retraining check at {datetime.now()}")
    print(f"{'=' * 50}")

    # 1. Load and preprocess data
    print("📦 Loading data...")
    df = load_and_preprocess(DATA_PATH)
    df = simulate_ticket_features(df)

    # 2. Create reference & current datasets
    reference = df.sample(1000, random_state=42)
    current = simulate_production_drift(df.sample(1000, random_state=99))

    # 3. Split features and target
    ref_features = reference.drop(columns=["Churn"])
    cur_features = current.drop(columns=["Churn"])

    y_true = current["Churn"]

    # 🔥 IMPORTANT FIX: Ensure binary target
    y_true = (y_true > 0.5).astype(int)

    # 4. Load trained model
    print("📥 Loading existing model...")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    # 5. Predict on current data
    y_pred = pipeline.predict(cur_features)

    # 6. Evaluate model performance
    f1 = f1_score(y_true, y_pred)
    print(f"\n📉 Current Model Performance:")
    print(f"   F1 Score: {f1:.4f}")

    # 7. Detect data drift
    print("\n🔍 Checking for data drift...")
    drift_summary = detect_data_drift(ref_features, cur_features)

    drift_detected = drift_summary["dataset_drift_detected"]
    drift_share = drift_summary["drift_share"]

    print(f"\n📊 Drift Results:")
    print(f"   Drift detected: {drift_detected}")
    print(f"   Drift share:    {drift_share:.2%}")

    # 8. Decision logic
    retrain_due_to_drift = drift_detected and drift_share > DRIFT_THRESHOLD
    retrain_due_to_performance = f1 < F1_THRESHOLD

    if retrain_due_to_drift or retrain_due_to_performance:
        print("\n⚠️ Retraining triggered due to:")

        if retrain_due_to_drift:
            print(f"   - High drift ({drift_share:.2%})")

        if retrain_due_to_performance:
            print(f"   - Low F1 score ({f1:.4f})")

        print("\n🚀 Starting retraining...")

        pipeline, X_test, y_test, metrics = train_and_track(
            data_path=DATA_PATH,
            model_output_path=MODEL_PATH,
            experiment_name="churn-prediction-retrain"
        )

        print("\n✅ Retraining complete!")
        print(f"   New F1 Score: {metrics['f1_score']}")
        print(f"   ROC-AUC:      {metrics['roc_auc']}")

        return {
            "action": "retrained",
            "drift_share": drift_share,
            "old_f1": f1,
            "new_metrics": metrics
        }

    else:
        print("\n✅ No retraining needed")
        print("   Model is stable and performing well")

        return {
            "action": "skipped",
            "drift_share": drift_share,
            "f1": f1
        }


# ---------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    result = check_and_retrain()

    print(f"\n📋 FINAL RESULT: {result['action'].upper()}")