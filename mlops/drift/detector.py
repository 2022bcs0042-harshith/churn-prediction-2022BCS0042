import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os


def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "mlops/drift_reports"
):
    print("🔍 Running data drift detection...")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Select only numeric columns
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

    reference_data_num = reference_data[numeric_cols].copy()
    current_data_num = current_data[numeric_cols].copy()

    # Create Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data_num,
        current_data=current_data_num
    )

    # Save HTML report (WORKING VERSION)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_path, f"drift_report_{timestamp}.html")
    report.save_html(html_path)

    print(f"📄 Drift report saved to {html_path}")

    # Convert report to dictionary
    report_dict = report.as_dict()

    # Extract metrics
    metrics = report_dict.get("metrics", [])
    drift_score = 0
    drift_detected = False

    for metric in metrics:
        result = metric.get("result", {})
        if "drifted_columns" in result:
            drift_score = result.get("share_of_drifted_columns", 0)
            drift_detected = result.get("dataset_drift", False)

    # Summary
    summary = {
        "timestamp": timestamp,
        "dataset_drift_detected": drift_detected,
        "drift_share": drift_score,
        "number_of_drifted_columns": int(drift_score * len(numeric_cols)),
        "total_columns": len(numeric_cols)
    }

    print(f"📊 Drift detected: {summary['dataset_drift_detected']}")
    print(f"📊 Drift share: {summary['drift_share']:.2%}")

    return summary


def simulate_production_drift(reference_data: pd.DataFrame) -> pd.DataFrame:
    """Simulate drift by modifying feature distributions"""

    current_data = reference_data.copy()

    numeric_cols = current_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        current_data[col] = (
            current_data[col] +
            np.random.normal(
                0,
                current_data[col].std() * 0.2,
                len(current_data)
            )
        )

    return current_data


if __name__ == "__main__":
    from ml.features import load_and_preprocess

    df = load_and_preprocess(
        "ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    # Split into reference and current datasets
    reference_data = df.sample(frac=0.5, random_state=42)
    current_data = df.drop(reference_data.index)

    # Simulate drift
    current_data = simulate_production_drift(current_data)

    # Detect drift
    summary = detect_data_drift(reference_data, current_data)

    print("📊 Drift Summary:", summary)