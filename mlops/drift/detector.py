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

    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Select only numeric columns
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for drift detection.")

        reference_data_num = reference_data[numeric_cols].copy()
        current_data_num = current_data[numeric_cols].copy()

        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_data_num,
            current_data=current_data_num
        )

        # Save HTML report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(output_path, f"drift_report_{timestamp}.html")
        report.save_html(html_path)

        print(f"📄 Drift report saved to {html_path}")

        # Convert report to dictionary
        report_dict = report.as_dict()

        # Extract metrics safely
        drift_score = 0
        drift_detected = False

        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})

            # More robust extraction
            if "dataset_drift" in result:
                drift_detected = result.get("dataset_drift", False)
                drift_score = result.get("share_of_drifted_columns", 0)

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

    except Exception as e:
        print("❌ Error during drift detection:", str(e))
        return None


def simulate_production_drift(reference_data: pd.DataFrame) -> pd.DataFrame:
    """Simulate drift by modifying feature distributions"""

    current_data = reference_data.copy()

    numeric_cols = current_data.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        std = current_data[col].std()

        # Avoid NaN or zero std issues
        if pd.isna(std) or std == 0:
            continue

        current_data[col] = current_data[col] + np.random.normal(
            0,
            std * 0.2,
            len(current_data)
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