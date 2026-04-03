import os
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_prediction_files(output_dir, preds, final_label, risk_level, summary):
    ensure_dir(output_dir)

    result_df = pd.DataFrame({
        "window_id": list(range(len(preds))),
        "pred_label": preds
    })
    result_df.to_csv(f"{output_dir}/minimal_predictions.csv", index=False)

    with open(f"{output_dir}/minimal_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Final Prediction: {final_label}\n")
        f.write(f"Risk Level: {risk_level}\n")
        f.write(f"Summary: {summary}\n")