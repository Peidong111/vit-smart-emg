from collections import Counter

from config import (
    CSV_PATH,
    SAMPLING_RATE,
    CHANNEL_NAME,
    WINDOW_SEC,
    STEP_SEC,
    OUTPUT_DIR
)
from preprocess import load_signal_from_csv, sliding_window
from features import extract_feature_matrix
from utils import save_prediction_files


def summarize_predictions(preds):
    cnt = Counter(preds)
    total = sum(cnt.values())
    summary = {k: round(v / total, 3) for k, v in cnt.items()}

    final_label = max(summary, key=summary.get)

    risk_map = {
        "rest": "low_risk",
        "normal": "medium_risk",
        "fatigue": "high_risk"
    }

    risk_level = risk_map.get(final_label, "unknown")
    return summary, final_label, risk_level


def rule_predict(feature_row):
    rms, mav, std, max_abs = feature_row

    # Temporary demo thresholds
    if rms < 30:
        return "rest"
    elif rms < 80:
        return "normal"
    else:
        return "fatigue"


def main():
    print("Loading CSV:", CSV_PATH)

    signal, df = load_signal_from_csv(CSV_PATH, CHANNEL_NAME)

    print("Loaded signal length:", len(signal))
    print("Using channel:", CHANNEL_NAME)
    print("Sampling rate:", SAMPLING_RATE)

    windows, win_size, step = sliding_window(
        signal,
        SAMPLING_RATE,
        WINDOW_SEC,
        STEP_SEC
    )

    print("Window size:", win_size)
    print("Step size:", step)
    print("Number of windows:", len(windows))

    if len(windows) == 0:
        raise ValueError("No windows created. Check CSV length or sampling rate.")

    feature_matrix = extract_feature_matrix(windows)
    print("Feature shape:", feature_matrix.shape)

    preds = [rule_predict(row) for row in feature_matrix]

    print("\nWindow predictions sample:", preds[:10])

    summary, final_label, risk_level = summarize_predictions(preds)

    print("\nSummary:", summary)
    print("Final Prediction:", final_label)
    print("Risk Level:", risk_level)

    save_prediction_files(
        OUTPUT_DIR,
        preds,
        final_label,
        risk_level,
        summary
    )

    print("\nSaved:")
    print(" - outputs/minimal_predictions.csv")
    print(" - outputs/minimal_summary.txt")


if __name__ == "__main__":
    main()