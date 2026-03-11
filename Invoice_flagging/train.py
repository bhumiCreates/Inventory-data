import joblib
from pathlib import Path

from model_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import load_invoice_data, apply_labels, split_data, scale_feature

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = load_invoice_data()
    df = apply_labels(df)

    X_train, X_test, Y_train, Y_test = split_data(df, FEATURES, TARGET)

    # Scale features and save scaler
    X_train_scaled, X_test_scaled = scale_feature(
        X_train, X_test, MODEL_DIR / "scaler.pkl"
    )

    grid_search = train_random_forest(X_train_scaled, Y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        Y_test,
        "Random Forest Classifier"
    )

    joblib.dump(grid_search.best_estimator_, MODEL_DIR / "predict_flag_invoice.pkl")
    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    main()