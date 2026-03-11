import joblib
from pathlib import Path

from Data_preprocessing import load_data, prepare_features, split_data
from model_evaluation import (
    train_LR,
    train_RF,
    train_DT,
    evaluate_model
)

def main():
    db_path = "data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_data(db_path)

    # Prepare data
    X, Y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Train models
    lr_model = train_LR(X_train, Y_train)
    dt_model = train_DT(X_train, Y_train)
    rf_model = train_RF(X_train, Y_train)

    # Evaluate results
    results = []
    results.append(evaluate_model(lr_model, X_test, Y_test, 'Linear Regression'))
    results.append(evaluate_model(dt_model, X_test, Y_test, 'Decision Tree Regression'))
    results.append(evaluate_model(rf_model, X_test, Y_test, 'Random Forest Regression'))

    # Select best model
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model,"models/predict_freight.pkl")

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")
    


if __name__ == "__main__":
    main()
