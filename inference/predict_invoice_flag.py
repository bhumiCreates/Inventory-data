import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_invoice_flag(input_data):

    model, scaler = load_model()

    input_df = pd.DataFrame(input_data)
    
    # Scale the input data using the same scaler used during training
    input_scaled = scaler.transform(input_df)
    
    # Make predictions using the trained model
    prediction = model.predict(input_scaled)

    input_df["Predicted_Flag"] = prediction

    return input_df
