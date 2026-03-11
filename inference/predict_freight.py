import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_freight_cost(input_data):

    model = load_model()

    input_df = pd.DataFrame(input_data)

    # Select only features used during training
    input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)

    input_df["Predict_Freight"] = prediction

    return input_df
