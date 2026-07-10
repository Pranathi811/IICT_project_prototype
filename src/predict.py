import torch
import joblib
import numpy as np

from src.model import ChemicalYieldPredictor

MODEL_PATH = "results/model.pt"
SCALER_PATH = "results/scaler.pkl"

# Load model only once
model = ChemicalYieldPredictor()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Load scaler
scaler = joblib.load(SCALER_PATH)


def predict_yield(input_array):
    """
    input_array -> numpy array shape (1,5)
    """

    # Scale inputs
    scaled_input = scaler.transform(input_array)

    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    # Keep prediction between 0 and 100
    prediction = max(0, min(100, prediction))

    return prediction