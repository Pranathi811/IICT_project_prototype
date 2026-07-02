import torch
import numpy as np
from src.model import ChemicalYieldPredictor

MODEL_PATH = "results/model.pt"

def predict_yield(input_array):
    """
    input_array: numpy array of shape (1, 5)
    returns: predicted yield (float)
    """

    # Load model
    model = ChemicalYieldPredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()

    # Convert input to tensor
    input_tensor = torch.tensor(input_array, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prediction = model(input_tensor)

    return prediction.item()