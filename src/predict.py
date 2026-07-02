import os
import argparse
import pickle
import numpy as np
import torch

from model import ChemicalYieldPredictor

def predict_yield(temperature, pressure, reaction_time, concentration, catalyst, project_root):
    """
    Predicts the chemical process yield for a set of raw process parameters.
    
    Parameters:
    - temperature (float): Temperature in °C
    - pressure (float): Pressure in bar
    - reaction_time (float): Reaction time in hours
    - concentration (float): Concentration in M
    - catalyst (float): Catalyst amount in wt%
    - project_root (str): Root directory of the project
    
    Returns:
    - float: Predicted yield (%)
    """
    results_dir = os.path.join(project_root, 'results')
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    model_path = os.path.join(results_dir, 'best_model.pth')
    
    # 1. Verify existence of models and scalers
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or scaler not found. Please train the model by running main.py first.")
        
    # 2. Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # 3. Load trained model
    model = ChemicalYieldPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # 4. Prepare and scale inputs
    raw_inputs = np.array([[temperature, pressure, reaction_time, concentration, catalyst]])
    scaled_inputs = scaler.transform(raw_inputs)
    
    # 5. Convert to tensor and run inference
    inputs_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
    with torch.no_grad():
        predicted_yield_tensor = model(inputs_tensor)
        predicted_yield = predicted_yield_tensor.item()
        
    return predicted_yield

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(description="ML-Based Decision-Support System for Chemical Process Yield Prediction")
    parser.add_argument('--temp', type=float, default=180.0, help="Reaction Temperature in °C (typical range: 100 to 250)")
    parser.add_argument('--pressure', type=float, default=30.0, help="System Pressure in bar (typical range: 1 to 50)")
    parser.add_argument('--time', type=float, default=6.0, help="Reaction Time in hours (typical range: 0.5 to 10)")
    parser.add_argument('--conc', type=float, default=2.2, help="Reactant Concentration in M (typical range: 0.5 to 3.0)")
    parser.add_argument('--cat', type=float, default=1.0, help="Catalyst Amount in wt% (typical range: 0.1 to 2.0)")
    
    args = parser.parse_args()
    
    print("\n" + "="*55)
    print("  CHEMICAL PROCESS YIELD IN SILICO DECISION-SUPPORT  ")
    print("="*55)
    print("Proposed Input Parameters:")
    print(f" - Temperature:   {args.temp:.1f} °C")
    print(f" - Pressure:      {args.pressure:.1f} bar")
    print(f" - Reaction Time: {args.time:.2f} hours")
    print(f" - Concentration: {args.conc:.2f} M")
    print(f" - Catalyst:      {args.cat:.2f} wt%")
    print("-"*55)
    
    try:
        pred_yield = predict_yield(args.temp, args.pressure, args.time, args.conc, args.cat, project_root)
        print(f"PREDICTED CHEMICAL YIELD: {pred_yield:.2f} %")
        print("-"*55)
        
        # Decision-support recommendations
        if pred_yield >= 85.0:
            print("Decision: HIGHLY RECOMMENDED PARAMETERS.")
            print("Action: Safe to proceed with production run. Predicted yield exceeds the standard QC threshold.")
        elif pred_yield >= 70.0:
            print("Decision: ACCEPTABLE PARAMETERS (Marginal).")
            print("Action: Consider tuning parameters (e.g., adjust temperature closer to 180°C or pressure to 30 bar) to optimize yield.")
        else:
            print("Decision: REJECTED PARAMETERS.")
            print("Action: DO NOT proceed. High likelihood of sub-optimal reaction, QC rejection, or high recycle fraction.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
    print("="*55 + "\n")

if __name__ == '__main__':
    main()
