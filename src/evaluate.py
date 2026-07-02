import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.model import ChemicalYieldPredictor

def load_metrics_and_plot(project_root, train_loss, val_loss):
    """
    Saves and plots the training and validation loss curves.
    """
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Model Training & Validation Loss History', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'loss_history.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Loss history plot saved to: {plot_path}")

def evaluate_model_performance(project_root):
    """
    Evaluates the model on the test dataset, computes metrics, 
    and saves diagnostic plots (Actual vs Predicted, Residuals).
    """
    results_dir = os.path.join(project_root, 'results')
    
    # 1. Load test data and scaler
    test_path = os.path.join(results_dir, 'test_data_summary.pkl')
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    model_path = os.path.join(results_dir, 'best_model.pth')
    
    if not (os.path.exists(test_path) and os.path.exists(model_path)):
        print("Required test data or model path not found. Cannot evaluate.")
        return None
        
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_test_scaled = test_data['X_test_scaled']
    y_true = test_data['y_test']
    
    # 2. Load model
    model = ChemicalYieldPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 3. Make predictions
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
        y_pred = y_pred_tensor.numpy().flatten()
        
    # 4. Compute Regression Metrics
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    # R-squared (R2)
    y_bar = np.mean(y_true)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    metrics = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MSE': mse}
    
    # Save metrics to text file
    metrics_path = os.path.join(results_dir, 'metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write("==================================================\n")
        f.write("      ML CHEMICAL PROCESS YIELD METRICS REPORT    \n")
        f.write("==================================================\n")
        f.write(f"R-squared (R2) Score:  {r2:.5f} (Percent variance explained: {r2*100:.2f}%)\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f} %\n")
        f.write(f"Mean Squared Error (MSE):  {mse:.4f} %^2\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f} %\n")
        f.write("==================================================\n")
    print(f"Metrics report written to: {metrics_path}")
    
    # 5. Plot Actual vs Predicted Scatter Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#2ca02c', edgecolors='k', s=40)
    # 45-degree reference line
    min_val = min(y_true.min(), y_pred.min()) - 2
    max_val = max(y_true.max(), y_pred.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (Y = Ŷ)')
    
    plt.xlabel('Experimental (Actual) Yield (%)', fontsize=12)
    plt.ylabel('Model Predicted Yield (%)', fontsize=12)
    plt.title('Experimental vs. Predicted Chemical Yield', fontsize=13, fontweight='bold', pad=15)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    scatter_path = os.path.join(results_dir, 'actual_vs_predicted.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    # 6. Plot Residual Distribution
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='#9467bd', edgecolor='black', bins=25)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.xlabel('Residual (Actual - Predicted) (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residual Error Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    residual_path = os.path.join(results_dir, 'residual_distribution.png')
    plt.savefig(residual_path, dpi=300)
    plt.close()
    
    print(f"Evaluation plots saved successfully in: {results_dir}")
    return metrics

def plot_parameter_sensitivity(project_root):
    """
    Performs sensitivity analysis on Temperature to demonstrate the model's physical comprehension
    of the kinetics optimum (180°C). Keeps all other variables constant at optimal/mean values:
    Pressure = 30 bar, Time = 6 hrs, Concentration = 2.2 M, Catalyst = 1.0 wt%
    """
    results_dir = os.path.join(project_root, 'results')
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    model_path = os.path.join(results_dir, 'best_model.pth')
    
    if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
        return
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    model = ChemicalYieldPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate range of temperatures
    temp_range = np.linspace(100.0, 250.0, 150)
    
    # Keep others constant
    constant_pressure = 30.0
    constant_time = 6.0
    constant_concentration = 2.2
    constant_catalyst = 1.0
    
    # Assemble input matrix
    sensitivity_inputs = np.zeros((len(temp_range), 5))
    sensitivity_inputs[:, 0] = temp_range
    sensitivity_inputs[:, 1] = constant_pressure
    sensitivity_inputs[:, 2] = constant_time
    sensitivity_inputs[:, 3] = constant_concentration
    sensitivity_inputs[:, 4] = constant_catalyst
    
    # Scale inputs
    scaled_inputs = scaler.transform(sensitivity_inputs)
    
    # Predict
    input_tensor = torch.tensor(scaled_inputs, dtype=torch.float32)
    with torch.no_grad():
        predicted_yields = model(input_tensor).numpy().flatten()
        
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(temp_range, predicted_yields, color='#d62728', linewidth=2.5, label='ML Model Prediction')
    plt.axvline(180.0, color='blue', linestyle=':', linewidth=1.5, label='Theoretical Kinetics Optimum (180°C)')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Process Yield (%)', fontsize=12)
    plt.title('Model Temperature Sensitivity Analysis', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    sensitivity_path = os.path.join(results_dir, 'temperature_sensitivity.png')
    plt.savefig(sensitivity_path, dpi=300)
    plt.close()
    print(f"Temperature sensitivity plot saved to: {sensitivity_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(__file__))
    evaluate_model_performance(project_root)
    plot_parameter_sensitivity(project_root)
