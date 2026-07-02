import os
import torch
import numpy as np

# Import modular pipeline steps
from src.generate_data import generate_chemical_dataset
from src.data_preprocessing import preprocess_and_split_data
from src.model import ChemicalYieldPredictor
from src.train import train_model
from src.evaluate import load_metrics_and_plot, evaluate_model_performance, plot_parameter_sensitivity
from src.predict import predict_yield

def run_pipeline():
    """
    Orchestrates the entire ML pipeline from data generation to evaluation and inference.
    """
    # 1. Paths & Directories Setup
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'chemical_process_data.csv')
    model_save_path = os.path.join(results_dir, 'best_model.pth')
    
    print("="*65)
    print("  ML-BASED CHEMICAL PROCESS YIELD PREDICTION PIPELINE  ")
    print("="*65)
    
    # 2. Step 1: Generate Dataset
    print("\n--- STEP 1: Generating Dataset ---")
    if not os.path.exists(csv_path):
        df = generate_chemical_dataset(num_samples=2000, random_seed=42)
        df.to_csv(csv_path, index=False)
        print(f"Dataset generated with 2,000 samples and saved to: {csv_path}")
    else:
        print(f"Dataset already exists at: {csv_path}. Skipping generation.")
        
    # 3. Step 2: Data Preprocessing
    print("\n--- STEP 2: Preprocessing & Data Loading ---")
    train_loader, val_loader, test_loader, scaler = preprocess_and_split_data(
        csv_path=csv_path, results_dir=results_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    
    # 4. Step 3: Model Setup & Device Selection
    print("\n--- STEP 3: Initializing PyTorch MLP Regressor ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChemicalYieldPredictor(input_dim=5, dropout_rate=0.1)
    print(model)
    
    # 5. Step 4: Model Training
    print("\n--- STEP 4: Training & Validation Loop ---")
    # Using 150 epochs with early stopping patience of 15 for optimal convergence
    trained_model, train_loss, val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        base_lr=0.005,
        patience=15,
        device=device
    )
    
    # Save best model state dict
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Best model weights saved to: {model_save_path}")
    
    # 6. Step 5: Model Evaluation & Reporting
    print("\n--- STEP 5: Model Evaluation & Metric Calculation ---")
    load_metrics_and_plot(project_root, train_loss, val_loss)
    metrics = evaluate_model_performance(project_root)
    plot_parameter_sensitivity(project_root)
    
    if metrics:
        print("\nFinal Test Metrics Summary:")
        print(f" - R² (Coefficient of Determination): {metrics['R2']:.4f}")
        print(f" - Mean Absolute Error (MAE):       {metrics['MAE']:.4f} %")
        print(f" - Root Mean Squared Error (RMSE):    {metrics['RMSE']:.4f} %")
        
    # 7. Step 6: In-Silico Decision-Support Demo
    print("\n--- STEP 6: Decision-Support Prediction Demo ---")
    # Let's test the model with typical reaction parameters (close to kinetics optimum)
    demo_temp = 180.0       # Optimum temperature
    demo_pressure = 30.0   # Optimum pressure
    demo_time = 6.0        # Adequate reaction time
    demo_conc = 2.2        # Optimum concentration
    demo_cat = 1.0         # Good catalyst amount
    
    pred_yield = predict_yield(demo_temp, demo_pressure, demo_time, demo_conc, demo_cat, project_root)
    
    print("\nSample Decision Query:")
    print(f"Inputs: Temp={demo_temp}°C, Press={demo_pressure}bar, Time={demo_time}h, Conc={demo_conc}M, Catalyst={demo_cat}wt%")
    print(f"Predicted Yield: {pred_yield:.2f} %")
    
    # Evaluate predicted yield class
    if pred_yield >= 85.0:
        decision = "HIGHLY RECOMMENDED (Passes QC threshold)"
    elif pred_yield >= 70.0:
        decision = "ACCEPTABLE (Consider minor adjustments to maximize conversion)"
    else:
        decision = "REJECTED (High QC failure risk)"
    print(f"System Decision Recommendation: {decision}")
    
    print("\n" + "="*65)
    print("  ML PIPELINE RUN COMPLETED SUCCESSFULLY  ")
    print("="*65)

if __name__ == '__main__':
    run_pipeline()
