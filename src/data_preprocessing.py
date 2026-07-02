import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class ChemicalProcessDataset(Dataset):
    """
    Custom PyTorch Dataset for Chemical Process Yield Prediction.
    Converts numpy arrays into PyTorch Tensors.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Unsqueeze to shape (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_and_split_data(csv_path, results_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Loads chemical process dataset, performs train/validation/test split, 
    normalizes features using StandardScaler, saves scaler, and returns PyTorch DataLoader objects.
    
    Parameters:
    - csv_path (str): Path to the source CSV file.
    - results_dir (str): Directory where scaler will be saved.
    - train_ratio (float): Fraction of data for training.
    - val_ratio (float): Fraction of data for validation.
    - test_ratio (float): Fraction of data for testing.
    - random_seed (int): Seed for reproducibility.
    
    Returns:
    - tuple: (train_loader, val_loader, test_loader, scaler)
    """
    # 1. Load dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Please run generate_data.py first.")
        
    df = pd.read_csv(csv_path)
    
    # 2. Extract input features (X) and target yield (y)
    feature_cols = ['Temperature_C', 'Pressure_bar', 'Reaction_Time_hrs', 'Concentration_M', 'Catalyst_wt']
    target_col = 'Yield_percent'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 3. Perform Train-Validation-Test Split
    # Split: Train vs Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=random_seed if isinstance(random_seed, int) else 42
    )
    
    # Split Temp into Validation and Test (50/50 of the remaining portion)
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, random_state=random_seed
    )
    
    print(f"Dataset Split Summary:")
    print(f" - Train samples:      {X_train.shape[0]} ({train_ratio * 100:.1f}%)")
    print(f" - Validation samples: {X_val.shape[0]} ({val_ratio * 100:.1f}%)")
    print(f" - Test samples:       {X_test.shape[0]} ({test_ratio * 100:.1f}%)")
    
    # 4. Standardize features
    # Standardizing makes training more stable and faster for deep learning models.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler so it can be re-loaded in evaluate.py and predict.py
    os.makedirs(results_dir, exist_ok=True)
    scaler_path = os.path.join(results_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Feature scaler saved successfully to: {scaler_path}")
    
    # 5. Create PyTorch Datasets
    train_dataset = ChemicalProcessDataset(X_train_scaled, y_train)
    val_dataset = ChemicalProcessDataset(X_val_scaled, y_val)
    test_dataset = ChemicalProcessDataset(X_test_scaled, y_test)
    
    # 6. Create PyTorch DataLoaders
    # Train loader uses shuffle=True to prevent the model from learning dependencies based on order
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Also save the raw test set for evaluation scripts
    test_data_summary = {
        'X_test_raw': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    test_summary_path = os.path.join(results_dir, 'test_data_summary.pkl')
    with open(test_summary_path, 'wb') as f:
        pickle.dump(test_data_summary, f)
        
    return train_loader, val_loader, test_loader, scaler

if __name__ == '__main__':
    # Test script in isolation
    project_root = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(project_root, 'data', 'chemical_process_data.csv')
    results_dir = os.path.join(project_root, 'results')
    
    # Check if data exists, if not generate it
    if not os.path.exists(csv_path):
        from generate_data import generate_chemical_dataset
        print("CSV not found, generating temporary test dataset...")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = generate_chemical_dataset()
        df.to_csv(csv_path, index=False)
        
    train_loader, val_loader, test_loader, scaler = preprocess_and_split_data(csv_path, results_dir)
    print("Preprocessing successfully validated.")
