import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=200, base_lr=0.005, patience=15, device='cpu'):
    """
    Trains the PyTorch model using training and validation DataLoaders.
    Implements L2 regularization, ReduceLROnPlateau learning rate scheduler, and Early Stopping.
    
    Parameters:
    - model (nn.Module): The PyTorch model to train.
    - train_loader (DataLoader): DataLoader containing scaled training data.
    - val_loader (DataLoader): DataLoader containing scaled validation data.
    - epochs (int): Maximum number of training epochs.
    - base_lr (float): Initial learning rate for Adam optimizer.
    - patience (int): Number of epochs to wait for validation loss improvement before stopping.
    - device (str): 'cpu' or 'cuda' (GPU).
    
    Returns:
    - tuple: (best_model, train_loss_history, val_loss_history)
    """
    model = model.to(device)
    
    # 1. Define Loss Function and Optimizer
    # MSE is the standard loss function for continuous regression targets
    criterion = nn.MSELoss()
    
    # Adam optimizer with L2 weight decay to penalize large weights and prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    # Learning rate scheduler: decays learning rate if validation loss doesn't improve for 8 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5
    )
    
    # 2. Initialize histories and early stopping variables
    train_loss_history = []
    val_loss_history = []
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    print("Starting Model Training...")
    print(f"Device: {device} | Max Epochs: {epochs} | Initial Learning Rate: {base_lr}\n")
    
    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_train_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_x.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # Print progress every 10 epochs or at epoch 1
        if epoch == 1 or epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")
            
        # --- Early Stopping check ---
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} epochs.")
                print(f"Stopping training at Epoch {epoch}. Restoring best model weights (Val MSE: {best_loss:.4f}).")
                break
                
    # Restore the best model weights
    model.load_state_dict(best_model_weights)
    return model, train_loss_history, val_loss_history

if __name__ == '__main__':
    from model import ChemicalYieldPredictor
    # Import the function from data_preprocessing
    from data_preprocessing import preprocess_and_split_data 

    # 1. Define paths 
    csv_path = 'data/chemical_process_data.csv' # Ensure this file path is correct
    results_dir = 'results'

    # 2. Call the function and unpack the 4 returned objects
    train_ldr, val_ldr, test_ldr, scaler = preprocess_and_split_data(csv_path, results_dir)

    # 3. Initialize and train
    model = ChemicalYieldPredictor()
    print("Starting real training...")
    # Now 'train_ldr' and 'val_ldr' are defined, so this line will work!
    trained_model, train_hist, val_hist = train_model(model, train_ldr, val_ldr, epochs=50)
    
    # 4. Save the model
    import torch
    import os
    os.makedirs('results', exist_ok=True)
    torch.save(trained_model.state_dict(), 'results/model.pt')
    print("Model saved successfully to results/model.pt")