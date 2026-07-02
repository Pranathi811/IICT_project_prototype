import torch
import torch.nn as nn

class ChemicalYieldPredictor(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Neural Network for Chemical Process Yield Regression.
    
    Architecture:
    - Input: 5 features (Temperature, Pressure, Time, Concentration, Catalyst)
    - Hidden Layer 1: 64 neurons + ReLU + Batch Normalization + Dropout (10%)
    - Hidden Layer 2: 32 neurons + ReLU + Batch Normalization + Dropout (10%)
    - Hidden Layer 3: 16 neurons + ReLU
    - Output Layer: 1 neuron (Yield %)
    """
    def __init__(self, input_dim=5, dropout_rate=0.1):
        super(ChemicalYieldPredictor, self).__init__()
        
        # Hidden Layer 1
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Hidden Layer 2
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Hidden Layer 3
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        
        # Output Layer (1 node for regression target)
        self.out = nn.Linear(16, 1)
        
        # Apply custom weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights using Kaiming Normal (He Normal) initialization,
        which is standard and effective for layers followed by ReLU activations.
        Biases are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
        - x (torch.Tensor): Input batch of shape (batch_size, 5)
        
        Returns:
        - torch.Tensor: Predicted yields of shape (batch_size, 1)
        """
        # Pass through FC1
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Pass through FC2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Pass through FC3
        out = self.fc3(out)
        out = self.relu3(out)
        
        # Output prediction
        out = self.out(out)
        return out

if __name__ == '__main__':
    # Verify model with a dummy batch
    model = ChemicalYieldPredictor()
    dummy_input = torch.randn(8, 5) # Batch of 8 samples, 5 features each
    dummy_output = model(dummy_input)
    
    print("PyTorch MLP Model summary:")
    print(model)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    print("Model validation successful.")
