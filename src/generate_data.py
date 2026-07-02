import os
import numpy as np
import pandas as pd

def generate_chemical_dataset(num_samples=2000, random_seed=42):
    """
    Generates a synthetic chemical process dataset based on physical-chemical kinetics principles.
    
    Parameters:
    - num_samples (int): Number of experimental data points to generate.
    - random_seed (int): Seed for reproducibility.
    
    Returns:
    - pd.DataFrame: Generated dataset containing process parameters and corresponding yield.
    """
    np.random.seed(random_seed)
    
    # 1. Generate uniform random distributions for the process parameters (inputs)
    # Temperature: 100°C to 250°C (influences reaction rate via Arrhenius but causes degradation at high T)
    temperature = np.random.uniform(100.0, 250.0, num_samples)
    
    # Pressure: 1.0 to 50.0 bar (affects equilibrium and gas-phase concentrations)
    pressure = np.random.uniform(1.0, 50.0, num_samples)
    
    # Reaction Time: 0.5 to 10.0 hours (yield increases with time but plateaus or drops due to decomposition)
    reaction_time = np.random.uniform(0.5, 10.0, num_samples)
    
    # Reactant Concentration: 0.5 to 3.0 M (molarity of the primary reactant)
    concentration = np.random.uniform(0.5, 3.0, num_samples)
    
    # Catalyst Amount: 0.1 to 2.0 wt% (accelerates reaction, exhibits saturation behavior)
    catalyst_amount = np.random.uniform(0.1, 2.0, num_samples)
    
    # 2. Define kinetics-inspired non-linear response functions
    
    # Temperature effect: Gaussian peak around 180°C (optimal temperature)
    T_opt = 180.0
    sigma_T = 35.0
    f_T = np.exp(-((temperature - T_opt) ** 2) / (2 * (sigma_T ** 2)))
    
    # Pressure effect: Quadratic peak around 30.0 bar
    P_opt = 30.0
    f_P = 1.0 - 0.15 * ((pressure - P_opt) / 20.0) ** 2
    f_P = np.clip(f_P, 0.2, 1.0) # Ensure no negative contributions
    
    # Reaction Time effect: Rapid initial rise, then plateau and slight decay
    # Modelled as: (1 - e^(-0.6 * t)) * e^(-0.015 * t)
    f_t = (1.0 - np.exp(-0.6 * reaction_time)) * np.exp(-0.015 * reaction_time)
    f_t = f_t / np.max(f_t) # Normalize to max 1.0
    
    # Concentration effect: Quadratic peak around 2.2 M
    C_opt = 2.2
    f_C = 1.0 - 0.25 * ((concentration - C_opt) ** 2)
    f_C = np.clip(f_C, 0.4, 1.0)
    
    # Catalyst effect: Saturation curve (hyperbolic tangent)
    f_Cat = np.tanh(1.8 * catalyst_amount)
    
    # 3. Multi-variable interaction terms
    # Synergy between Temperature and Catalyst (catalyst is more active at higher temperature,
    # but also deactivates if temp is extremely high)
    f_T_Cat_interaction = 0.08 * (temperature / 180.0) * f_Cat * (1.0 - np.abs(temperature - 180.0) / 150.0)
    
    # 4. Integrate components to compute Yield (%)
    # Base yield is around 40%. The maximum theoretical yield is 100%.
    # We combine the components with weights that sum to 1.0 (excluding interactions and noise).
    w1, w2, w3, w4, w5 = 0.35, 0.15, 0.20, 0.15, 0.15
    
    yield_clean = 40.0 + 50.0 * (
        w1 * f_T +
        w2 * f_P +
        w3 * f_t +
        w4 * f_C +
        w5 * f_Cat +
        f_T_Cat_interaction
    )
    
    # Add random experimental/measurement noise (Gaussian, mean=0, std=1.2)
    noise = np.random.normal(0.0, 1.2, num_samples)
    process_yield = yield_clean + noise
    
    # Clip yield between 0% and 100% to reflect physical limits
    process_yield = np.clip(process_yield, 0.0, 100.0)
    
    # 5. Create DataFrame
    df = pd.DataFrame({
        'Temperature_C': np.round(temperature, 1),
        'Pressure_bar': np.round(pressure, 1),
        'Reaction_Time_hrs': np.round(reaction_time, 2),
        'Concentration_M': np.round(concentration, 2),
        'Catalyst_wt': np.round(catalyst_amount, 2),
        'Yield_percent': np.round(process_yield, 2)
    })
    
    return df

if __name__ == '__main__':
    # Define paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'chemical_process_data.csv')
    
    print(f"Generating chemical process dataset with 2,000 samples...")
    dataset = generate_chemical_dataset(num_samples=2000)
    
    dataset.to_csv(csv_path, index=False)
    print(f"Dataset successfully created and saved to: {csv_path}")
    print("\nSummary Statistics of Generated Dataset:")
    print(dataset.describe())
    print("\nFirst 5 rows of dataset:")
    print(dataset.head())
