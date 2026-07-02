# ML-Based Chemical Process Yield Prediction
**An In-Silico Decision-Support System for Chemical Engineers**

This repository contains a modular PyTorch machine learning pipeline designed to predict chemical reaction yield (%) using pre-production process parameters: **Temperature**, **Pressure**, **Reaction Time**, **Reactant Concentration**, and **Catalyst Amount**.

The model is designed as an academic decision-support tool. It enables process engineers to simulate parameter combinations *in silico* and check predicted yield before actual laboratory runs, thereby reducing QC (Quality Control) reworks and reagent waste.

---

## Directory Structure

```
IICT_project/
│
├── data/
│   └── chemical_process_data.csv       # Generated dataset (2,000 samples)
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py                # Kinetics-inspired dataset generator
│   ├── data_preprocessing.py            # Train/val/test splits & standard scaling
│   ├── model.py                        # PyTorch Multi-Layer Perceptron (MLP)
│   ├── train.py                        # Training loop with Early Stopping
│   ├── evaluate.py                     # Performance evaluation & regression plots
│   └── predict.py                      # Interactive query command-line interface
│
├── results/                            # Saved models, metrics, and visualization plots
│   ├── best_model.pth                  # Trained PyTorch weights
│   ├── scaler.pkl                      # Fitted standard scaler
│   ├── metrics_report.txt              # MAE, RMSE, and R2 results
│   ├── loss_history.png                # Training vs Validation loss curve
│   ├── actual_vs_predicted.png         # Regression diagonal fit plot
│   ├── residual_distribution.png       # Error normality histogram
│   └── temperature_sensitivity.png     # Validation of kinetics optimum learning
│
├── reports/
│   ├── internship_report.md            # CSIR-level academic report (Markdown)
│   └── presentation_outline.md         # PowerPoint slide-by-slide structure
│
├── main.py                             # Master pipeline execution script
├── requirements.txt                    # Python library requirements
└── README.md                           # This documentation file
```

---

## Installation & Setup

1. **Clone or navigate** to the project directory:
   ```bash
   cd c:/Users/pulag/Desktop/IICT_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Requires PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn)*

---

## Running the Pipeline

You can run the entire pipeline end-to-end (data generation -> preprocessing -> training -> evaluation -> prediction demo) with a single command:

```bash
python main.py
```

### Steps Executed by `main.py`:
1. **Generates Synthetic Data:** Runs a kinetics-inspired model simulating 2,000 experiment data points.
2. **Preprocesses Data:** Splits datasets (70% Train, 15% Val, 15% Test) and applies standard Z-score normalization to inputs.
3. **Trains PyTorch Model:** Initializes the MLP and trains it using MSE loss, Adam optimizer, learning rate scaling, and early stopping.
4. **Evaluates & Generates Metrics:** Computes $R^2$, MAE, and RMSE on the held-out test set and saves plots to `results/`.
5. **Decision-Support Demo:** Runs a sample yield prediction on optimal parameters and outputs process recommendations.

---

## Interactive Decision-Support Query (`predict.py`)

Engineers can query the trained model with arbitrary parameters prior to production to predict expected yields:

```bash
python src/predict.py --temp 180.0 --pressure 30.0 --time 6.0 --conc 2.2 --cat 1.0
```

### Output Recommendations:
* **Yield $\ge 85.0\%$**: **HIGHLY RECOMMENDED**. Safe to proceed.
* **$70.0\% \le$ Yield $< 85.0\%$**: **ACCEPTABLE (Marginal)**. Consider adjusting temperature (near 180°C) or pressure (near 30 bar) to optimize yield.
* **Yield $< 70.0\%$**: **REJECTED**. High risk of reaction failure or high waste fraction. Do not run.

---

## Model Evaluation Metrics Interpretation

* **$R^2$ Score (Coefficient of Determination):** Target is $\ge 0.95$. Indicates the percentage of yield variation explained by the input parameters.
* **Mean Absolute Error (MAE):** Target is $< 1.0\%$. Represents the average deviation of predictions from actual physical yield.
* **Residual Analysis:** Residual error distribution is validated to ensure it follows a normal Gaussian curve centered at zero, showing that the model has captured all structured trends.
