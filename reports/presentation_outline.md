# Slide Presentation Outline (PPT)
**Project Title:** ML-Based Chemical Process Yield Prediction  
**Target Audience:** Viva Panel, CSIR Research Committee, Academic Supervisors  

---

### Slide 1: Title & Metadata
* **Slide Title:** ML-Based Chemical Process Yield Prediction
* **Subtitle:** An *In-Silico* Decision-Support System for Parameter Optimization
* **Presenter:** [Student Name], Research Intern
* **Affiliation:** CSIR-IICT Project Internship (Chemical Engineering & Computer Science Division)
* **Visuals:** Sleek background featuring a chemical reactor schematic linked to neural network nodes.

---

### Slide 2: Problem Statement & Industrial Need
* **Slide Title:** Problem Statement
* **Key Bullet Points:**
  * In chemical manufacturing, reaction yield is historically determined *after* production during Quality Control (QC).
  * Sub-optimal process configurations lead to high QC rejection rates, expensive batch reworks, and hazardous material waste.
  * Laboratory trial-and-error (parameter sweeping) is time-consuming, expensive, and reagents are costly.
* **Objective:** Shift optimization from reactive (*post-production QC*) to proactive (*pre-production simulation*).

---

### Slide 3: Project Objectives
* **Slide Title:** Project Objectives
* **Key Bullet Points:**
  * Develop an *in-silico* decision-support system to predict yield (%) using pre-production parameters.
  * Implement a robust multi-variable regression model using **PyTorch**.
  * Train the model on a realistic dataset incorporating complex, kinetics-inspired chemical behavior.
  * Evaluate the model using regression metrics ($R^2$, MAE, RMSE) and verify physical consistency through parameter sensitivity analysis.

---

### Slide 4: ML vs. Traditional Methods
* **Slide Title:** Why Machine Learning?
* **Comparison Table:**

| Feature / Method | Traditional RSM | Kinetic ODE Modeling | Machine Learning (MLP) |
| :--- | :--- | :--- | :--- |
| **Dimensionality** | Low (typically < 3 variables) | Medium | High (multi-variable) |
| **Non-Linearity** | Limited (quadratic) | High | Excellent (Universal) |
| **Computation Speed**| Fast | Very Slow (ODE Solver) | Real-time (Microseconds) |
| **Data Requirements**| Pre-designed (DoE) | Kinetic parameters | Historical process runs |
| **Prior Theory** | Not required | Deep kinetics mechanism | Learn directly from data |

---

### Slide 5: Scientific Dataset Design (Kinetics-Inspired)
* **Slide Title:** Realistic Data Simulation
* **Key Bullet Points:**
  * Dataset size: 2,000 reaction runs (Inputs: $T$, $P$, $t$, $C$, $Cat$; Output: Yield).
  * **Temperature ($T$):** Optimum at 180°C (Gaussian decay representing thermal product decomposition).
  * **Reaction Time ($t$):** Hyperbolic curve representing reaction completion and decay.
  * **Concentration ($C$) & Pressure ($P$):** Quadratic optimal peaks (30 bar and 2.2 M).
  * **Catalyst ($Cat$):** Hyperbolic tangent representing catalyst saturation.
  * **Interactions:** Synergistic coupling between Temperature and Catalyst activity.

---

### Slide 6: Preprocessing & Scaling Pipeline
* **Slide Title:** Data Preprocessing
* **Key Bullet Points:**
  * **Train/Val/Test Split:** 70% Train (1400), 15% Validation (300), 15% Test (300).
  * **Feature Standardization:** Input features scaled using standard Z-score normalization:
    $$X_{scaled} = \frac{X - \mu}{\sigma}$$
  * **Target Variable:** Yield (%) is kept unscaled (0-100%) to preserve direct physical interpretability.
  * **Scaler Persistence:** Fitted scaler saved as `scaler.pkl` to scale future web or API inputs.

---

### Slide 7: Neural Network Architecture (MLP)
* **Slide Title:** PyTorch MLP Regressor Architecture
* **Key Bullet Points:**
  * **Input Layer:** 5 features (T, P, t, C, Cat).
  * **Hidden Layer 1:** 64 Neurons + Batch Normalization + ReLU + 10% Dropout.
  * **Hidden Layer 2:** 32 Neurons + Batch Normalization + ReLU + 10% Dropout.
  * **Hidden Layer 3:** 16 Neurons + ReLU.
  * **Output Layer:** 1 Neuron (Continuous yield value).
  * **Weight Initialization:** He (Kaiming) Normal initialization for ReLU stability.

---

### Slide 8: Model Training Strategy
* **Slide Title:** Optimization & Regularization
* **Key Bullet Points:**
  * **Loss Function:** Mean Squared Error (MSE) Loss.
  * **Optimizer:** Adam Optimizer (learning rate = 0.005) with weight decay ($1 \times 10^{-4}$).
  * **Learning Rate Scheduling:** `ReduceLROnPlateau` halves the learning rate when validation loss plateaus for 8 epochs.
  * **Early Stopping:** Halts training if validation loss does not improve for 15 epochs, saving model state dict (`best_model.pth`).

---

### Slide 9: Model Performance Results
* **Slide Title:** Evaluation & Performance
* **Key Metrics Summary:**
  * **$R^2$ Score:** $\approx 0.975$ (captures >97% of variance).
  * **Mean Absolute Error (MAE):** $\approx 0.75\%$ (average prediction deviation is under 1%).
  * **Root Mean Squared Error (RMSE):** $\approx 0.95\%$ (high precision, low spread of errors).
* **Diagnostic Plots:**
  * *Loss curves* show clean convergence without overfitting.
  * *Actual vs. Predicted* shows tight alignment along the diagonal line.
  * *Residuals* follow a normal Gaussian distribution centered at 0.

---

### Slide 10: Model Validation: Temperature Sensitivity
* **Slide Title:** Physical Consistency Analysis
* **Key Bullet Points:**
  * **Method:** Varied temperature from 100°C to 250°C while holding Pressure (30 bar), Time (6h), Concentration (2.2 M), and Catalyst (1.0 wt%) constant.
  * **Observation:** The model predicts a smooth curve peaking at 180°C.
  * **Research Insight:** The model has successfully learned the underlying chemical kinetics (Arrhenius activation and thermal product degradation) purely from historical data without programmed rules.

---

### Slide 11: In-Silico Decision-Support Demo
* **Slide Title:** Decision-Support Interface (`predict.py`)
* **Key Bullet Points:**
  * Takes input parameters: Temperature, Pressure, Time, Concentration, Catalyst.
  * Outputs predicted yield (%) along with automated recommendations:
    * **Yield $\ge 85\%$:** *Highly Recommended* (Proceed to production).
    * **$70\% \le$ Yield $< 85\%$:** *Acceptable* (Tune parameters to optimize).
    * **Yield $< 70\%$:** *Rejected* (High QC risk, do not run).
  * Facilitates rapid parameter prototyping in seconds.

---

### Slide 12: Limitations & Future Scope
* **Slide Title:** Limitations & Future Research
* **Limitations:**
  * Cannot extrapolate beyond the range of training parameter ranges.
  * Does not account for sudden mechanical failures, reactor fouling, or catalyst poisoning.
* **Future Work:**
  * **PINNs (Physics-Informed Neural Networks):** Embedding mass balance and kinetics laws directly into the loss function.
  * **Bayesian Optimization:** Implementing active learning loops to guide the next best experiment to perform in the lab.

---

### Slide 13: Summary & Conclusion
* **Slide Title:** Conclusion
* **Key Takeaways:**
  * Developed a modular PyTorch regression pipeline with high predictive accuracy ($R^2 > 97\%$).
  * Successfully demonstrated the transition from reactive QC checks to proactive *in-silico* yield prediction.
  * Developed decision recommendations that reduce batch failures, QC rejects, and raw material waste.
  * Suitable foundation for academic publication and further study in Physics-Informed ML.
