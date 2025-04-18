# Metro System Failure Prediction
A machine learning-based solution to predict potential failures in metro trains using the MetroPT dataset. This project uses sensor data collected from metro air production units (APUs) to anticipate component breakdowns and reduce unexpected disruptions in train operations.[Dataset](https://github.com/user-attachments/files/19814673/s41597-022-01877-3.pdf)





##  Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributing](#contributing)

---

##  About the Project

The "Metro System Failure Prediction" project utilizes machine learning to predict and detect critical failures in the Air Production Unit (APU) of metro trains. This allows maintenance teams to intervene before a failure causes service disruption, enhancing safety and efficiency in metro operations.

## ❗ Problem Statement

Failures in metro systems can cause significant inconvenience and cost. The APU, a key component in air management (brakes, suspension), can experience air leaks or oil leaks that often go undetected. The Goal is to Use machine learning models to predict failures at least 2 hours in advance to allow trains to be removed from operation safely.

##  Tech Stack

- **Language**: Python 3.x
- **ML Libraries**: scikit-learn(V-1.6.0), XGBoost, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Optimization**: Optuna, GridSearchCV
- **Platform**: Jupyter Notebook / PyCharm
- **Dataset**: MetroPT dataset (from Porto Metro, Portugal)

##  Features

- Predicts three classes:
  - 0: Optimal Condition
  - 1: Two Hours Before Failure
  - 2: Actual Failure
- Supports dynamic threshold tuning using precision-recall tradeoffs.
- Handles class imbalance using NearMiss undersampling.
- Hyperparameter tuning using Optuna for Logistic Regression, Random Forest, and XGBoost.
- Precision-recall curve visualization for model confidence.

## ⚙️ How It Works

The project is structured into four main stages, each represented by a separate Jupyter Notebook:

**1. Data Cleaning (MetroPT_datacleaning.ipynb)**
- Removed unnecessary features such as GPS coordinates and the pressure switch column.
- Ensured the timestamp column was correctly formatted to datetime.
- Validated feature types, and checked for anomalies or inconsistencies.

**2. Preprocessing (MetroPT_data_preprocessing.ipynb)**
- Converted data types for memory efficiency: float64 → float32, int64 → int8.
- Handled missing or noisy values in sensor readings.
- Created and saved a clean, processed dataset in .feather format for fast loading in later stages.
- Prepared the target label (Target) for classification based on failure annotations.

**3. Exploratory Data Analysis (MetroPT_EDA.ipynb)**
- Conducted deep visual analysis for each of the labeled failure periods.
- Plotted Correlation matrix.

- For each failure type:
    - Plotted time-series trends of key sensor readings such as:
      - TP2
      - TP3
      - H1
      - DV_pressure
      - Reservoirs
      - Oil_temperature
      - Flowmeter
      - Motor_current
      - gpsSpeed  

   - These visualizations helped distinguish how each sensor behaves in the lead-up to and during a failure, identifying clear anomalies.
  
- Performed binary variable distribution analysis:
  - Compared activation patterns of on/off signals like COMP, DV_electric, LPS, and Oil_Level across classes.
  - Observed distinct distribution shifts in binary features during failure periods, helping inform their relevance in classification.
- Insights from this EDA were crucial in:
  - Understanding failure mechanisms.
  - Informing feature importance.
- Validating the time-based labeling of failures.

**4. Model Training and Tuning**
- Initial Modeling (``)
  - Implemented and evaluated baseline versions of Logistic Regression, Random Forest, and XGBoost classifiers.
  - Models were trained using a stratified train-test split (70%-30%) after class imbalance correction using NearMiss undersampling.
  - Applied StandardScaler to normalize feature inputs.
  - Performance was evaluated using classification report (precision, recall, F1-score) and normalized confusion matrix.
- Hyperparameter Tuning with Optuna (``)
  - Developed a modular tuning framework using Optuna.
  - Created separate objective functions for each model (Logistic Regression, Random Forest, and XGBoost).
  - The Tuner class allowed specifying:
    - Custom search spaces (parameter ranges)
    - Number of trials
    - Pruning strategy
    - Evaluation metric (log loss)
  - Used StratifiedKFold cross-validation and predict_proba() for scoring to ensure robust optimization.
  - Tuned models significantly outperformed the baseline versions—especially in terms of recall and F1-score for class 1 (pre-failure) and class 2 (failure). Evaluation Metrics include precision, recall, F1-score, AUC. Thresholds optimized using PR curves.
  - In Modeling multiple models are trained and tuned:
   - SVM
   - Random Forest
   - Logistic Regression
   - XGBoost

##  Installation

```bash
git clone https://github.com/mdahsan11/Metro-System-Failure-Prediction.git
cd Metro-System-Failure-Prediction
pip install -r requirements.txt
```

##  Results

- **Best Model:** XGBoost (after hyperparameter tuning)
- **Important Evaluation Metrics:**

| Class                            | Precision | Recall | F1-Score |
| -------------------------------- | --------- | ------ | -------- |
| **0** (Optimal Condition)        | 0.82      | 0.83   | 0.82     |
| **1** (Two Hours Before Failure) | 0.82      | 0.92   | 0.86     |
| **2** (Actual Failure)           | 0.83      | 0.62   | 0.71     |

- **Macro Avg F1-Score:** `0.80`
- **Weighted Avg F1-Score:** `0.82`

While accuracy was \~82%, **the key focus was on precision, recall, and F1-score**, which are more critical for failure detection—especially in minimizing false negatives for class 2.

##  Future Scope

- Advanced **feature engineering** to improve early prediction.
- Integration with real-time IoT sensor feeds.
- Time-series models (e.g. LSTM) to better capture patterns.
- Deployment of a dashboard for visualization and alerting.

##  Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

**Contact:** Mohd Ahsan Ullah | [ahsanmohd564@gmail.com / LinkedIn: Mohd Ahsan Ullah] | GitHub: [@mdahsan11]

**Citation:** If you use the MetroPT dataset, please cite the original [Nature Scientific Data paper](https://doi.org/10.1038/s41597-022-01877-3).

