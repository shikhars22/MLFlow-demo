# 🍷 Wine Quality Experiment Tracking with MLFlow

This project demonstrates a complete MLOps workflow for tracking machine learning experiments using **MLFlow** and **DagsHub**. It uses an ElasticNet model to predict wine quality based on chemical properties.

## 🚀 Key Features

*   **Remote Experiment Tracking**: All training runs are automatically logged to a remote DagsHub server.
*   **Model Registry**: Automatically versions and registers models in the cloud (v1, v2, etc.).
*   **Automated Grid Search**: Includes a `main.py` script to test multiple hyperparameters automatically.
*   **Secure Serialization**: Uses the `skops` format for safe model persistence.

## 🛠️ Project Structure

```bash
├── demo.py             # Main training script with MLFlow logic
├── main.py             # Automation script for Grid Search
├── explanation_of_metrics.md  # Detailed guide on RMSE, MAE, R2, Alpha, and L1
└── requirements.txt    # Project dependencies
```

## 📦 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/shikhars22/MLFlow-demo.git
    cd MLFlow-demo
    ```

2.  **Create and activate environment**:
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate  # On Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run

### 1. Single Run
Pass custom `alpha` and `l1_ratio` values:
```bash
python demo.py 0.2 0.3
```

### 2. Automated Grid Search
Run 25 experiments automatically across a range of parameters:
```bash
python main.py
```

## 📊 Visualization

Visit your **DagsHub Dashboard** under the **Experiments** tab to see:
*   Real-time Accuracy vs. Error graphs.
*   Side-by-side comparisons of different Hyperparameters.
*   The Model Registry for versioning your trained artifacts.

---
*Developed for MLOps Learning Journey.* 🦅
