# --- IMPORTS ---
# Standard Python utility for interacting with the operating system (e.g., environment variables)
import os
# Used to suppress non-critical warnings that might clutter the terminal output
import warnings
# Provides access to variables used or maintained by the interpreter (like command-line arguments)
import sys

# High-performance data manipulation library used for handling the CSV dataset
import pandas as pd
# Fundamental package for scientific computing with Python, used here for math and arrays
import numpy as np
# Import specific evaluation metrics to measure model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Utility to split the dataset into two parts: one for training and one for testing
from sklearn.model_selection import train_test_split
# The actual Machine Learning model: Linear regression with combined L1 and L2 priors
from sklearn.linear_model import ElasticNet
# Used to break down URLs into components (scheme, netloc, etc.) for tracking URIs
from urllib.parse import urlparse 
# The core library for tracking experiments, packaging code, and deploying models
import mlflow
# Generates a 'schema' for the model input/output so others know how to use it
from mlflow.models.signature import infer_signature
# Specialized MLFlow module that understands how to save and load Scikit-Learn models
import mlflow.sklearn
# Connects your local MLFlow runs to the DagsHub remote community platform
import dagshub
# Standard Python logging library to output status messages or errors
import logging

# --- INITIALIZATION ---
# Links this script to your DagsHub repository to store experiment results in the cloud
dagshub.init(repo_owner='shikhars22', repo_name='MLFlow-demo', mlflow=True)

# Configures the logging level to WARNING to avoid seeing too many debug messages
logging.basicConfig(level=logging.WARN)
# Creates a logger instance for this specific file
logger = logging.getLogger(__name__)

# --- UTILITY FUNCTIONS ---
# Defines a helper function to calculate the three main error metrics for our model
def eval_metrics(actual, pred):
    # Root Mean Squared Error: Measures the average magnitude of the error
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # Mean Absolute Error: Measures the average absolute difference between actual and predicted
    mae = mean_absolute_error(actual, pred)
    # R-squared Score: Indicates how well the model fits the data (1.0 is perfect)
    r2 = r2_score(actual, pred)
    # Return all three scores as a tuple
    return rmse, mae, r2

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Prevent the terminal from filling up with library-specific warnings
    warnings.filterwarnings("ignore")
    # Set a fixed seed for random number generation to ensure results are reproducible
    np.random.seed(40)

    # Define the remote URL for the classic Red Wine Quality dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    
    # Wrap the download in a try-except block to handle internet or server failures
    try:
        # Read the semicolon-separated CSV file directly from the URL into a DataFrame
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        # Log a descriptive error message if the download fails
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data: 75% for training the model, 25% for checking its performance
    train, test = train_test_split(data)

    # Remove the target column 'quality' from the features (X) for training
    train_x = train.drop(["quality"], axis=1)
    # Do the same for the test features
    test_x = test.drop(["quality"], axis=1)
    # Isolate the 'quality' column as the target labels (y) we want to predict
    train_y = train[["quality"]]
    # Do the same for the test labels
    test_y = test[["quality"]]

    # Read the 'alpha' hyperparameter from the first command-line argument (default to 0.5)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # Read the 'l1_ratio' hyperparameter from the second command-line argument (default to 0.5)
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # --- MLFLOW TRACKING SECTION ---
    # Start a new MLFlow 'run' - this is the container for all tracking data
    with mlflow.start_run():
        # Instantiate the ElasticNet model with the provided hyperparameters
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        # Train the model using the training features and labels
        lr.fit(train_x, train_y)

        # Use the trained model to predict quality scores for the unseen test data
        predicted_qualities = lr.predict(test_x)

        # Calculate performance metrics by comparing predictions to the actual test labels
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print the results to the terminal for immediate feedback
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # LOGGING PARAMS: Record the 'Alpha' input value so we can compare different runs
        mlflow.log_param("alpha", alpha)
        # LOGGING PARAMS: Record the 'L1 Ratio' input value
        mlflow.log_param("l1_ratio", l1_ratio)
        
        # LOGGING METRICS: Record the resulting RMSE score to track model accuracy
        mlflow.log_metric("rmse", rmse)
        # LOGGING METRICS: Record the R2 score
        mlflow.log_metric("r2", r2)
        # LOGGING METRICS: Record the MAE score
        mlflow.log_metric("mae", mae)

        # --- MODEL REGISTRATION ---
        # Since we are using dagshub.init(mlflow=True), we are always on a remote server
        # We can directly log and register the model using the 'skops' format
        mlflow.sklearn.log_model(
            sk_model=lr, 
            name="model", 
            registered_model_name="ElasticnetWineModel",
            serialization_format="skops"
        )
