import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from diamond.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from diamond.logger import logging
from diamond.exception.exception import DiamondException

class ModelEvaluation:
    """
    Class to handle model evaluation using MLflow for tracking and logging model performance metrics.

    Methods:
    --------
    eval_metrics(actual, pred):
        Computes evaluation metrics such as RMSE, MAE, and R-squared.
    
    initiate_model_evaluation(train_array, test_array, models):
        Evaluates the model, logs metrics to MLflow, and registers the model if needed.
    """

    def __init__(self):
        """
        Initializes the ModelEvaluation class and logs the initialization process.
        """
        logging.info("Model Evaluation Initialized")

    def eval_metrics(self, actual, pred):
        """
        Computes and logs the evaluation metrics: RMSE, MAE, and R-squared.

        Parameters:
        -----------
        actual : numpy array
            Actual values (ground truth) from the test dataset.
        pred : numpy array
            Predicted values from the model.

        Returns:
        --------
        rmse : float
            Root Mean Squared Error.
        mae : float
            Mean Absolute Error.
        r2 : float
            R-squared value, indicating the proportion of variance explained by the model.
        """
        logging.info("Calculating evaluation metrics: RMSE, MAE, R2")
        rmse = np.sqrt(mean_squared_error(actual, pred))  # Calculate RMSE
        mae = mean_absolute_error(actual, pred)  # Calculate MAE
        r2 = r2_score(actual, pred)  # Calculate R-squared

        logging.info(f"Evaluation Metrics - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        """
        Evaluates the model on test data, logs the evaluation metrics to MLflow, and registers the model if required.

        Parameters:
        -----------
        train_array : numpy array
            Array containing training data, which can be used to split into features and target if needed.
        test_array : numpy array
            Array containing test data, with features and target values for evaluation.
        models : dict
            Dictionary of model names and corresponding models to be evaluated.

        Raises:
        -------
        DiamondException:
            Custom exception to handle errors during model evaluation and logging.
        """
        try:
            logging.info('Started model evaluation process.')

            # Splitting test data into independent (X_test) and dependent variables (y_test)
            logging.info('Splitting test data into features (X_test) and labels (y_test)')
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Loading the model from artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            logging.info(f"Loading model from {model_path}")
            model = load_object(model_path)
            logging.info("Model loaded successfully")

            # Set MLflow registry URI for logging the results
            mlflow.set_registry_uri("")
            logging.info("MLflow registry URI set")

            # Determine the tracking URI type (file, http, etc.)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"Tracking URL type store detected: {tracking_url_type_store}")

            # Start an MLflow run to log the model evaluation process
            with mlflow.start_run():
                logging.info("MLflow run started")

                # Make predictions using the test data
                logging.info("Generating predictions on test data")
                predictions = model.predict(X_test)

                # Evaluate and log the model performance metrics (RMSE, MAE, R2)
                logging.info("Calculating and logging evaluation metrics")
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                logging.info(f"Metrics logged to MLflow - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

                # Model registration if tracking store is not a local file system
                if tracking_url_type_store != "file":
                    logging.info("Registering the model in MLflow Model Registry")
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                    logging.info("Model registered as 'ml_model' in MLflow Model Registry")
                else:
                    logging.info("Logging model as an artifact (file-based tracking store)")
                    # mlflow.sklearn.log_model(model, "model")
                    logging.info("Model logged successfully in MLflow (file-based storage)")

        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise DiamondException(e, sys)

