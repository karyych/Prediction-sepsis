import numpy as np
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from tensorflow.keras.models import load_model

import mlflow
import mlflow.sklearn


@dag(schedule_interval="@daily", start_date=days_ago(1))
def meta_model_workflow():
    @task
    def load_models():
        lstm_model = load_model("path_to_lstm_model")
        rf_model = mlflow.sklearn.load_model("path_to_random_forest_model")
        return lstm_model, rf_model

    @task
    def combine_predictions(lstm_model, rf_model, X_test):
        lstm_pred = lstm_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        combined_pred = (lstm_pred + rf_pred) / 2
        return combined_pred

    lstm_model, rf_model = load_models()
    X_test = np.array([])
    combined_pred = combine_predictions(lstm_model, rf_model, X_test)


meta_model_workflow()
