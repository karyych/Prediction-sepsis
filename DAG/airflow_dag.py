import mlflow
import mlflow.keras
import mlflow.sklearn
import pandas as pd
import redis
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago


# ETL процесс
def extract_data_from_postgres(**kwargs):
    postgres_conn_id = kwargs["postgres_conn_id"]
    query = kwargs["query"]
    pg_hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    connection = pg_hook.get_conn()
    df = pd.read_sql(query, connection)
    return df


def extract_data_from_redis(**kwargs):
    redis_conn_id = kwargs["redis_conn_id"]
    key = kwargs["key"]
    r = redis.Redis.from_url(redis_conn_id)
    data = r.get(key)
    df = pd.read_json(data)
    return df


def preprocess_data(df):
    selected_features = [
        "HR",
        "O2Sat",
        "Temp",
        "MAP",
        "Resp",
        "Glucose",
        "Lactate",
        "WBC",
        "Hgb",
        "Platelets",
        "Age",
        "Gender",
        "SepsisLabel",
    ]
    data_selected = df[selected_features]
    data_array = data_selected.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_array)
    return scaled_data, scaler


def train_lstm_model(**kwargs):
    scaled_data = kwargs["scaled_data"]
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    history_size = 50
    x_train_lstm, y_train_lstm = prepare_data_for_lstm(X, history_size)
    model = build_lstm_model((x_train_lstm.shape[1], x_train_lstm.shape[2]))
    model.fit(x_train_lstm, y_train_lstm, epochs=10, batch_size=64)
    mlflow.keras.log_model(model, "lstm_model")


def train_random_forest_model(**kwargs):
    df = kwargs["df"]
    features = ["HR", "MAP", "O2Sat", "Temp"]
    target = "CompositeIndex"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "random_forest_model")


# DAG definition
with DAG(
    dag_id="etl_model_training", schedule_interval="@daily", start_date=days_ago(1)
) as dag:
    extract_postgres_task = PythonOperator(
        task_id="extract_postgres",
        python_callable=extract_data_from_postgres,
        op_kwargs={
            "postgres_conn_id": "your_postgres_conn",
            "query": "SELECT * FROM your_table",
        },
    )

    extract_redis_task = PythonOperator(
        task_id="extract_redis",
        python_callable=extract_data_from_redis,
        op_kwargs={"redis_conn_id": "your_redis_conn", "key": "your_key"},
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data", python_callable=preprocess_data, provide_context=True
    )

    train_lstm_task = PythonOperator(
        task_id="train_lstm", python_callable=train_lstm_model, provide_context=True
    )

    train_random_forest_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_random_forest_model,
        provide_context=True,
    )

    extract_postgres_task >> preprocess_task
    extract_redis_task >> preprocess_task
    preprocess_task >> [train_lstm_task, train_random_forest_task]
