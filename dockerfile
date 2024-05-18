FROM python:3.8-slim

# Маппинг томов
ENV AIRFLOW_HOME=/opt/airflow

# Установка зависимостей 
RUN pip install apache-airflow==2.0.2 \
                apache-airflow-providers-postgres \
                apache-airflow-providers-redis \
                pandas numpy tensorflow scikit-learn mlflow yandexcloud

# Копия дага
COPY airflow_dag.py $AIRFLOW_HOME/dags/

# Точка входа
ENTRYPOINT ["airflow", "webserver"]