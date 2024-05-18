FROM python:3.8-slim

# Маппинг томов
ENV AIRFLOW_HOME=/opt/airflow

# Установка зависимостей 
RUN apt-get update && apt-get install -y \ 
    build-essential \ 
    default-libmysqlclient-dev \ 
    libssl-dev \ 
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install apache-airflow==2.0.2 \
                apache-airflow-providers-postgres \
                apache-airflow-providers-redis \
                pandas numpy tensorflow scikit-learn mlflow \
                yandexcloud

# Копия дага
COPY airflow_dag.py $AIRFLOW_HOME/dags/

# Копирование всех скриптов проекта 
COPY . ${AIRFLOW_HOME}/ 

# Инициализация базы данных Airflow и создание пользователя
RUN airflow db init && \
    airflow users create -u admin -p admin -r Admin -e admin@example.com -f Admin -l User

# Точка входа
ENTRYPOINT ["airflow", "sheduler", "&", "airflow", "webserver"]