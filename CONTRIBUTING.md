## Начало работы 

### Предварительные требования
- Python 3.x
- Apache Airflow
- MLflow
- PostgreSQL 16
- Redis
- Docker 

### Установка 

1. Клонируйте репозиторий:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/Prediction-sepsis 

2. Установите необходимые пакеты 
    pip install -r requirements.txt

3. Настройте PostgreSQL и Redis: 
    Убедитесь, что PostgreSQL 16 установлен и работает.
    Убедитесь, что Redis установлен и работает. 

4. Запустите сервер MLflow 
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

### Запуск моделей 

1. Обучите и протестируйте модель LSTM:
    python MODELS/LSTM.py 

2. Обучите и протестируйте модель ANN 
    python MODELS/ANN.py 

3. Оркестрируйте метамодель 
    python MLflow/Meta-model.py 

4. Запланируйте рабочие процессы с помощью Airflow 
    airflow scheduler
    airflow webserver 

### Развертывание в Docker 

1. Постройте Docker-образ: 
    docker build -t sepsis-prediction .

2. Запустите контейнер
    docker run -d -p 8080:8080 sepsis-prediction 

### Вклад 
Вклады приветствуются! Если у вас есть предложения или исправления, отправьте pull request или откройте issue для обсуждения изменений. 

### Участники 
• Sh1nttaro (epev.nd@mail.ru)
• karyych (zakhar.karavaev.01@mail.ru)