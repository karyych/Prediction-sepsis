# Импорт библиотек
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# Загрузка данных
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Предпроцессинг данных
def preprocess_data(df, selected_features):
    data_selected = df[selected_features]
    data_array = data_selected.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_array)
    return scaled_data, scaler


# Объеденение данных
def split_features_target(scaled_data):
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    return X, y


# Преобразование данных
def prepare_data_for_lstm(data, history_size):
    features, labels = [], []
    for i in range(len(data) - history_size):
        features.append(data[i : i + history_size, :])
        labels.append(data[i + history_size, -1])
    return np.array(features), np.array(labels)


# Сборка модели
def build_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# Обучение модели
def train_lstm_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )
    return history


# Оценка модели
def evaluate_model(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return test_loss, mae, mse, r2, y_pred


# ROC кривая
def plot_roc_curve(y_test, y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve - LSTM")
    plt.legend()
    plt.show()


# Контроль предикта для прогнозирвоания раннего сепсиса
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.plot(y_test, label="True Values")
    plt.plot(y_pred, label="Predicted Values")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Model Prediction vs True Values - LSTM")
    plt.legend()
    plt.show()


# Создание композитного индекса
def create_composite_index(df, selected_columns):
    composite_index = df[selected_columns].mean(axis=1)
    df["CompositeIndex"] = composite_index
    return df


# Линейная модель
def train_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, mae, r2, y_pred


# MAIN функция
def main(file_path):
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

    df = load_data(file_path)
    scaled_data, scaler = preprocess_data(df, selected_features)
    X, y = split_features_target(scaled_data)

    history_size = 50
    x_train_lstm, y_train_lstm = prepare_data_for_lstm(X, history_size)
    x_test_lstm, y_test_lstm = prepare_data_for_lstm(X, history_size)

    lstm_model = build_lstm_model((x_train_lstm.shape[1], x_train_lstm.shape[2]))
    train_lstm_model(lstm_model, x_train_lstm, y_train_lstm, x_test_lstm, y_test_lstm)
    test_loss, mae, mse, r2, y_pred = evaluate_model(
        lstm_model, x_test_lstm, y_test_lstm
    )

    print(f"Test Loss LSTM: {test_loss}")
    print(f"MAE LSTM: {mae}")
    print(f"MSE LSTM: {mse}")
    print(f"R2 Score LSTM: {r2}")

    plot_roc_curve(y_test_lstm, y_pred)
    plot_predictions(y_test_lstm, y_pred)

    selected_columns = ["HR", "MAP", "O2Sat", "Temp"]
    df = create_composite_index(df, selected_columns)

    features = ["HR", "MAP", "O2Sat", "Temp"]
    target = "CompositeIndex"
    X = df[features]
    y = df[target]

    lr_model, lr_mse, lr_mae, lr_r2, lr_y_pred = train_linear_regression(X, y)

    print(f"Mean Squared Error: {lr_mse}")
    print(f"Mean Absolute Error: {lr_mae}")
    print(f"R^2 Score: {lr_r2}")


# Path
if __name__ == "__main__":
    file_path = ""
    main(file_path)
