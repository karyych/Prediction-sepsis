import pandas as pd
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Извлечение данных
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    columns_drop = ["Unnamed: 0.1", "Unnamed: 0", "timestamp"]
    data.drop(columns_drop, axis=1, inplace=True)
    X = data.drop("SOFA", axis=1).values
    y = data["SOFA"].values
    return X, y


# Скалирование данных
def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return X_train, X_train, y_train, y_test


# Сборка и компиляция модели
def build_and_compile(input_dim):
    model = Sequential()
    model.add(Dense(units=12, activation="relu", input_dim=input_dim))
    model.add(Dense(units=6, activation="relu"))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae", "mse", "accuracy"])
    return model


# Сбор метрик
class MetricsCallback(Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = X_test

    def on_epoch_end(self, epoch, logs=None):
        # Вычисление accuracy
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)

        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        train_mae = mean_absolute_error(self.y_train, train_pred)
        test_mae = mean_absolute_error(self.y_test, test_pred)

        # Вывод метрик
        print(
            f'Epoch {epoch + 1}/{self.params["epochs"]}, '
            f'Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}, '
            f'Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}'
        )


# Обучение модели с использованием созданного обратного вызова
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test)
    early_stopp = EarlyStopping(monitor="val_loss", patience=4)
    model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[metrics_callback, early_stopp],
    )


# Оценка модели ANN
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_eval = model.evaluate(X_train, y_train)
    test_eval = model.evaluate(X_test, y_test)
    return train_eval, test_eval


# Предикт и вычисление метрик
def predict_and_evaluate(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Расчет метрик на обучающем наборе
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    # Расчет метрик на тестовом наборе
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "train": {"mse": train_mse, "mae": train_mae, "r2": train_r2},
        "test": {"mse": test_mse, "mae": test_mae, "r2": test_r2},
    }


# MAIN функция для работы кода
def main(file_path):
    X, y = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_and_scale_data(X, y)
    model = build_compile_model(X_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test)
    train_eval, test_eval = evaluate_model(model, X_train, y_train, X_test, y_test)
    metrics = predict_and_evaluate(model, X_train, y_train, X_test, y_test)
    print("Train Evaluation:", train_eval)
    print("Test Evaluation:", test_eval)
    print("Metrics:", metrics)
    predictions = model.predict(X_test)
    visualize_results(y_test, predictions)


# file path
file_path = ""
main(file_path)
