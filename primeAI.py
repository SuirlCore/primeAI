import pymysql
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression

# Connect to MariaDB
def get_numbers_from_db(limit=100):
    connection = pymysql.connect(host='your_host', user='your_user', password='your_password', database='your_db')
    cursor = connection.cursor()
    cursor.execute(f"SELECT number FROM numbers_table ORDER BY id ASC LIMIT {limit}")
    numbers = [row[0] for row in cursor.fetchall()]
    connection.close()
    return numbers

# Train the neural network model
def train_model(numbers):
    X = np.array(range(len(numbers))).reshape(-1, 1)  # Indices as input
    y = np.array(numbers)  # Numbers as output

    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=500, verbose=0)
    return model

# Train a simple Linear Regression model (backup option)
def train_regression(numbers):
    X = np.array(range(len(numbers))).reshape(-1, 1)
    y = np.array(numbers)
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict the next number
def predict_next(model, length):
    return model.predict(np.array([[length]]))[0][0]

# Main logic loop
def run_prediction_loop():
    known_numbers = get_numbers_from_db(100)
    model = train_model(known_numbers)  # Initial training
    
    while True:
        predicted_number = round(predict_next(model, len(known_numbers)))
        actual_number = get_numbers_from_db(len(known_numbers) + 1)[-1]  # Fetch the next actual number

        print(f"Predicted: {predicted_number}, Actual: {actual_number}")

        if predicted_number == actual_number:
            known_numbers.append(actual_number)  # Continue predicting
        else:
            print("Prediction failed. Recalculating with new data...")
            known_numbers = get_numbers_from_db(len(known_numbers) + 1)  # Get the updated dataset
            model = train_model(known_numbers)  # Retrain the model

# Run the loop
run_prediction_loop()
