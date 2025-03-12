import pymysql
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from sklearn.linear_model import LinearRegression

# Connect to MariaDB
def get_numbers_from_db(limit=100):
    connection = pymysql.connect(host='192.168.1.73', user='suirl', password='letmeinnow', database='primes')
    cursor = connection.cursor()
    cursor.execute(f"SELECT multiPrimeNum FROM multiPrimes ORDER BY multiPrimeNum ASC LIMIT {limit}")
    numbers = [row[0] for row in cursor.fetchall()]
    connection.close()
    return numbers

# Detect a pattern in the numbers
def detect_pattern(numbers):
    if len(numbers) < 3:
        return "Not enough data to detect a pattern."

    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    if all(d == diffs[0] for d in diffs):
        return f"Arithmetic sequence with difference {diffs[0]}."

    ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers)-1) if numbers[i] != 0]
    if len(ratios) == len(numbers) - 1 and all(r == ratios[0] for r in ratios):
        return f"Geometric sequence with ratio {ratios[0]:.2f}."

    if len(numbers) >= 3:
        fib_check = all(numbers[i] == numbers[i-1] + numbers[i-2] for i in range(2, len(numbers)))
        if fib_check:
            return "Fibonacci-like sequence detected."

    return "Pattern is unclear, using AI prediction."

# Train the neural network model
def train_model(numbers):
    X = np.array(range(len(numbers))).reshape(-1, 1)  # Indices as input
    y = np.array(numbers)  # Numbers as output

    model = Sequential([
        layers.Dense(10, activation='relu', input_shape=(1,)),
        layers.Dense(1)
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
        pattern_description = detect_pattern(known_numbers)
        print(f"Detected pattern: {pattern_description}")

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
