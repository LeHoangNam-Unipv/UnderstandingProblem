import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

parent_directory = os.getcwd()
now = datetime.now()
date_string = now.strftime("%d-%m-%Y-%H-%M-%S")

NUMBER_OF_FEATURES = 70  # Update based on the number of features
NUMB_ITERATION = 20

INPUT_FOLDER = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data\statistic\20s\slide2+3"


def load_and_preprocess_data(root_dir):
    X_train_U, X_train_N = [], []
    y_train_U, y_train_N = [], []
    validation_X_U, validation_X_N = [], []
    validation_y_U, validation_y_N = [], []

    def process_file(file_path):
        # Read the entire CSV file into a DataFrame (skip header row if present)
        df = pd.read_csv(file_path, header=None, usecols=range(NUMBER_OF_FEATURES), dtype='float64')

        # Check if there are any NaN values in the file
        if df.isnull().values.any():
            print(f"NaN values found in file: {file_path}")
            # Get the row and column indices of the NaN values
            nan_locations = df.isnull().stack()  # Stack the DataFrame to get boolean index
            nan_rows_cols = nan_locations[nan_locations].index.tolist()  # Filter where True (NaN values)

            # Print the specific rows and columns with NaN values
            for row, col in nan_rows_cols:
                print(f"NaN found at row {row + 1}, column {col + 1}")  # Add +1 for 1-based index

        return df.values  # Return as numpy array for easier concatenation

    # Load data for each folder 'U' and 'N'
    for folder, label in [('U', 0), ('N', 1)]:
        folder_path = os.path.join(root_dir, folder)

        # Get list of files in the folder
        tester_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        # Split files at the tester level into training and testing (80/20 split)
        train_files, test_files = train_test_split(tester_files, test_size=0.2)
        print(test_files)
        # Process training files
        for tester_file in train_files:
            file_path = os.path.join(folder_path, tester_file)
            tester_data = process_file(file_path)
            if folder == 'U':
                X_train_U.append(tester_data)
                y_train_U.extend([label] * len(tester_data))
            else:
                X_train_N.append(tester_data)
                y_train_N.extend([label] * len(tester_data))

        # Process testing files (for validation data, split by tester)
        for tester_file in test_files:
            tester_data = []  # Collect chunks for this tester
            tester_labels = []  # Collect labels for this tester
            file_path = os.path.join(folder_path, tester_file)
            file_data = process_file(file_path)

            for row in file_data:
                tester_data.append(row)
                tester_labels.append(label)

            # Add the tester's data and labels to validation arrays
            if folder == 'U':
                validation_X_U.append(np.array(tester_data))
                validation_y_U.append(tester_labels[0])  # Label for the entire tester
            else:
                validation_X_N.append(np.array(tester_data))
                validation_y_N.append(tester_labels[0])  # Label for the entire tester

    # Class balancing for training data
    min_train_samples = min(len(np.vstack(X_train_U)), len(np.vstack(X_train_N)))

    X_train_U = np.vstack(X_train_U)[:min_train_samples]
    y_train_U = y_train_U[:min_train_samples]
    X_train_N = np.vstack(X_train_N)[:min_train_samples]
    y_train_N = y_train_N[:min_train_samples]

    # Stack training data and labels
    X_train = np.vstack((X_train_U, X_train_N))
    y_train = np.array(y_train_U + y_train_N)  # Flatten labels into a 1D array


    # Class balancing for validation data
    min_test_samples = min(len(validation_X_U), len(validation_X_N))

    validation_X_U = validation_X_U[:min_test_samples]
    validation_y_U = validation_y_U[:min_test_samples]
    validation_X_N = validation_X_N[:min_test_samples]
    validation_y_N = validation_y_N[:min_test_samples]

    # Combine validation data
    validation_X = validation_X_U + validation_X_N
    validation_y = validation_y_U + validation_y_N

    return X_train, y_train, validation_X, validation_y

def return_predicted_label_and_confidence(pred_labels):
    # Count the number of 1s
    count_1 = np.sum(pred_labels)
    # Count the number of 0s
    count_0 = len(pred_labels) - count_1
    # Determine which value is higher and calculate the probability
    if count_1 > count_0:
        higher_value = 1
        higher_count = count_1
    else:
        higher_value = 0
        higher_count = count_0

    probability_higher_value = higher_count / len(pred_labels)
    return higher_value, probability_higher_value

def implementation():
    results = pd.DataFrame(
        columns=['clf_name', 'predicted_labels', 'predicted_validation_with_confidence',
                 'accuracy_by_tester'])

    # Example usage
    X_train, y_train, validation_X, validation_y = load_and_preprocess_data(INPUT_FOLDER)
    print(X_train.shape)
    print(len(validation_X))
    print(validation_X[0].shape)
    print(validation_y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    for i in range(len(validation_X)):
        validation_X[i] = scaler.fit_transform(validation_X[i])

    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=500),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=500),
        'AdaBoostClassifier':AdaBoostClassifier(n_estimators=500),
        'MLPClassifier':    MLPClassifier(),
        'KNeighborsClassifier':KNeighborsClassifier(),
        #'ComplementNB':    ComplementNB()
    }

    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X_train, y_train)

        # Validation loop (same as before)
        predicted_labels = []
        predicted_label_with_confidence = []

        for tester_index in range(len(validation_X)):
            tester_data = validation_X[tester_index]
            true_label = validation_y[tester_index]

            predictions = []

            # Get predictions for each sample in tester data
            for sample in tester_data:
                sample = sample.reshape(1, -1)  # Reshape sample to match the model's expected input shape
                pred = clf.predict(sample)
                predictions.append(pred[0])

            # Calculate the predicted label for this tester and the confidence
            pred_label, confidence = return_predicted_label_and_confidence(predictions)
            predicted_labels.append(int(pred_label))
            predicted_label_with_confidence.append([pred_label, confidence])

        # Calculate accuracy for the current classifier
        accuracy_by_tester = accuracy_score(validation_y, predicted_labels)

        print(f"Validation Accuracy for {clf_name}: {accuracy_by_tester * 100:.2f}%")
        result_in_one_run = {'clf_name': clf_name, 'predicted_labels': predicted_labels, 'predicted_validation_with_confidence': predicted_label_with_confidence,
                             'accuracy_by_tester': accuracy_by_tester}
        results = results._append(result_in_one_run, ignore_index=True)

    return results

def save_result_to_csv(file_path, results):
    f = os.path.join(parent_directory, "result", file_path)
    # Append new DataFrame to the existing CSV file
    append_to_csv(f, results)

def append_to_csv(file_path, df):
    # Check if the file exists
    if not os.path.isfile(file_path):
        # File does not exist, write the DataFrame with the header
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        # File exists, append the DataFrame without the header
        df.to_csv(file_path, mode='a', index=False, header=False)

def main():
    for i in range(NUMB_ITERATION):
        results = implementation()
        save_result_to_csv(f"{str(date_string)}.csv", results)




main()