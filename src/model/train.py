# Import libraries
import argparse
import glob
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn


# Define functions
def main(args):
    # Enable autologging
    mlflow.autolog()

    # Start an MLflow run
    with mlflow.start_run():
        # Debugging: Print the training data path before loading
        print(f"DEBUG: training_data path received -> {args.training_data}")

        # Read data
        df = get_csvs_df(args.training_data)

        # Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # Train model
        train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    print(f"DEBUG: Checking existence of path -> {path}")
    if not os.path.isdir(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    try:
        print(f"DEBUG: Listing files in dataset path -> {path}")
        print(os.listdir(path))
    except Exception as e:
        print(f"ERROR: Unable to list files - {str(e)}")

    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """
    Splits the dataset into training and testing sets.
    Assumes the target column is named 'Diabetic'.
    """
    if 'Diabetic' not in df.columns:
        raise RuntimeError("The dataset must contain a 'Diabetic' column.")

    columns = [
        'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness',
        'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'
    ]
    X = df[columns].values
    y = df['Diabetic'].values

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # Train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")


def parse_args():
    # Setup argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest="training_data",
                        type=str, required=True)
    parser.add_argument("--reg_rate", dest="reg_rate",
                        type=float, default=0.01)

    # Parse args
    args = parser.parse_args()

    # Return args
    return args


# Run script
if __name__ == "__main__":
    # Add space in logs
    print("\n\n")
    print("*" * 60)

    # Parse args
    args = parse_args()

    # Run main function
    main(args)

    # Add space in logs
    print("*" * 60)
    print("\n\n")
