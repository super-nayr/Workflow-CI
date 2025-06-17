import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import dagshub
import argparse

def setup_mlflow():
    """Menginisialisasi DagsHub & MLflow."""
    print("🚀 Menginisialisasi DagsHub dan MLflow...")
    dagshub.init(mlflow=True)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=True, log_model_signatures=True)
    print("✅ MLflow siap digunakan.")

def load_data(data_path: str):
    """Memuat dataset dan membaginya."""
    print(f"💾 Memuat data dari: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"::error::File tidak ditemukan: {data_path}")
        exit(1)

    target_column = "lung_cancer"
    if target_column not in df.columns:
        print(f"::error::Kolom target '{target_column}' tidak ditemukan.")
        exit(1)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Data dimuat: {len(X_train)} data train, {len(X_test)} data test")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, max_iter, C):
    print("🧠 Melatih model Logistic Regression...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")  # Penting!

        mlflow.log_params({"max_iter": max_iter, "C": C})
        mlflow.set_tag("Model", "LogisticRegression")

        model = LogisticRegression(max_iter=max_iter, C=C, solver="liblinear")
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, artifact_path="model_files")

        print(f"✅ Akurasi Model: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    train_model(X_train, X_test, y_train, y_test, args.max_iter, args.C)
    print("🎉 Training selesai tanpa error.")
