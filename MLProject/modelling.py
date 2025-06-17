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
    print("✅ Pengaturan MLflow selesai.")

def load_data(data_path: str) -> tuple:
    """Memuat dataset dan membaginya."""
    print(f"\n💾 Memuat data dari: {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"::error::File data tidak ditemukan di path: {data_path}")
        exit(1)
        
    target_column = "LUNG_CANCER"
    if target_column not in df.columns:
        print(f"::error::Kolom target '{target_column}' tidak ditemukan di dataset.")
        exit(1)
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data dimuat. Set latih memiliki {len(X_train)} sampel.")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, max_iter: int, C: float):
    """Melatih model dan mencatat hasilnya."""
    print("\n🧠 Memulai pelatihan model...")
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

        mlflow.log_params({"max_iter": max_iter, "C": C, "solver": "liblinear"})
        mlflow.set_tag("ModelType", "LogisticRegression")
        
        model = LogisticRegression(max_iter=max_iter, solver="liblinear", C=C)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_files"
        )
        
        print(f"✅ Pelatihan selesai. Akurasi Model: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data(data_path=args.data_path)
    train_model(
        X_train, X_test, y_train, y_test,
        max_iter=args.max_iter,
        C=args.C
    )
    print("\n🎉 Proses berhasil diselesaikan!")