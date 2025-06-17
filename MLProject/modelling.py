import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas.api.types import is_numeric_dtype
import mlflow
import dagshub
import argparse
import os

def setup_mlflow():
    """Menginisialisasi koneksi MLflow ke DagsHub."""
    print("🚀 Menginisialisasi MLflow dan DagsHub...")
    # Inisialisasi menggunakan nama repo Anda di DagsHub
    dagshub.init(repo_owner='super-nayr', repo_name='Workflow-CI', mlflow=True)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=True, log_model_signatures=True)
    print("✅ MLflow siap digunakan.")

def load_data(data_path):
    """Memuat, memproses, dan membagi data dari path yang diberikan."""
    print(f"💾 Memuat data dari: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"::error::File tidak ditemukan di path: {data_path}")
        exit(1)

    # --- PERBAIKAN UTAMA 1: NAMA KOLOM TARGET ---
    # PENTING: Pastikan nama kolom ini SAMA PERSIS dengan di file CSV Anda.
    # 'lung_cancer' (huruf kecil) adalah format yang paling umum.
    target_column = "lung_cancer" 
    
    if target_column not in df.columns:
        print(f"::error::Kolom target '{target_column}' tidak ditemukan dalam dataset.")
        exit(1)

    # --- PERBAIKAN UTAMA 2: KONVERSI KOLOM TARGET YANG LEBIH AMAN ---
    # Cek jika kolom target bukan numerik (misal: berisi 'YES'/'NO')
    if not is_numeric_dtype(df[target_column]):
        print(f"🔄 Mengonversi kolom target '{target_column}' dari teks ke numerik (YES=1, NO=0)...")
        df[target_column] = df[target_column].str.upper().apply(lambda x: 1 if x == 'YES' else 0)
    else:
        print(f"☑️ Kolom target '{target_column}' sudah dalam format numerik.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✅ Data dimuat: {len(X_train)} baris data latih, {len(X_test)} baris data uji")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, max_iter, C):
    """Melatih model, mencatat metrik & parameter, dan menyimpan model ke MLflow."""
    print("🧠 Memulai pelatihan model LogisticRegression...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}") # Penting untuk ditangkap oleh workflow CI

        mlflow.log_params({"max_iter": max_iter, "C": C})
        mlflow.set_tag("model_type", "LogisticRegression")

        model = LogisticRegression(max_iter=max_iter, C=C, solver="liblinear")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Menyimpan model ke artifact store MLflow (DagsHub)
        mlflow.sklearn.log_model(model, artifact_path="model_files")

        print(f"✅ Akurasi model: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model.")
    parser.add_argument("--data_path", type=str, default="survey_lung_cancer_clean.csv", help="Path ke file dataset CSV.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Iterasi maksimum untuk solver.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength.")
    args = parser.parse_args()

    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    train_model(X_train, X_test, y_train, y_test, args.max_iter, args.C)
    print("🎉 Proses training selesai dengan sukses.")