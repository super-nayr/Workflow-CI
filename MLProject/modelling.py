import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import dagshub
import argparse
import sys

def setup_mlflow():
    """Menginisialisasi DagsHub & MLflow."""
    print("🚀 Menginisialisasi DagsHub dan MLflow...")
    dagshub.init(mlflow=True)
    # Menonaktifkan autologging model untuk logging manual yang lebih terkontrol
    mlflow.sklearn.autolog(log_models=False, log_input_examples=True, log_model_signatures=True)
    print("✅ Pengaturan MLflow selesai.")

def load_data(data_path: str) -> tuple:
    """Memuat dataset dan membaginya menjadi data latih dan uji."""
    print(f"\n💾 Memuat data dari: {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"::error::File data tidak ditemukan di path: {data_path}")
        print("❌ Program dihentikan karena file tidak ditemukan.")
        sys.exit(1)
    except Exception as e:
        print(f"::error::Terjadi kesalahan saat membaca file: {e}")
        sys.exit(1)

    # Pastikan kolom target tersedia
    target_column = "lung_cancer"
    if target_column not in df.columns:
        print(f"::error::Kolom target '{target_column}' tidak ditemukan di dataset.")
        print(f"Kolom yang tersedia: {list(df.columns)}")
        sys.exit(1)

    # Split data
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Data dimuat. Set latih memiliki {len(X_train)} sampel.")
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, max_iter: int, C: float):
    """Melatih model dan mencatat hasilnya di dalam MLflow run."""
    print("\n🧠 Memulai pelatihan model...")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # Run ID untuk ditangkap oleh GitHub Actions
        print(f"MLflow Run ID: {run_id}")

        # Logging parameter dan tag
        mlflow.log_params({"max_iter": max_iter, "C": C, "solver": "liblinear"})
        mlflow.set_tag("ModelType", "LogisticRegression")

        # Training
        model = LogisticRegression(max_iter=max_iter, solver="liblinear", C=C)
        model.fit(X_train, y_train)

        # Evaluasi dan logging metrik
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Simpan model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_files"
        )

        print(f"✅ Pelatihan selesai. Akurasi Model: {accuracy:.4f}")

if __name__ == "__main__":
    print("📌 Memulai skrip modelling.py...")

    # Parsing argumen CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path ke file CSV data.")
    parser.add_argument("--C", type=float, default=1.0, help="Parameter regularisasi untuk Logistic Regression.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Jumlah iterasi maksimum.")
    args = parser.parse_args()

    # Pipeline eksekusi
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data(data_path=args.data_path)
    train_model(
        X_train, X_test, y_train, y_test,
        max_iter=args.max_iter,
        C=args.C
    )

    print("\n🏁 Skrip selesai. 🎉 Proses berhasil diselesaikan!")
