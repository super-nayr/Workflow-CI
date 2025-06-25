import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from mlflow.models.signature import infer_signature

def main(data_path):
    df = pd.read_csv(data_path)

    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Prediksi untuk signature dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Buat signature model supaya input/output terekam
        signature = infer_signature(X_test, y_pred)

        # Log model dengan mlflow.sklearn.log_model (bukan log_artifact)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="rf_best_model",
            signature=signature,
            input_example=X_test.iloc[:3]
        )

        # Log metric akurasi
        mlflow.log_metric("accuracy", acc)

        print(f"Akurasi: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)