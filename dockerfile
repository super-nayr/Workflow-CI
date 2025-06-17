# MLProject/dockerfile

# Gunakan base image yang ringan
FROM python:3.9-slim

# Terima Run ID sebagai argumen saat build
ARG MLFLOW_RUN_ID
# Terima token DagsHub sebagai argumen build untuk keamanan
ARG DAGSHUB_TOKEN

# Set environment variables
ENV MLFLOW_TRACKING_URI=https://dagshub.com/super-nayr/Workflow-CI.mlflow
ENV DAGSHUB_USER_TOKEN=${DAGSHUB_TOKEN}

# Set working directory
WORKDIR /app

# Salin file requirements dan install
COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Port yang akan digunakan oleh server
EXPOSE 8080

# Perintah untuk menyajikan model dari DagsHub menggunakan Run ID yang diberikan
# Model akan disajikan sebagai REST API di port 8080
CMD ["mlflow", "models", "serve", "-h", "0.0.0.0", "-p", "8080", "--model-uri", "runs:/${MLFLOW_RUN_ID}/model_files"]