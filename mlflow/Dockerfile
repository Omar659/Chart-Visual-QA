# MLflow tracking server (Phase 2.3 / 3.6). The official image is minimal and ships no
# database driver, so add psycopg2 for the Postgres backend store. Everything else
# (the server, the UI) comes from the base image.
FROM ghcr.io/mlflow/mlflow:v2.22.0
RUN pip install --no-cache-dir psycopg2-binary==2.9.9
