#!/usr/bin/env bash
# Runs any Airflow command with this project's settings (notebook 06), for example
#   airflow/airflow.sh standalone                 scheduler + web UI on http://localhost:8080
#   airflow/airflow.sh dags trigger qafi_refresh  start a run now
# Airflow keeps its database, logs and generated login in airflow/.airflow, which git ignores.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

export AIRFLOW_HOME="$here/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$here/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__API__HOST=127.0.0.1 # the web UI answers only on this computer, not on the network

# Connections are read from AWS Secrets Manager: connection <id> is the secret airflow/connections/<id>.
export AIRFLOW__SECRETS__BACKEND=airflow.providers.amazon.aws.secrets.secrets_manager.SecretsManagerBackend
export AIRFLOW__SECRETS__BACKEND_KWARGS='{"connections_prefix": "airflow/connections", "variables_prefix": null, "config_prefix": null}'

# AWS calls use your `aws login` session; an Airflow run by a company has its own IAM role instead.
export AWS_PROFILE="${AWS_PROFILE:-qafi}" AWS_DEFAULT_REGION=eu-central-1
export PYTHONPATH="$here/../src" # the DAG reuses the project's ingestion code

exec uv run --project "$here" --frozen airflow "$@"
