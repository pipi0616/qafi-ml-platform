# Notebook 06: the safe where Airflow finds its Databricks credential.
#
# Airflow reads the secret airflow/connections/<id> as its connection <id> (see airflow/airflow.sh).
# Terraform creates the secret but never its value: notebook 06 puts the value in, straight from
# Databricks, so the credential never lands in the Terraform state file. About 0.40 USD / month.

resource "aws_secretsmanager_secret" "airflow_databricks" {
  name        = "airflow/connections/databricks_default"
  description = "Airflow's connection to Databricks: service principal qafi-airflow, OAuth (notebook 06)"
  kms_key_id  = aws_kms_key.lake.arn

  # dev: deleted at once by terraform destroy, so destroy and apply can follow each other.
  # Production keeps the default 30-day window to recover a secret deleted by mistake.
  recovery_window_in_days = 0
}

output "airflow_databricks_secret" {
  description = "Secret holding Airflow's Databricks connection; its value is set by notebook 06"
  value       = aws_secretsmanager_secret.airflow_databricks.name
}
