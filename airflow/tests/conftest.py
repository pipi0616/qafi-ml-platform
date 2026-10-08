import os
import sys
import tempfile
from pathlib import Path

# Airflow reads its settings when first imported: give it a throwaway home, not airflow/.airflow.
os.environ["AIRFLOW_HOME"] = tempfile.mkdtemp(prefix="airflow-test-")
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
# Like airflow.sh: the DAG and these tests use the project's code in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
