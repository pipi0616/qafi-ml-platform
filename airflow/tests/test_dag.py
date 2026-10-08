"""The DAG must load and keep its shape: a DAG file that fails to load silently stops all of its runs."""

from pathlib import Path

import pytest

DAGS = Path(__file__).parents[1] / "dags"


@pytest.fixture(scope="module")
def dag():
    from airflow.dag_processing.dagbag import DagBag

    bag = DagBag(dag_folder=DAGS)
    assert bag.import_errors == {}
    return bag.dags["qafi_refresh"]


def test_tasks_run_in_order(dag):
    upstream = {task_id: task.upstream_task_ids for task_id, task in dag.task_dict.items()}
    assert upstream == {
        "fetch_new_releases": set(),
        "send_transfer_list": {"fetch_new_releases"},
        "run_lakehouse_job": {"fetch_new_releases", "send_transfer_list"},
    }


def test_runs_monthly_and_retries(dag):
    assert dag.timetable.expression == "0 6 10 * *"
    assert not dag.catchup and dag.max_active_runs == 1
    assert all(task.retries == 2 for task in dag.tasks)


def test_the_job_gets_the_versions_found_by_the_first_task(dag):
    run_job = dag.task_dict["run_lakehouse_job"]
    assert run_job.job_name == "qafi training data"
    assert run_job.job_parameters.operator.task_id == "fetch_new_releases"  # that task's result
