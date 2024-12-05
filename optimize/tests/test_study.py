from optimize.models import StudyConfig
from optimize.study import run_study


def test_run_study_happy_path(study_config: StudyConfig, test_db_client):
    run_study(study_config=study_config)

    results = test_db_client.json().get(f"study:{study_config.study_id}")

    assert results["obj_val"][0] > 0.0

    # cleanup
    test_db_client.flushall()
