from unittest.mock import MagicMock

import pandas as pd

from optimize.calc_metrics import (
    calc_best_threshold,
    calc_precision_recall_f1,
    calc_ret_metrics,
    find_optimum_distance_threshold,
    first_result_metrics,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 6, 7, 8]
    k = 3
    assert precision_at_k(y_true, y_pred, k) == 2 / 3


def test_recall_at_k():
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 6, 7, 8]
    k = 3
    assert recall_at_k(y_true, y_pred, k) == 2 / 5


def test_calc_precision_recall_f1():
    conf_matrix = pd.DataFrame([[5, 2], [1, 7]], index=[0, 1], columns=[0, 1])
    metrics = calc_precision_recall_f1(conf_matrix)
    assert metrics["precision"] == 7 / 9
    assert metrics["recall"] == 7 / 8
    assert metrics["f1"] == 2 * ((7 / 9) * (7 / 8)) / ((7 / 9) + (7 / 8))


def test_first_result_metrics():
    tests_df = pd.DataFrame(
        {"cos_dists": [[0.1], [0.4], [0.6], [0.8]], "is_pos": [1, 0, 1, 0]}
    )
    dist_threshold = 0.5
    # currently only looks at first result
    metrics = first_result_metrics(dist_threshold, tests_df)
    assert metrics["precision"] == 1 / 2
    assert metrics["recall"] == 1
    assert metrics["f1"] == 2 * (1 / 2) * (1) / ((1 / 2) + (1))


def test_find_optimum_distance_threshold():
    tests_df = pd.DataFrame(
        {"cos_dists": [[0.1], [0.4], [0.6], [0.8]], "is_pos": [1, 0, 1, 0]}
    )
    best_threshold, max_f1 = find_optimum_distance_threshold(tests_df)
    assert best_threshold == 0.11
    assert max_f1 == 2 / 3


def test_calc_ret_metrics(monkeypatch, settings):
    mock_redis = MagicMock()
    mock_redis.json().get.return_value = {
        "distance_samples": {
            "retrieval": {
                "responses": [
                    {"ground_truth": [1, 2, 3], "retrieved": [1, 2, 4]},
                    {"ground_truth": [1, 2, 3], "retrieved": [1, 3, 5]},
                ]
            }
        }
    }
    monkeypatch.setattr("optimize.calc_metrics.Redis.from_url", lambda _: mock_redis)
    overall_f1_at_k = calc_ret_metrics(settings)
    assert overall_f1_at_k == 2 / 3


def test_calc_best_threshold(monkeypatch, settings):
    mock_redis = MagicMock()
    mock_redis.json().get.return_value = {
        "distance_samples": {
            "threshold": {
                "responses": [
                    {"cos_dists": [0.1], "is_pos": 1},
                    {"cos_dists": [0.4], "is_pos": 0},
                    {"cos_dists": [0.6], "is_pos": 1},
                    {"cos_dists": [0.8], "is_pos": 0},
                ]
            }
        }
    }
    monkeypatch.setattr("optimize.calc_metrics.Redis.from_url", lambda _: mock_redis)
    best_threshold, max_f1 = calc_best_threshold(settings)
    assert best_threshold == 0.11
    assert max_f1 == 2 / 3
