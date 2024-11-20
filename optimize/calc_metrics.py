import json
import logging

import numpy as np
import pandas as pd
from redis import Redis
from redis.commands.json.path import Path

from optimize.models import Settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to calculate precision at k
def precision_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    tp = len(set(y_true) & set(y_pred))
    return tp / k


# Function to calculate recall at k
def recall_at_k(y_true, y_pred, k):
    y_pred = y_pred[:k]
    tp = len(set(y_true) & set(y_pred))
    return tp / len(y_true)


def calc_ret_metrics(settings: Settings):
    logging.info(f"Calculating retrieval metrics for test_id: {settings.test_id}")
    client = Redis.from_url(settings.redis_url)
    res = client.json().get(f"eval:{settings.test_id}")
    ret_samples = res["distance_samples"]["retrieval"]["responses"]
    ret = pd.DataFrame.from_records(ret_samples)

    avg_query_latency = ret["query_latency"].mean()

    # Calculate precision and recall at k for each query
    ret["precision_at_k"] = ret.apply(
        lambda row: precision_at_k(
            row["ground_truth"], row["retrieved"], settings.ret_k
        ),
        axis=1,
    )
    ret["recall_at_k"] = ret.apply(
        lambda row: recall_at_k(row["ground_truth"], row["retrieved"], settings.ret_k),
        axis=1,
    )
    ret["f1_at_k"] = (
        2
        * (ret["precision_at_k"] * ret["recall_at_k"])
        / (ret["precision_at_k"] + ret["recall_at_k"])
    )

    ret.fillna(0, inplace=True)

    # Aggregate the results to get the overall precision and recall at k
    overall_precision_at_k = ret["precision_at_k"].mean()
    overall_recall_at_k = ret["recall_at_k"].mean()
    overall_f1_at_k = ret["f1_at_k"].mean()

    logging.info(f"Overall f1 at {settings.ret_k} for retrieval: {overall_f1_at_k}")

    client.json().set(
        f"eval:{settings.test_id}",
        f"{Path.root_path()}.metrics.retrieval",
        {
            "precision_at_k": overall_precision_at_k,
            "recall_at_k": overall_recall_at_k,
            "f1_at_k": overall_f1_at_k,
            "avg_query_latency": avg_query_latency,
        },
    )

    return (
        overall_f1_at_k,
        overall_precision_at_k,
        overall_recall_at_k,
        avg_query_latency,
    )


def calc_precision_recall_f1(conf_matrix):
    tp = conf_matrix.loc[1, 1] if 1 in conf_matrix else 0
    fp = conf_matrix.loc[0, 1] if 1 in conf_matrix else 0
    fn = conf_matrix.loc[1, 0] if 0 in conf_matrix else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def first_result_metrics(dist_threshold, tests_df):
    # only for first result currently
    tests_df["pred"] = int(float(tests_df["cos_dists"][0][0]) < dist_threshold)
    tests_df["actual"] = tests_df["is_pos"]  # not sure why I need to do this
    conf_matrix = tests_df.groupby(["actual", "pred"]).size().unstack().fillna(0)
    return calc_precision_recall_f1(conf_matrix)


def find_optimum_distance_threshold(tests: pd.DataFrame):
    thresholds = np.arange(0.01, 0.8, 0.025)
    f1_scores = [
        first_result_metrics(threshold, tests)["f1"] for threshold in thresholds
    ]
    max_f1 = max(f1_scores)
    best_threshold = thresholds[f1_scores.index(max_f1)]
    return best_threshold, max_f1


def calc_best_threshold(settings: Settings):
    logging.info(f"Calculating best threshold for test_id: {settings.test_id}")
    client = Redis.from_url(settings.redis_url)
    res = client.json().get(f"eval:{settings.test_id}")
    dist_samples = res["distance_samples"]["threshold"]["responses"]
    results_df = pd.DataFrame.from_records(dist_samples)

    best_threshold, max_f1 = find_optimum_distance_threshold(results_df)
    client.json().set(
        f"eval:{settings.test_id}",
        f"{Path.root_path()}.metrics.threshold",
        {"best_distance_threshold": best_threshold, "max_f1": max_f1},
    )
    logging.info(f"Best threshold: {best_threshold}, Max F1: {max_f1}")
    return best_threshold, max_f1
