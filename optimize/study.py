import argparse
import warnings
from functools import partial

import numpy as np
import optuna
import pandas as pd
import yaml
from redis import Redis
from redis.commands.json.path import Path

from optimize.eval import Eval
from optimize.models import StudyConfig

warnings.filterwarnings("ignore")

METRICS = {
    "ret_k": [],
    "algorithm": [],
    "ef_construction": [],
    "ef_runtime": [],
    "m": [],
    "distance_metric": [],
    "vector_data_type": [],
    "model": [],
    "model_dim": [],
    "f1@k": [],
    "embedding_latency": [],
    "indexing_time": [],
}


def load_config(config_path: str) -> StudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return StudyConfig(**config)


def update_metric_row(eval_obj):
    METRICS["ret_k"].append(eval_obj.settings.ret_k)
    METRICS["algorithm"].append(eval_obj.settings.index.algorithm)
    METRICS["ef_construction"].append(eval_obj.settings.index.ef_construction)
    METRICS["ef_runtime"].append(eval_obj.settings.index.ef_runtime)
    METRICS["m"].append(eval_obj.settings.index.m)
    METRICS["distance_metric"].append(eval_obj.settings.index.distance_metric)
    METRICS["vector_data_type"].append(eval_obj.settings.index.vector_data_type)
    METRICS["model"].append(eval_obj.settings.embedding.model)
    METRICS["model_dim"].append(eval_obj.settings.embedding.dim)
    METRICS["f1@k"].append(eval_obj.f1_at_k)
    METRICS["embedding_latency"].append(eval_obj.embedding_latency)
    METRICS["indexing_time"].append(eval_obj.total_indexing_time)


def cost_fn(metrics: list, weights: list):
    return np.dot(np.array(metrics), np.array(weights))


def norm_metric(value: float):
    return 1 / (1 + value)


def objective(trial, study_config):
    # we want to max the overall F1 score
    model_info = trial.suggest_categorical(
        "model_info",
        [m.model_dump() for m in study_config.embedding_models],
    )

    algorithm = trial.suggest_categorical("algorithm", study_config.algorithms)
    vec_dtype = trial.suggest_categorical("var_dtype", study_config.vector_data_types)

    ret_k = trial.suggest_int("ret_k", study_config.ret_k[0], study_config.ret_k[1])

    if algorithm == "hnsw":
        ef_runtime = trial.suggest_categorical("ef_runtime", study_config.ef_runtime)
        ef_construction = trial.suggest_categorical(
            "ef_construction", study_config.ef_construction
        )
        m = trial.suggest_categorical("m", study_config.m)

        print(
            f"\n\n Running for: \n model_str: {model_info['model']} \n ef_runtime: {ef_runtime} \n ef_construction: {ef_construction} \n m: {m} \n\n"
        )

        e = Eval(
            model_provider=model_info["provider"],
            model_str=model_info["model"],
            embedding_dim=model_info["dim"],
            raw_data_path=study_config.raw_data_path,
            labeled_data_path=study_config.labeled_data_path,
            input_data_type=study_config.input_data_type,
            vector_data_type=vec_dtype,
            algorithm=algorithm,
            ef_runtime=ef_runtime,
            ef_construction=ef_construction,
            m=m,
            ret_k=ret_k,  # maybe make a independent variable
            find_threshold=False,
        )
    else:
        print(
            f"\n\n Running for: \n model_str: {model_info['model']} \n algorithm: {algorithm}"
        )
        e = Eval(
            model_provider=model_info["provider"],
            model_str=model_info["model"],
            embedding_dim=model_info["dim"],
            raw_data_path=study_config.raw_data_path,
            labeled_data_path=study_config.labeled_data_path,
            input_data_type=study_config.input_data_type,
            vector_data_type=vec_dtype,
            algorithm=algorithm,
            ret_k=ret_k,
            find_threshold=False,
        )

    e.calc_metrics()

    update_metric_row(e)

    norm_index_time = norm_metric(e.total_indexing_time)
    norm_latency = norm_metric(e.embedding_latency)

    metric_values = [e.f1_at_k, norm_index_time, norm_latency]

    print(f"Metrics: {metric_values}")

    # TODO: define better objective function, normalize, and maybe make tunable
    return cost_fn(metric_values, study_config.weights)


def run_study(study_config: StudyConfig):

    study = optuna.create_study(
        study_name="test",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    obj = partial(objective, study_config=study_config)

    study.optimize(obj, n_trials=study_config.n_trials, n_jobs=study_config.n_jobs)
    print(f"Completed Bayesian optimization...")

    best_trial = study.best_trial
    print(f"Best Configuration: {best_trial.number}: {best_trial.params}:")
    print(f"Best Score: {best_trial.values}")

    # save study metrics to DB
    client = Redis.from_url(study_config.redis_url)

    client.json().set(f"study:{study_config.study_id}", Path.root_path(), METRICS)

    # pd.DataFrame(METRICS).to_csv(
    #     f"optimize/results/{study_config.study_id[:6]}-metrics.csv"
    # )

    # df = study.trials_dataframe()
    # df.to_csv(f"optimize/results/{study_config.study_id[:6]}-optuna_res.csv")


def run_study_cli():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for Redis Vector Store given config file"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    study_config = load_config(args.config)
    run_study(study_config)
