import argparse
import warnings
from functools import partial

import numpy as np
import optuna
import yaml

from eval import Eval
from models import StudyConfig

# Mute all warnings
warnings.filterwarnings("ignore")


def load_config(config_path: str) -> StudyConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return StudyConfig(**config)


def calc_baseline(study_config):
    # for comparison run objective against baseline

    small = Eval(
        model_provider="hf",
        model_str="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        raw_data_path=study_config.raw_data_path,
        labeled_data_path=study_config.labeled_data_path,
        input_data_type=study_config.input_data_type,
        vector_data_type="float16",
        algorithm="flat",
        ret_k=6,  # maybe make a independent variable
        find_threshold=False,
    )

    small.calc_metrics()

    large = Eval(
        model_provider="hf",
        model_str="intfloat/e5-large-v2",
        embedding_dim=1024,
        raw_data_path=study_config.raw_data_path,
        labeled_data_path=study_config.labeled_data_path,
        input_data_type=study_config.input_data_type,
        vector_data_type="float32",
        algorithm="flat",
        ret_k=6,  # maybe make a independent variable
        find_threshold=False,
    )

    large.calc_metrics()

    return small, large


def cost_fn(metrics: list, weights: list):
    return np.dot(np.array(metrics), np.array(weights))


def norm_metrics(value: float, baseline: list):
    min_v = min(baseline + [value])
    max_v = max(baseline + [value])
    return (value - min_v) / (max_v - min_v)


def objective(trial, study_config, baseline_sm, baseline_lg):
    # we want to max the overall F1 score
    model_info = trial.suggest_categorical(
        "model_info",
        [m.model_dump() for m in study_config.embedding_models],
    )

    algorithm = trial.suggest_categorical("algorithm", ["flat", "hnsw"])
    vec_dtype = trial.suggest_categorical(
        "var_dtype", ["float16", "float32"]
    )  # only float32 for now
    ret_k = trial.suggest_int("ret_k", study_config.ret_k[0], study_config.ret_k[1])

    if algorithm == "hnsw":
        ef_runtime = trial.suggest_int(
            "ef_runtime", study_config.ef_runtime[0], study_config.ef_runtime[1]
        )
        ef_construction = trial.suggest_int(
            "ef_construction",
            study_config.ef_construction[0],
            study_config.ef_construction[1],
        )
        m = trial.suggest_int("m", study_config.m[0], study_config.m[1])

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

    norm_index_time = norm_metrics(
        e.total_indexing_time,
        [baseline_sm.total_indexing_time, baseline_lg.total_indexing_time],
    )
    norm_latency = norm_metrics(
        e.embedding_latency,
        [baseline_sm.embedding_latency, baseline_lg.embedding_latency],
    )

    metric_values = [e.f1_at_k, -norm_index_time, -norm_latency]

    print(f"Metrics: {metric_values}")

    # TODO: define better objective function, normalize, and maybe make tunable
    return cost_fn(metric_values, study_config.weights)


def run_study():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for Redis Vector Store given config file"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    study_config = load_config(args.config)

    # calculate baselines in order to normalize study
    baseline_sm, baseline_lg = calc_baseline(study_config)

    study = optuna.create_study(
        study_name="test",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )

    obj = partial(
        objective,
        study_config=study_config,
        baseline_sm=baseline_sm,
        baseline_lg=baseline_lg,
    )
    study.optimize(obj, n_trials=study_config.n_trials, n_jobs=study_config.n_jobs)
    print(f"Completed Bayesian optimization...")

    best_trial = study.best_trial
    print(f"Best Configuration: {best_trial.number}: {best_trial.params}:")
    print(f"Best Score: {best_trial.values}")


if __name__ == "__main__":
    run_study()
