import argparse
import shutil
import time
from importlib.metadata import version

import joblib
import mlflow
import pandas as pd
import yaml
from joblib import parallel_backend
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_validate,
)
from threadpoolctl import threadpool_limits

from pd_model.config import (
    FEATURES,
    SCHEMA_VERSION,
    TARGET,
    file_sha256,
    load_config,
    resolve_path,
    write_json,
)
from pd_model.data.validation import load_csv, require_valid
from pd_model.models.pipeline import create_pipeline
from pd_model.monitoring.reference import build_reference
from pd_model.tracking.mlflow_utils import configure_tracking, git_commit, log_pipeline

SCORING = {
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def fit_scenario(scenario, X, y, folds, config):
    pipeline = create_pipeline(
        scenario["algorithm"],
        scenario["aggregates"],
        config["training"]["seed"],
        scenario.get("params"),
    )
    jobs = config["training"]["n_jobs"]
    with parallel_backend("threading", n_jobs=jobs), threadpool_limits(limits=1):
        if scenario["search"] == "none":
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=folds,
                scoring=SCORING,
                n_jobs=jobs,
                error_score="raise",
            )
            metrics = {}
            for metric in SCORING:
                metrics[f"cv_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
                metrics[f"cv_{metric}_std"] = float(scores[f"test_{metric}"].std())
            pipeline.fit(X, y)
            results = pd.DataFrame(scores)
            best_params = pipeline.named_steps["classifier"].get_params()
        else:
            options = {
                "estimator": pipeline,
                "cv": folds,
                "scoring": SCORING,
                "refit": "roc_auc",
                "n_jobs": jobs,
                "error_score": "raise",
                "return_train_score": True,
            }
            if scenario["search"] == "grid":
                search = GridSearchCV(param_grid=scenario["space"], **options)
            else:
                search = RandomizedSearchCV(
                    param_distributions=scenario["space"],
                    n_iter=scenario["n_iter"],
                    random_state=config["training"]["seed"],
                    **options,
                )
            search.fit(X, y)
            index = search.best_index_
            metrics = {
                f"cv_{metric}_{stat}": float(
                    search.cv_results_[f"{stat}_test_{metric}"][index]
                )
                for metric in SCORING
                for stat in ("mean", "std")
            }
            pipeline = search.best_estimator_
            best_params = search.best_params_
            results = pd.DataFrame(search.cv_results_)
    return pipeline, metrics, best_params, results


def train(config, profile="full"):
    data_path = resolve_path(config, config["data"]["processed"]) / "train.csv"
    data = load_csv(data_path)
    require_valid(data, "processed")
    scenarios = yaml.safe_load(
        resolve_path(config, config["training"]["experiments"]).read_text()
    )["experiments"]
    cv_count = config["training"]["cv_folds"]
    if profile == "smoke":
        data = data.sample(
            n=min(len(data), config["training"]["smoke_rows"]),
            random_state=config["training"]["seed"],
        ).reset_index(drop=True)
        scenarios = [item for item in scenarios if item["id"] == "E1_lr_raw"]
        cv_count = 2
        models_dir = resolve_path(config, ".ci_artifacts/models")
        report_dir = resolve_path(config, ".ci_artifacts/reports")
    else:
        models_dir = resolve_path(config, config["training"]["models"])
        report_dir = resolve_path(config, "reports")
    X, y = data[FEATURES], data[TARGET]
    splitter = StratifiedGroupKFold(
        n_splits=cv_count, shuffle=True, random_state=config["training"]["seed"]
    )
    folds = list(splitter.split(X, y, groups=data["feature_group"]))
    for fold_train, fold_valid in folds:
        groups = data["feature_group"]
        if set(groups.iloc[fold_train]) & set(groups.iloc[fold_valid]):
            raise ValueError("Пересечение групп внутри CV")
    configure_tracking(config, profile)
    leaderboard = []
    trained = {}
    source_hash = config["data"]["source_sha256"]
    for scenario in scenarios:
        identifier = scenario["id"]
        print(f"START {identifier}: {len(data)} rows, {cv_count} folds", flush=True)
        started = time.perf_counter()
        with mlflow.start_run(run_name=identifier) as run:
            mlflow.set_tags(
                {
                    "run_type": (
                        "baseline" if identifier.startswith("E0") else "experiment"
                    ),
                    "scenario_id": identifier,
                    "profile": profile,
                    "git_commit": git_commit(config["_root"]),
                    "dataset_sha256": source_hash,
                    "schema_version": SCHEMA_VERSION,
                    "selection": "CV ROC-AUC; test not used",
                }
            )
            mlflow.log_params(
                {
                    "algorithm": scenario["algorithm"],
                    "aggregates": scenario["aggregates"],
                    "search": scenario["search"],
                    "cv_folds": cv_count,
                    "seed": config["training"]["seed"],
                    "threshold": config["training"]["threshold"],
                    "train_rows": len(data),
                    "search_budget": scenario.get("n_iter", "fixed/grid"),
                }
            )
            pipeline, metrics, best_params, results = fit_scenario(
                scenario, X, y, folds, config
            )
            metrics["training_seconds"] = time.perf_counter() - started
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {f"best_{key}": str(value) for key, value in best_params.items()}
            )
            candidate_dir = models_dir / "candidates" / identifier
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model_file = candidate_dir / "pipeline.joblib"
            joblib.dump(pipeline, model_file)
            model_info = log_pipeline(pipeline, X.head(3), config["_root"])
            metadata = {
                "scenario_id": identifier,
                "algorithm": scenario["algorithm"],
                "profile": profile,
                "model_version": f"{identifier}-{file_sha256(model_file)[:12]}",
                "model_sha256": file_sha256(model_file),
                "source_sha256": source_hash,
                "train_sha256": file_sha256(data_path),
                "schema_version": SCHEMA_VERSION,
                "features": FEATURES,
                "threshold": config["training"]["threshold"],
                "git_commit": git_commit(config["_root"]),
                "mlflow_run_id": run.info.run_id,
                "mlflow_model_uri": model_info.model_uri,
                "best_params": best_params,
                "cv_metrics": metrics,
                "selection_protocol": (
                    f"{cv_count} train group folds; champion selected before test"
                ),
                "versions": {
                    name: version(name)
                    for name in (
                        "scikit-learn",
                        "pandas",
                        "numpy",
                        "mlflow",
                        "great-expectations",
                    )
                },
            }
            write_json(candidate_dir / "metadata.json", metadata)
            result_path = report_dir / "experiments/cv_results" / f"{identifier}.csv"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(result_path, index=False)
            config_path = report_dir / "experiments/cv_results" / f"{identifier}.json"
            write_json(config_path, scenario)
            mlflow.log_artifact(str(result_path), "cv")
            mlflow.log_artifact(str(config_path), "configuration")
            mlflow.log_artifact(str(candidate_dir / "metadata.json"), "provenance")
            mlflow.log_artifact(str(config["_root"] / "params.yaml"), "configuration")
            trained[identifier] = (pipeline, metadata)
            leaderboard.append(
                {"scenario_id": identifier, **metrics, "run_id": run.info.run_id}
            )
            print(
                f"DONE {identifier}: CV ROC-AUC={metrics['cv_roc_auc_mean']:.5f}",
                flush=True,
            )
    main_rows = [
        item for item in leaderboard if not item["scenario_id"].startswith("E0")
    ]
    if profile == "full" and len(main_rows) != 5:
        raise ValueError("Не завершены все пять обязательных экспериментов")
    best = sorted(
        main_rows,
        key=lambda row: (
            -row["cv_roc_auc_mean"],
            row["cv_roc_auc_std"],
            row["scenario_id"],
        ),
    )[0]
    champion_id = best["scenario_id"]
    source = models_dir / "candidates" / champion_id
    champion = models_dir / "champion"
    champion.mkdir(parents=True, exist_ok=True)
    for name in ("pipeline.joblib", "metadata.json"):
        shutil.copy2(source / name, champion / name)
    pipeline, metadata = trained[champion_id]
    build_reference(
        pipeline,
        X,
        metadata,
        config["monitoring"]["reference"],
        models_dir / "monitoring_reference.json",
    )
    (report_dir / "experiments").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(leaderboard).to_csv(
        report_dir / "experiments/leaderboard.csv", index=False
    )
    write_json(
        report_dir / "metrics/cv_metrics.json",
        {
            "champion": champion_id,
            "selection_metric": "cv_roc_auc_mean",
            "runs": leaderboard,
        },
    )
    with mlflow.start_run(run_id=metadata["mlflow_run_id"]):
        mlflow.set_tag("selected_champion", "true")
        mlflow.log_artifact(str(models_dir / "monitoring_reference.json"), "monitoring")
    print(f"CHAMPION (selected before test): {champion_id}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    train(load_config(args.config), args.profile)


if __name__ == "__main__":
    main()
