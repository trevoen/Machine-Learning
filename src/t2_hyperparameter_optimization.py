"""T2 pipeline: hyperparameter optimization and final model interpretation.

This script intentionally starts from the raw Ames dataset and rebuilds the
modeling matrix without target leakage. It removes auxiliary strain labels
before training and keeps every learned preprocessing step inside sklearn
pipelines evaluated by cross-validation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "ames_mutagenicity_data.csv"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics" / "t2"
FIGURES_DIR = RESULTS_DIR / "figures" / "t2"
MODELS_DIR = RESULTS_DIR / "models" / "t2"
DOCS_IMAGES_DIR = ROOT / "docs" / "Imagens"
RANDOM_STATE = 42

TARGET_COL = "Overall"
METADATA_COLS = ["Id", "Name", "CAS", "SMILES RDKit", "Partition"]
AUXILIARY_LABEL_COLS = ["TA98", "TA100", "TA102", "TA1535", "TA1537"]
MODEL_ORDER = ["Gradient Boosting", "Decision Tree", "Random Forest"]


def ensure_output_dirs() -> None:
    for directory in (METRICS_DIR, FIGURES_DIR, MODELS_DIR, DOCS_IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_leakage_safe_data(raw_data_path: Path) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Dataset bruto nao encontrado em {raw_data_path}. "
            "Baixe o CSV do Mendeley e salve como data/raw/ames_mutagenicity_data.csv."
        )

    df = pd.read_csv(raw_data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COL}' nao encontrada no dataset.")

    df = df[df[TARGET_COL] != 0].copy()
    y_raw = df[TARGET_COL].copy()

    columns_to_drop = [
        col for col in METADATA_COLS + AUXILIARY_LABEL_COLS + [TARGET_COL] if col in df.columns
    ]
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number]).copy()

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), index=y_raw.index, name=TARGET_COL)

    feature_report = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_mapping": {
            str(original): int(encoded)
            for original, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        },
        "dropped_columns": columns_to_drop,
        "class_counts_original": {str(k): int(v) for k, v in y_raw.value_counts().sort_index().items()},
        "class_counts_encoded": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
    (METRICS_DIR / "dataset_summary.json").write_text(
        json.dumps(feature_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return X, y, label_encoder


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def suggest_classifier(trial: optuna.Trial, model_name: str, search_profile: str) -> BaseEstimator:
    robust = search_profile == "robust"

    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=trial.suggest_int("model__n_estimators", 50, 500 if robust else 100),
            learning_rate=trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("model__max_depth", 1, 6 if robust else 3),
            min_samples_split=trial.suggest_int("model__min_samples_split", 2, 40 if robust else 20),
            min_samples_leaf=trial.suggest_int("model__min_samples_leaf", 1, 30 if robust else 15),
            subsample=trial.suggest_float("model__subsample", 0.6, 1.0),
            max_features=trial.suggest_categorical(
                "model__max_features",
                [None, "sqrt", "log2"] if robust else ["sqrt", "log2"],
            ),
        )

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            criterion=trial.suggest_categorical("model__criterion", ["gini", "entropy", "log_loss"]),
            max_depth=trial.suggest_int("model__max_depth", 2, 60 if robust else 35),
            min_samples_split=trial.suggest_int("model__min_samples_split", 2, 80 if robust else 50),
            min_samples_leaf=trial.suggest_int("model__min_samples_leaf", 1, 40 if robust else 25),
            max_features=trial.suggest_categorical("model__max_features", [None, "sqrt", "log2"]),
            class_weight=trial.suggest_categorical("model__class_weight", [None, "balanced"]),
            ccp_alpha=trial.suggest_float("model__ccp_alpha", 0.0, 0.02),
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            n_estimators=trial.suggest_int("model__n_estimators", 100, 800 if robust else 350),
            max_depth=trial.suggest_int("model__max_depth", 3, 80 if robust else 45),
            min_samples_split=trial.suggest_int("model__min_samples_split", 2, 60 if robust else 35),
            min_samples_leaf=trial.suggest_int("model__min_samples_leaf", 1, 30 if robust else 20),
            max_features=trial.suggest_categorical(
                "model__max_features",
                ["sqrt", "log2", None] if robust else ["sqrt", "log2"],
            ),
            bootstrap=trial.suggest_categorical("model__bootstrap", [True, False]),
            class_weight=trial.suggest_categorical("model__class_weight", [None, "balanced"]),
        )

    raise ValueError(f"Modelo nao suportado: {model_name}")


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    cv: StratifiedKFold,
    search_profile: str,
) -> tuple[Pipeline, dict[str, Any], optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        pipeline = build_pipeline(suggest_classifier(trial, model_name, search_profile))
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=model_name.replace(" ", "_").lower(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_pipeline = build_pipeline(suggest_classifier(study.best_trial, model_name, search_profile))
    cv_scores = cross_val_score(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1,
        error_score="raise",
    )
    best_pipeline.fit(X_train, y_train)

    summary = {
        "model": model_name,
        "best_cv_f1_weighted": float(study.best_value),
        "recomputed_cv_f1_weighted_mean": float(cv_scores.mean()),
        "recomputed_cv_f1_weighted_std": float(cv_scores.std()),
        "best_params": study.best_params,
    }
    return best_pipeline, summary, study


def evaluate_on_test(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    row: dict[str, float | str] = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "precision_class_0": precision_per_class[0],
        "recall_class_0": recall_per_class[0],
        "f1_class_0": f1_per_class[0],
        "support_class_0": support[0],
        "precision_class_1": precision_per_class[1],
        "recall_class_1": recall_per_class[1],
        "f1_class_1": f1_per_class[1],
        "support_class_1": support[1],
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(METRICS_DIR / f"confusion_matrix_{slugify(model_name)}.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Nao mutagenico", "Mutagenico"]).plot(
        ax=ax, values_format="d", cmap="Blues", colorbar=False
    )
    ax.set_title(f"Matriz de confusao - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{slugify(model_name)}.png", dpi=300)
    fig.savefig(DOCS_IMAGES_DIR / f"t2_confusion_matrix_{slugify(model_name)}.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title(f"Curva ROC - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_curve_{slugify(model_name)}.png", dpi=300)
    fig.savefig(DOCS_IMAGES_DIR / f"t2_roc_curve_{slugify(model_name)}.png", dpi=300)
    plt.close(fig)

    errors = X_test.copy()
    errors["y_true"] = y_test.to_numpy()
    errors["y_pred"] = y_pred
    errors["positive_probability"] = y_score
    errors["is_error"] = errors["y_true"] != errors["y_pred"]
    error_summary = (
        errors.groupby(["y_true", "y_pred"])
        .size()
        .reset_index(name="n_samples")
        .sort_values(["y_true", "y_pred"])
    )
    error_summary.to_csv(METRICS_DIR / f"error_summary_{slugify(model_name)}.csv", index=False)
    return row


def plot_model_comparison(test_results: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
    long_df = test_results.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=long_df, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Metrica")
    ax.set_ylabel("Score")
    ax.set_title("Comparacao dos modelos otimizados no teste")
    ax.legend(title="Modelo", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "optimized_models_test_comparison.png", dpi=300)
    fig.savefig(DOCS_IMAGES_DIR / "t2_optimized_models_test_comparison.png", dpi=300)
    plt.close(fig)


def interpret_best_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int,
    n_top_features: int,
) -> pd.DataFrame:
    variance_step = pipeline.named_steps["variance"]
    selected_features = np.array(X_test.columns)[variance_step.get_support()]
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(selected_features):
        native_importances = pd.Series(model.feature_importances_, index=selected_features)
        candidate_features = native_importances.sort_values(ascending=False).head(n_top_features).index.tolist()
    else:
        candidate_features = list(X_test.columns[:n_top_features])

    baseline_score = f1_score(
        y_test,
        pipeline.predict(X_test),
        average="weighted",
        zero_division=0,
    )
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, float | str]] = []

    for feature in candidate_features:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            score = f1_score(
                y_test,
                pipeline.predict(X_permuted),
                average="weighted",
                zero_division=0,
            )
            scores.append(baseline_score - score)

        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    importance_df = pd.DataFrame(
        rows,
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(METRICS_DIR / f"permutation_importance_{slugify(model_name)}.csv", index=False)

    top = importance_df.head(20).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4477AA")
    ax.set_xlabel("Queda media no F1 ponderado")
    ax.set_ylabel("Feature")
    ax.set_title(f"Importancia por permutacao - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"permutation_importance_{slugify(model_name)}.png", dpi=300)
    fig.savefig(DOCS_IMAGES_DIR / "t2_permutation_importance.png", dpi=300)
    plt.close(fig)
    return importance_df


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa a Parte 2 do trabalho de AM.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH, help="Caminho para o CSV bruto.")
    parser.add_argument("--trials", type=int, default=100, help="Numero de trials Optuna por modelo.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Numero de folds da validacao cruzada.")
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=10,
        help="Repeticoes da permutation importance para o melhor modelo.",
    )
    parser.add_argument(
        "--importance-top-features",
        type=int,
        default=30,
        help="Numero maximo de features candidatas avaliadas por permutation importance.",
    )
    parser.add_argument(
        "--search-profile",
        choices=["pragmatic", "robust"],
        default="pragmatic",
        help="Perfil do espaco de busca: pragmatic limita configuracoes caras; robust amplia os intervalos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        X, y, label_encoder = load_leakage_safe_data(args.data)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    split_summary = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
    }
    (METRICS_DIR / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_STATE)
    optimization_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}

    for model_name in MODEL_ORDER:
        print(f"\nOtimizando {model_name}...")
        pipeline, summary, study = optimize_model(
            model_name,
            X_train,
            y_train,
            args.trials,
            cv,
            args.search_profile,
        )
        fitted_models[model_name] = pipeline
        optimization_rows.append(summary)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(METRICS_DIR / f"optuna_trials_{slugify(model_name)}.csv", index=False)
        joblib.dump(pipeline, MODELS_DIR / f"{slugify(model_name)}_best_pipeline.joblib")

        print(f"Avaliando {model_name} no teste final...")
        test_rows.append(evaluate_on_test(model_name, pipeline, X_test, y_test))

    optimization_df = pd.DataFrame(optimization_rows)
    optimization_df["best_params_json"] = optimization_df["best_params"].apply(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    optimization_df.drop(columns=["best_params"]).to_csv(
        METRICS_DIR / "optimization_summary.csv", index=False
    )

    test_results = pd.DataFrame(test_rows).sort_values("f1_weighted", ascending=False)
    test_results.to_csv(METRICS_DIR / "final_test_results.csv", index=False)
    plot_model_comparison(test_results)

    best_model_name = str(test_results.iloc[0]["model"])
    best_pipeline = fitted_models[best_model_name]
    print(f"\nMelhor modelo no teste: {best_model_name}")
    importance_df = interpret_best_model(
        best_model_name,
        best_pipeline,
        X_test,
        y_test,
        args.importance_repeats,
        args.importance_top_features,
    )

    final_summary = {
        "best_model": best_model_name,
        "best_model_test_metrics": test_results.iloc[0].to_dict(),
        "top_20_permutation_features": importance_df.head(20).to_dict(orient="records"),
        "target_mapping": {
            str(original): int(encoded)
            for original, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        },
    }
    (METRICS_DIR / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nArtefatos gerados em:")
    print(f"  Metricas: {METRICS_DIR}")
    print(f"  Figuras:  {FIGURES_DIR}")
    print(f"  Modelos:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
