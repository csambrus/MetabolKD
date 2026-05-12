"""Metabolomics ML utilities and validated benchmark pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import SEED

HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None


def random_forest_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    positive_class: str,
    *,
    n_estimators: int = 300,
    max_depth: int | None = 12,
    random_state: int = SEED,
    n_jobs: int = -1,
) -> pd.Series:
    """Fast exploratory RF importance for binary labels."""
    y = y.reindex(X.index)
    y01 = (y.astype(str) == str(positive_class)).astype(int).to_numpy()
    xm = X.astype(float).fillna(0).to_numpy()
    xm = np.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    rf.fit(xm, y01)
    return pd.Series(rf.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)


def _ensure_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, np.ndarray):
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    raise TypeError(f"Unsupported X type: {type(X)}")


def _ensure_series(y: pd.Series | np.ndarray | list[Any], index: pd.Index | None = None) -> pd.Series:
    if isinstance(y, pd.Series):
        ys = y.copy()
    elif isinstance(y, (np.ndarray, list)):
        ys = pd.Series(y)
    else:
        raise TypeError(f"Unsupported y type: {type(y)}")
    if index is not None and len(index) == len(ys):
        ys.index = index
    return ys


def _encode_binary_target(y: pd.Series, positive_label: str) -> pd.Series:
    y_str = y.astype(str)
    allowed = {"Control", str(positive_label)}
    unique = set(y_str.unique())
    if not unique.issubset(allowed):
        unknown = sorted(unique - allowed)
        raise ValueError(
            f"Unsupported class labels in y: {unknown}. "
            f"Expected only 'Control' and '{positive_label}'."
        )
    return (y_str == str(positive_label)).astype(int)


class SampleNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "median", eps: float = 1e-12):
        self.method = method
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        if self.method in ("none", None):
            return x
        if self.method == "median":
            denom = np.nanmedian(x, axis=1, keepdims=True)
        elif self.method in ("total", "tic"):
            denom = np.nansum(x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown sample normalization method: {self.method}")
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
        return x / denom


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_min: float = 0.0):
        self.clip_min = clip_min

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_min is not None:
            x = np.clip(x, self.clip_min, None)
        return np.log1p(x)


class ParetoScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        self.scale_ = np.sqrt(std + self.eps)
        self.scale_ = np.where(self.scale_ < self.eps, 1.0, self.scale_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("ParetoScaler is not fitted.")
        x = np.asarray(X, dtype=float)
        return (x - self.mean_) / self.scale_


class TensorFlowMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 500,
        early_stopping: bool = True,
        patience: int = 20,
        validation_split: float = 0.15,
        class_weight: dict[int, float] | None = None,
        use_balanced_class_weight: bool = True,
        random_state: int = SEED,
        verbose: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        self.class_weight = class_weight
        self.use_balanced_class_weight = use_balanced_class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.model_: tf.keras.Model | None = None
        self.history_ = None
        self.classes_ = np.array([0, 1], dtype=int)

    def _build_model(self, n_features: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(n_features,), dtype=tf.float32)
        x = inputs
        for units in self.hidden_layer_sizes:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tf_mlp_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model

    def _compute_class_weight(self, y_np: np.ndarray) -> dict[int, float] | None:
        if self.class_weight is not None:
            return self.class_weight
        if not self.use_balanced_class_weight:
            return None
        n_all = len(y_np)
        n_pos = int(np.sum(y_np == 1))
        n_neg = int(np.sum(y_np == 0))
        if n_pos == 0 or n_neg == 0:
            return None
        return {0: float(n_all / (2.0 * n_neg)), 1: float(n_all / (2.0 * n_pos))}

    def fit(self, X, y):
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.model_ = self._build_model(X_np.shape[1])
        callbacks = []
        if self.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )
        class_weight = self._compute_class_weight(y_np)
        val_split = self.validation_split if self.early_stopping else 0.0
        self.history_ = self.model_.fit(
            X_np,
            y_np,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("TensorFlowMLPClassifier is not fitted.")
        X_np = np.asarray(X, dtype=np.float32)
        p1 = self.model_.predict(X_np, verbose=0).reshape(-1)
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_train_valid_test_split(
    X,
    y,
    positive_label="Lung cancer",
    test_size=0.15,
    valid_size=0.15,
    random_state=SEED,
):
    X_df = _ensure_dataframe(X)
    y_s = _ensure_series(y, index=X_df.index)
    y_bin = _encode_binary_target(y_s, positive_label=positive_label)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X_df, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )
    valid_ratio_inside_train_valid = valid_size / (1.0 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_ratio_inside_train_valid,
        random_state=random_state,
        stratify=y_train_valid,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _build_preprocess_pipeline(
    sample_normalization: str = "median",
    scaler: str = "standard",
    variance_threshold: float = 0.0,
) -> Pipeline:
    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "pareto":
        scaler_step = ParetoScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("sample_normalization", SampleNormalizer(method=sample_normalization)),
            ("log1p", Log1pTransformer()),
            ("variance", VarianceThreshold(threshold=variance_threshold)),
            ("scaler", scaler_step),
        ]
    )


def _build_selector(method: str, k: int, random_state: int) -> BaseEstimator:
    if method == "kbest":
        return SelectKBest(score_func=f_classif, k=k)
    if method == "l1":
        return SelectFromModel(
            estimator=LogisticRegression(
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
                max_iter=4000,
            ),
            threshold=-np.inf,
            max_features=k,
        )
    raise ValueError(f"Unknown selector_method: {method}")


def _build_models(random_state: int = SEED) -> dict[str, BaseEstimator]:
    models: dict[str, BaseEstimator] = {
        "lasso": LogisticRegression(
            penalty="l1", solver="saga", class_weight="balanced", max_iter=4000, random_state=random_state
        ),
        "ridge": LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=4000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=random_state
        ),
        "mlp": TensorFlowMLPClassifier(
            hidden_layer_sizes=(128, 64),
            dropout_rate=0.2,
            learning_rate=1e-3,
            batch_size=32,
            max_epochs=500,
            early_stopping=True,
            patience=20,
            validation_split=0.15,
            class_weight=None,
            use_balanced_class_weight=True,
            random_state=random_state,
            verbose=0,
        ),
    }
    if HAS_XGBOOST:
        try:
            from xgboost import XGBClassifier as _XGBClassifier

            models["xgboost"] = _XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                tree_method="hist",
            )
        except Exception as exc:
            warnings.warn(f"xgboost import failed; XGBoost model skipped ({exc}).", RuntimeWarning)
    else:
        warnings.warn("xgboost is unavailable; XGBoost model is skipped.", RuntimeWarning)
    return models


def evaluate_binary_classifier(model, X, y, dataset_name, model_name):
    y_true = np.asarray(y, dtype=int)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": f"[[{tn},{fp}],[{fn},{tp}]]",
    }
    preds = pd.DataFrame(
        {"model": model_name, "dataset": dataset_name, "y_true": y_true, "y_pred": y_pred, "y_proba": y_prob}
    )
    return metrics, preds


def _features_after_variance(preprocess: Pipeline, original_feature_names: list[str]) -> np.ndarray:
    names = np.asarray(original_feature_names, dtype=object)
    variance_step = preprocess.named_steps["variance"]
    if hasattr(variance_step, "get_support"):
        names = names[variance_step.get_support()]
    return names


def extract_model_features(fitted_pipeline, original_feature_names, model_name, top_n=30):
    preprocess = fitted_pipeline.named_steps["preprocess"]
    selector = fitted_pipeline.named_steps["selector"]
    estimator = fitted_pipeline.named_steps["model"]
    names_after_variance = _features_after_variance(preprocess, list(original_feature_names))
    selected_names = names_after_variance
    selection_method = type(selector).__name__
    if hasattr(selector, "get_support"):
        selected_names = names_after_variance[selector.get_support()]
    importance = None
    if model_name in ("lasso", "ridge") and hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_).ravel())
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(selector, "scores_") and selector.scores_ is not None:
        scores = np.asarray(selector.scores_, dtype=float)
        if hasattr(selector, "get_support"):
            scores = scores[selector.get_support()]
        importance = np.nan_to_num(scores, nan=0.0)
        selection_method = f"{selection_method}+scores"
    if importance is None:
        importance = np.ones(len(selected_names), dtype=float)
    out = (
        pd.DataFrame(
            {"model": model_name, "feature": selected_names, "importance": importance, "selection_method": selection_method}
        )
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["model", "feature", "importance", "rank", "selection_method"]].head(top_n)


def plot_metric_comparison(metrics_df, metric="roc_auc", output_path=None):
    df = metrics_df.copy()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not present in metrics_df.")
    mean_df = df.groupby(["model", "dataset"], as_index=False)[metric].mean()
    models = sorted(mean_df["model"].unique().tolist())
    datasets = ["validation", "test"]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, ds in enumerate(datasets):
        vals = []
        for model in models:
            row = mean_df[(mean_df["model"] == model) & (mean_df["dataset"] == ds)]
            vals.append(row[metric].iloc[0] if len(row) else np.nan)
        offset = -width / 2 if idx == 0 else width / 2
        ax.bar(x + offset, vals, width=width, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison by {metric}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
    return fig, ax


def plot_roc_curves(predictions_df, output_path=None):
    df = predictions_df.copy()
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for model, group in df[df["dataset"] == "test"].groupby("model"):
        if group["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_proba"])
        auc_val = roc_auc_score(group["y_true"], group["y_proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{model} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test ROC curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
    return fig, ax


def plot_confusion_matrices(metrics_df, output_dir=None):
    figs = []
    for _, row in metrics_df.iterrows():
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm_norm = cm / row_sum
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        img = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n{cm_norm[i, j]*100:.1f}%", ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1], labels=["Control", "Lung cancer"])
        ax.set_yticks([0, 1], labels=["Control", "Lung cancer"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{row['model']} ({row['dataset']})")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        figs.append(fig)
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"cm_{row['model']}_{row['dataset']}.png".replace(" ", "_")
            fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
    return figs


def plot_top_features(selected_features_df, model_name=None, top_n=20, output_path=None):
    df = selected_features_df.copy()
    if model_name is not None:
        df = df[df["model"] == model_name]
    if df.empty:
        raise ValueError("No selected features available for plotting.")
    df = df.sort_values(["model", "rank"]).groupby("model", as_index=False).head(top_n)
    models = df["model"].unique().tolist()
    fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(10, 3.8 * len(models)))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        cur = df[df["model"] == model].sort_values("importance", ascending=True)
        ax.barh(cur["feature"], cur["importance"])
        ax.set_title(f"Top features - {model}")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
    return fig, axes


@dataclass
class _Artifacts:
    metrics: list[dict[str, Any]]
    predictions: list[pd.DataFrame]
    best_estimators: dict[str, Pipeline]
    selected_features: list[pd.DataFrame]


def run_ml_benchmark(
    X,
    y,
    output_dir="outputs/ml_benchmark",
    positive_label="Lung cancer",
    random_state=SEED,
    k_values=(50, 100, 200, 500),
    refit_on_train_valid=True,
    sample_normalization="median",
    scaler="standard",
    variance_threshold=0.0,
    selector_method="kbest",
    top_n_features=30,
    verbose=False,
):
    X_df = _ensure_dataframe(X)
    y_s = _ensure_series(y, index=X_df.index)
    X_train, X_valid, X_test, y_train, y_valid, y_test = make_train_valid_test_split(
        X_df,
        y_s,
        positive_label=positive_label,
        test_size=0.15,
        valid_size=0.15,
        random_state=random_state,
    )
    split_info = {
        "n_total": int(len(X_df)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "positive_label": str(positive_label),
        "negative_label": "Control",
    }
    out_dir = Path(output_dir)
    out_conf = out_dir / "confusion_matrices"
    out_top = out_dir / "top_features"
    out_conf.mkdir(parents=True, exist_ok=True)
    out_top.mkdir(parents=True, exist_ok=True)
    artifacts = _Artifacts(metrics=[], predictions=[], best_estimators={}, selected_features=[])
    model_pool = _build_models(random_state=random_state)
    for model_name, model in model_pool.items():
        if verbose:
            print(f"[INFO] model={model_name}")
        best_auc = -np.inf
        best_cfg = None
        best_valid_metrics = None
        best_valid_preds = None
        for k in k_values:
            k_eff = int(min(k, X_train.shape[1]))
            if k_eff <= 0:
                continue
            preprocess = _build_preprocess_pipeline(
                sample_normalization=sample_normalization, scaler=scaler, variance_threshold=variance_threshold
            )
            selector = _build_selector(selector_method, k_eff, random_state=random_state)
            pipe = Pipeline(steps=[("preprocess", preprocess), ("selector", selector), ("model", model)])
            pipe.fit(X_train, y_train)
            valid_metrics, valid_preds = evaluate_binary_classifier(
                model=pipe, X=X_valid, y=y_valid, dataset_name="validation", model_name=model_name
            )
            valid_metrics["k"] = int(k_eff)
            valid_metrics["selector_method"] = selector_method
            if valid_metrics["roc_auc"] > best_auc:
                best_auc = valid_metrics["roc_auc"]
                best_cfg = {"k": k_eff}
                best_valid_metrics = valid_metrics
                best_valid_preds = valid_preds
        if best_cfg is None or best_valid_metrics is None or best_valid_preds is None:
            continue
        preprocess = _build_preprocess_pipeline(
            sample_normalization=sample_normalization, scaler=scaler, variance_threshold=variance_threshold
        )
        selector = _build_selector(selector_method, best_cfg["k"], random_state=random_state)
        best_model_pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("selector", selector),
                ("model", _build_models(random_state=random_state)[model_name]),
            ]
        )
        if refit_on_train_valid:
            X_fit = pd.concat([X_train, X_valid], axis=0)
            y_fit = pd.concat([y_train, y_valid], axis=0)
        else:
            X_fit, y_fit = X_train, y_train
        best_model_pipeline.fit(X_fit, y_fit)
        test_metrics, test_preds = evaluate_binary_classifier(
            model=best_model_pipeline, X=X_test, y=y_test, dataset_name="test", model_name=model_name
        )
        test_metrics["k"] = int(best_cfg["k"])
        test_metrics["selector_method"] = selector_method
        artifacts.metrics.append(best_valid_metrics)
        artifacts.metrics.append(test_metrics)
        artifacts.predictions.append(best_valid_preds)
        artifacts.predictions.append(test_preds)
        artifacts.best_estimators[model_name] = best_model_pipeline
        feat_df = extract_model_features(
            fitted_pipeline=best_model_pipeline,
            original_feature_names=list(X_df.columns),
            model_name=model_name,
            top_n=top_n_features,
        )
        feat_df["k"] = int(best_cfg["k"])
        artifacts.selected_features.append(feat_df)
        plot_top_features(
            selected_features_df=feat_df,
            model_name=model_name,
            top_n=min(20, top_n_features),
            output_path=out_top / f"top_features_{model_name}.png",
        )
    metrics_df = pd.DataFrame(artifacts.metrics).sort_values(["model", "dataset"]).reset_index(drop=True)
    predictions_df = pd.concat(artifacts.predictions, ignore_index=True) if artifacts.predictions else pd.DataFrame()
    selected_features_df = (
        pd.concat(artifacts.selected_features, ignore_index=True)
        if artifacts.selected_features
        else pd.DataFrame(columns=["model", "feature", "importance", "rank", "selection_method"])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "ml_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "ml_predictions.csv", index=False)
    selected_features_df.to_csv(out_dir / "ml_selected_features.csv", index=False)
    if not metrics_df.empty:
        plot_metric_comparison(metrics_df=metrics_df, metric="roc_auc", output_path=out_dir / "metric_comparison_roc_auc.png")
        plot_metric_comparison(metrics_df=metrics_df, metric="f1", output_path=out_dir / "metric_comparison_f1.png")
        plot_confusion_matrices(metrics_df=metrics_df, output_dir=out_conf)
    if not predictions_df.empty:
        plot_roc_curves(predictions_df=predictions_df, output_path=out_dir / "roc_curves_test.png")
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "best_estimators": artifacts.best_estimators,
        "selected_features": selected_features_df,
        "split_info": split_info,
    }
