from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from qc_thesis.modeling.core.stack import StackSet, build_stack_frame as core_build_stack_frame
from qc_thesis.modeling.neural.mmd.model import target_inverse, target_transform
from qc_thesis.modeling.neural.mmd.train import score_predictions


@dataclass
class FittedTreeRegressor:
    backend: str
    imputer: SimpleImputer
    model: Any
    target_transform_mode: str


@dataclass
class DomainWeightResult:
    weights: np.ndarray
    summary: dict[str, float]
    classifier: Pipeline
    source_target_prob: np.ndarray | None = None
    target_target_prob: np.ndarray | None = None


@dataclass
class QuantileTransportFeatureMap:
    feature: str
    source_grid: np.ndarray
    target_grid: np.ndarray
    target_std: float


@dataclass
class QuantileTransportModel:
    feature_maps: dict[str, QuantileTransportFeatureMap]
    selected_features: list[str]
    quantile_points: int
    blend: float
    noise_scale: float


@dataclass
class KMeansSubgroupModel:
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    centers_: np.ndarray


@dataclass
class PCASupportProjector:
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    target_raw: np.ndarray
    target_scores: np.ndarray
    neighbors: NearestNeighbors


def _logit_inverse(values: np.ndarray, mode: str) -> np.ndarray:
    return np.clip(target_inverse(values, mode=mode), 0.0, 1.0)


def fit_tree_regressor(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    *,
    backend: str,
    random_state: int,
    target_transform_mode: str,
    lightgbm_estimators: int,
    lightgbm_learning_rate: float,
    lightgbm_num_leaves: int,
    xgboost_estimators: int,
    xgboost_learning_rate: float,
    xgboost_max_depth: int,
    xgboost_min_child_weight: float,
    xgboost_subsample: float,
    xgboost_colsample_bytree: float,
    sample_weight: np.ndarray | None = None,
) -> FittedTreeRegressor:
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    y_train_t = target_transform(y_train, mode=target_transform_mode)

    backend_norm = backend.lower()
    if backend_norm == "lightgbm":
        model = LGBMRegressor(
            n_estimators=lightgbm_estimators,
            learning_rate=lightgbm_learning_rate,
            num_leaves=lightgbm_num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif backend_norm == "xgboost":
        model = XGBRegressor(
            n_estimators=xgboost_estimators,
            learning_rate=xgboost_learning_rate,
            max_depth=xgboost_max_depth,
            min_child_weight=xgboost_min_child_weight,
            subsample=xgboost_subsample,
            colsample_bytree=xgboost_colsample_bytree,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise KeyError(f"Unknown tree backend: {backend}")

    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train_imp, y_train_t, **fit_kwargs)
    return FittedTreeRegressor(
        backend=backend_norm,
        imputer=imputer,
        model=model,
        target_transform_mode=target_transform_mode,
    )


def predict_tree_regressor(model: FittedTreeRegressor, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    X_imp = model.imputer.transform(X)
    pred_t = np.asarray(model.model.predict(X_imp), dtype=np.float32)
    return _logit_inverse(pred_t, model.target_transform_mode)


def filter_target_rows(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return df.loc[mask].copy(), y.loc[mask].to_numpy(dtype=float)


def compute_importance_weights(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    random_state: int,
    clip_min: float,
    clip_max: float,
    c_value: float,
) -> DomainWeightResult:
    source_X = source_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    target_X = target_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    domain_X = pd.concat([source_X, target_X], ignore_index=True)
    domain_y = np.concatenate([np.zeros(len(source_X), dtype=int), np.ones(len(target_X), dtype=int)])
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(domain_X, domain_y)
    p_target = np.clip(clf.predict_proba(source_X)[:, 1], 1e-4, 1.0 - 1e-4)
    odds = p_target / (1.0 - p_target)
    weights = odds / max(float(np.mean(odds)), 1e-6)
    weights = np.clip(weights, clip_min, clip_max)
    summary = {
        "n_source": float(len(source_df)),
        "n_target": float(len(target_df)),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "p10": float(np.quantile(weights, 0.10)),
        "p50": float(np.quantile(weights, 0.50)),
        "p90": float(np.quantile(weights, 0.90)),
        "max": float(np.max(weights)),
    }
    target_prob = np.clip(clf.predict_proba(target_X)[:, 1], 1e-4, 1.0 - 1e-4)
    return DomainWeightResult(
        weights=weights,
        summary=summary,
        classifier=clf,
        source_target_prob=p_target.astype(np.float32),
        target_target_prob=target_prob.astype(np.float32),
    )


def build_stack_frame(
    df: pd.DataFrame,
    *,
    context_cols: list[str],
    extra_features: dict[str, np.ndarray],
) -> pd.DataFrame:
    return core_build_stack_frame(df, context_cols=context_cols, extra_features=extra_features)


def build_stack_set(
    *,
    train_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_cols: list[str],
    train_features: pd.DataFrame | dict[str, object] | None = None,
    unlabeled_features: pd.DataFrame | dict[str, object] | None = None,
    test_features: pd.DataFrame | dict[str, object] | None = None,
) -> StackSet:
    return StackSet.from_context_frames(
        train_df=train_df,
        unlabeled_df=unlabeled_df,
        test_df=test_df,
        context_cols=context_cols,
        train_features=train_features,
        unlabeled_features=unlabeled_features,
        test_features=test_features,
    )


def make_study_indicator_frame(
    df: pd.DataFrame,
    *,
    categories: list[str] | tuple[str, ...] | None = None,
    prefix: str = "study",
) -> pd.DataFrame:
    study = df["study_set"].astype(str)
    if categories is None:
        categories = sorted(study.unique().tolist())
    cat = pd.Categorical(study, categories=list(categories))
    frame = pd.get_dummies(cat, prefix=prefix, dtype=float)
    return frame.reset_index(drop=True)


def is_shape_feature(column: str) -> bool:
    name = str(column)
    return name.startswith("wf_bin_") or name.startswith("acg_")


def mask_feature_columns(
    df: pd.DataFrame,
    drop_cols: list[str] | tuple[str, ...] | set[str],
) -> pd.DataFrame:
    cols = [col for col in drop_cols if col in df.columns]
    if not cols:
        return df.copy()
    out = df.copy()
    for column in cols:
        out[column] = np.nan
    return out


def select_shift_filtered_features(
    shift_df: pd.DataFrame,
    *,
    allowed_cols: list[str],
    top_n: int,
    min_abs_smd: float,
    preserve_shape_features: bool,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    allowed = set(str(col) for col in allowed_cols)
    working = shift_df[shift_df["feature"].astype(str).isin(allowed)].copy()
    if working.empty:
        manifest = pd.DataFrame(
            columns=[
                "feature_set",
                "feature",
                "feature_group",
                "abs_smd",
                "ks_stat",
                "missing_gap",
                "is_shape_feature",
                "protected_reason",
                "selected_for_drop",
            ]
        )
        return manifest, []

    working["feature_set"] = feature_set
    working["is_shape_feature"] = working["feature"].astype(str).map(is_shape_feature)
    working["protected_reason"] = np.where(
        working["is_shape_feature"] & preserve_shape_features,
        "preserve_shape_feature",
        "",
    )
    candidates = working[
        (working["abs_smd"].astype(float) >= float(min_abs_smd))
        & (~working["is_shape_feature"] | (not preserve_shape_features))
    ].copy()
    candidates = candidates.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
    selected = set(candidates.head(int(top_n))["feature"].astype(str).tolist())
    working["selected_for_drop"] = working["feature"].astype(str).isin(selected)
    manifest = working.loc[
        :,
        [
            "feature_set",
            "feature",
            "feature_group",
            "abs_smd",
            "ks_stat",
            "missing_gap",
            "is_shape_feature",
            "protected_reason",
            "selected_for_drop",
        ],
    ].sort_values(["selected_for_drop", "abs_smd", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    return manifest, sorted(selected)


def select_shift_protected_features(
    shift_df: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    target: str,
    allowed_cols: list[str],
    top_n: int,
    min_abs_smd: float,
    preserve_shape_features: bool,
    protect_top_k: int,
    min_abs_corr_to_protect: float,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    manifest, _ = select_shift_filtered_features(
        shift_df,
        allowed_cols=allowed_cols,
        top_n=top_n,
        min_abs_smd=min_abs_smd,
        preserve_shape_features=preserve_shape_features,
        feature_set=feature_set,
    )
    if manifest.empty:
        manifest["target"] = target
        manifest["filter_strategy"] = "target_protected_shift_filter"
        manifest["paired_abs_corr"] = np.nan
        return manifest, []

    feature_df = train_df.loc[:, [col for col in allowed_cols if col in train_df.columns]].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train_df[target], errors="coerce")
    corr_rows: list[dict[str, float | str]] = []
    for feature in feature_df.columns:
        mask = y.notna() & feature_df[feature].notna()
        if int(mask.sum()) < 8:
            corr = 0.0
        else:
            x_vals = feature_df.loc[mask, feature].to_numpy(dtype=float)
            y_vals = y.loc[mask].to_numpy(dtype=float)
            if float(np.nanstd(x_vals)) <= 1e-10 or float(np.nanstd(y_vals)) <= 1e-10:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(x_vals, y_vals)[0, 1]))
                if not np.isfinite(corr):
                    corr = 0.0
        corr_rows.append({"feature": str(feature), "paired_abs_corr": corr})
    corr_df = pd.DataFrame(corr_rows)
    working = manifest.merge(corr_df, on="feature", how="left")
    working["paired_abs_corr"] = working["paired_abs_corr"].fillna(0.0)
    protectable = working[
        (~working["is_shape_feature"] | (not preserve_shape_features))
        & (working["paired_abs_corr"].astype(float) >= float(min_abs_corr_to_protect))
    ].copy()
    protectable = protectable.sort_values(
        ["paired_abs_corr", "abs_smd", "feature"],
        ascending=[False, False, True],
    )
    protected = set(protectable.head(int(protect_top_k))["feature"].astype(str).tolist())

    reasons = working["protected_reason"].astype(str).replace("nan", "")
    new_reasons: list[str] = []
    for feature, existing in zip(working["feature"].astype(str), reasons):
        reason = existing.strip()
        if feature in protected:
            reason = "preserve_target_predictive_feature" if not reason else f"{reason}|preserve_target_predictive_feature"
        new_reasons.append(reason)
    working["protected_reason"] = new_reasons

    candidates = working[
        (working["abs_smd"].astype(float) >= float(min_abs_smd))
        & (~working["feature"].astype(str).isin(protected))
        & (~working["is_shape_feature"] | (not preserve_shape_features))
    ].copy()
    candidates = candidates.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
    selected = set(candidates.head(int(top_n))["feature"].astype(str).tolist())
    working["selected_for_drop"] = working["feature"].astype(str).isin(selected)
    working["target"] = target
    working["filter_strategy"] = "target_protected_shift_filter"
    working = working.sort_values(["selected_for_drop", "abs_smd", "feature"], ascending=[False, False, True]).reset_index(drop=True)
    return working, sorted(selected)


def fit_quantile_transport_model(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    shift_df: pd.DataFrame | None = None,
    top_n: int | None = None,
    min_abs_smd: float = 0.0,
    quantile_points: int = 129,
    blend: float = 1.0,
    noise_scale: float = 0.0,
    min_feature_rows: int = 64,
) -> QuantileTransportModel:
    allowed = [str(col) for col in feature_cols if col in source_df.columns and col in target_df.columns]
    if shift_df is not None:
        working = shift_df[shift_df["feature"].astype(str).isin(allowed)].copy()
        working = working[working["abs_smd"].astype(float) >= float(min_abs_smd)].copy()
        working = working.sort_values(["abs_smd", "ks_stat", "feature"], ascending=[False, False, True])
        if top_n is not None:
            selected_features = working.head(int(top_n))["feature"].astype(str).tolist()
        else:
            selected_features = working["feature"].astype(str).tolist()
    else:
        selected_features = list(allowed if top_n is None else allowed[: int(top_n)])

    q = np.linspace(0.0, 1.0, int(max(9, quantile_points)))
    feature_maps: dict[str, QuantileTransportFeatureMap] = {}
    for feature in selected_features:
        source_vals = pd.to_numeric(source_df[feature], errors="coerce").to_numpy(dtype=float)
        target_vals = pd.to_numeric(target_df[feature], errors="coerce").to_numpy(dtype=float)
        source_vals = source_vals[np.isfinite(source_vals)]
        target_vals = target_vals[np.isfinite(target_vals)]
        if len(source_vals) < int(min_feature_rows) or len(target_vals) < int(min_feature_rows):
            continue
        source_grid = np.quantile(source_vals, q)
        target_grid = np.quantile(target_vals, q)
        source_grid = np.maximum.accumulate(source_grid.astype(float))
        target_grid = np.maximum.accumulate(target_grid.astype(float))
        feature_maps[str(feature)] = QuantileTransportFeatureMap(
            feature=str(feature),
            source_grid=source_grid.astype(np.float32),
            target_grid=target_grid.astype(np.float32),
            target_std=float(np.std(target_vals)),
        )
    return QuantileTransportModel(
        feature_maps=feature_maps,
        selected_features=sorted(feature_maps.keys()),
        quantile_points=int(len(q)),
        blend=float(blend),
        noise_scale=float(noise_scale),
    )


def apply_quantile_transport_model(
    df: pd.DataFrame,
    model: QuantileTransportModel,
    *,
    random_state: int,
) -> pd.DataFrame:
    if not model.feature_maps:
        return df.copy()
    rng = np.random.default_rng(int(random_state))
    out = df.copy()
    for feature, fmap in model.feature_maps.items():
        values = pd.to_numeric(out[feature], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        transported = np.interp(values[mask], fmap.source_grid, fmap.target_grid)
        if float(model.blend) < 1.0:
            transported = ((1.0 - float(model.blend)) * values[mask]) + (float(model.blend) * transported)
        if float(model.noise_scale) > 0.0 and fmap.target_std > 0.0:
            transported = transported + (rng.normal(0.0, float(model.noise_scale), size=transported.shape) * fmap.target_std)
        values[mask] = transported
        out[feature] = values
    return out


def fit_kmeans_subgroup_model(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_clusters: int,
    random_state: int,
) -> KMeansSubgroupModel:
    feature_cols = [str(col) for col in feature_cols if col in df.columns]
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one present column")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_clusters_eff = min(max(2, int(n_clusters)), len(X_scaled))
    kmeans = KMeans(n_clusters=n_clusters_eff, n_init=20, random_state=random_state)
    kmeans.fit(X_scaled)
    return KMeansSubgroupModel(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        centers_=kmeans.cluster_centers_.astype(np.float32),
    )


def assign_kmeans_subgroups(
    df: pd.DataFrame,
    subgroup_model: KMeansSubgroupModel,
) -> np.ndarray:
    X = subgroup_model.imputer.transform(df.loc[:, subgroup_model.feature_cols].apply(pd.to_numeric, errors="coerce"))
    X_scaled = subgroup_model.scaler.transform(X)
    distances_sq = ((X_scaled[:, None, :] - subgroup_model.centers_[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances_sq, axis=1).astype(int)


def fit_pca_support_projector(
    target_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    n_components: int,
    n_neighbors: int,
) -> PCASupportProjector:
    feature_cols = [str(col) for col in feature_cols if col in target_df.columns]
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one present column")
    imputer = SimpleImputer(strategy="median")
    raw = imputer.fit_transform(target_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce"))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    n_components_eff = min(max(2, int(n_components)), scaled.shape[1], max(2, len(scaled) - 1))
    pca = PCA(n_components=n_components_eff, random_state=0)
    scores = pca.fit_transform(scaled)
    neighbors = NearestNeighbors(n_neighbors=min(max(2, int(n_neighbors)), len(scores))).fit(scores)
    return PCASupportProjector(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        target_raw=raw.astype(np.float32),
        target_scores=scores.astype(np.float32),
        neighbors=neighbors,
    )


def make_local_regression_mixup(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    *,
    subgroup_labels: pd.Series | np.ndarray | None = None,
    n_synthetic: int,
    k_neighbors: int,
    alpha_range: tuple[float, float],
    random_state: int,
    noise_scale: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = feature_frame.reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    y = np.asarray(target, dtype=float).reshape(-1)
    if len(frame) != len(y):
        raise ValueError("feature_frame and target must have the same length")
    if len(frame) < 2 or int(n_synthetic) <= 0:
        return frame.iloc[0:0].copy(), np.array([], dtype=float)

    labels = None
    if subgroup_labels is not None:
        labels = pd.Series(subgroup_labels).astype(str).reset_index(drop=True)
        if len(labels) != len(frame):
            raise ValueError("subgroup_labels must match feature_frame length")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(frame)
    col_scale = np.nanstd(X, axis=0, ddof=0)
    col_scale = np.where(np.isfinite(col_scale) & (col_scale > 1e-8), col_scale, 1.0)
    X_scaled = (X - np.nanmean(X, axis=0)) / col_scale

    rng = np.random.default_rng(int(random_state))
    alpha_lo, alpha_hi = alpha_range
    synth_rows: list[np.ndarray] = []
    synth_y: list[float] = []
    global_nbrs = NearestNeighbors(n_neighbors=min(max(2, int(k_neighbors) + 1), len(frame))).fit(X_scaled)

    subgroup_cache: dict[str, tuple[np.ndarray, NearestNeighbors]] = {}
    if labels is not None:
        for subgroup, idx in labels.groupby(labels).groups.items():
            indices = np.asarray(sorted(idx), dtype=int)
            if len(indices) < 2:
                continue
            n_neighbors_eff = min(max(2, int(k_neighbors) + 1), len(indices))
            subgroup_cache[str(subgroup)] = (
                indices,
                NearestNeighbors(n_neighbors=n_neighbors_eff).fit(X_scaled[indices]),
            )

    for _ in range(int(n_synthetic)):
        i = int(rng.integers(0, len(frame)))
        subgroup = str(labels.iloc[i]) if labels is not None else None
        if subgroup is not None and subgroup in subgroup_cache:
            subgroup_idx, subgroup_nbrs = subgroup_cache[subgroup]
            distances, neighbors = subgroup_nbrs.kneighbors(X_scaled[i : i + 1], return_distance=True)
            candidates = subgroup_idx[neighbors[0]]
            candidates = candidates[candidates != i]
        else:
            distances, neighbors = global_nbrs.kneighbors(X_scaled[i : i + 1], return_distance=True)
            candidates = neighbors[0]
            candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        j = int(rng.choice(candidates))
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        row = (alpha * X[i]) + ((1.0 - alpha) * X[j])
        if noise_scale > 0.0:
            row = row + rng.normal(0.0, noise_scale, size=row.shape) * col_scale
        y_row = float(np.clip((alpha * y[i]) + ((1.0 - alpha) * y[j]), 0.0, 1.0))
        synth_rows.append(row.astype(float))
        synth_y.append(y_row)

    if not synth_rows:
        return frame.iloc[0:0].copy(), np.array([], dtype=float)
    synth_df = pd.DataFrame(np.vstack(synth_rows), columns=frame.columns)
    return synth_df, np.asarray(synth_y, dtype=float)


def make_latent_summary_frame(
    latent: np.ndarray,
    *,
    prefix: str = "latent_summary",
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    arr = np.asarray(latent, dtype=float)
    if arr.ndim != 2:
        raise ValueError("latent must be a 2D array")
    if arr.shape[1] == 0:
        arr = np.zeros((arr.shape[0], 1), dtype=float)
    summary = {
        f"{prefix}_mean": arr.mean(axis=1),
        f"{prefix}_std": arr.std(axis=1),
        f"{prefix}_min": arr.min(axis=1),
        f"{prefix}_max": arr.max(axis=1),
        f"{prefix}_l2": np.linalg.norm(arr, axis=1),
    }
    for quantile in quantiles:
        q_label = int(round(float(quantile) * 100.0))
        summary[f"{prefix}_q{q_label:02d}"] = np.quantile(arr, quantile, axis=1)
    return pd.DataFrame(summary)


def pick_r2_first_recommendation(
    metrics_df: pd.DataFrame,
    *,
    target: str,
    reference_variant_id: str,
    selection_metric: str,
    mae_guardrail_pct: float,
    min_r2_improvement: float,
) -> dict[str, Any]:
    target_df = metrics_df[metrics_df["target"] == target].copy()
    if target_df.empty:
        raise ValueError(f"No metric rows available for target={target}")
    if reference_variant_id not in set(target_df["variant_id"].astype(str)):
        raise KeyError(f"Reference variant {reference_variant_id} missing for target={target}")

    reference = target_df.loc[target_df["variant_id"].astype(str) == reference_variant_id].iloc[0]
    reference_mae = float(reference["mae"])
    reference_r2 = float(reference["r2"])
    mae_limit = reference_mae * (1.0 + float(mae_guardrail_pct))

    eligible = target_df[target_df["mae"].astype(float) <= mae_limit].copy()
    metric = selection_metric.lower()
    if metric == "r2":
        eligible = eligible.sort_values(["r2", "mae", "variant_id"], ascending=[False, True, True])
        best = eligible.iloc[0]
        improves = float(best["r2"]) >= (reference_r2 + float(min_r2_improvement))
    else:
        eligible = eligible.sort_values(["mae", "r2", "variant_id"], ascending=[True, False, True])
        best = eligible.iloc[0]
        improves = float(best["mae"]) <= reference_mae

    recommended = best if improves else reference
    recommended_variant = str(recommended["variant_id"])
    recommended_mae = float(recommended["mae"])
    recommended_r2 = float(recommended["r2"])
    return {
        "target": target,
        "selection_metric": metric,
        "reference_variant_id": reference_variant_id,
        "reference_mae": reference_mae,
        "reference_r2": reference_r2,
        "mae_guardrail_limit": mae_limit,
        "recommended_variant_id": recommended_variant,
        "recommended_mae": recommended_mae,
        "recommended_r2": recommended_r2,
        "r2_improvement": recommended_r2 - reference_r2,
        "mae_regression_pct": ((recommended_mae / reference_mae) - 1.0) if reference_mae > 0 else 0.0,
        "passes_guardrail": bool(recommended_mae <= mae_limit),
        "beats_reference": bool(recommended_variant != reference_variant_id),
    }


def make_metric_row(
    *,
    target: str,
    variant_id: str,
    family: str,
    backend: str,
    source_target: str | None,
    metrics: dict[str, float],
    notes: str,
) -> dict[str, Any]:
    return {
        "protocol": "main",
        "target": target,
        "variant_id": variant_id,
        "family": family,
        "backend": backend,
        "source_target": source_target,
        "notes": notes,
        **metrics,
    }


def make_prediction_frame(
    *,
    target: str,
    variant_id: str,
    family: str,
    backend: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protocol": "main",
            "target": target,
            "variant_id": variant_id,
            "family": family,
            "backend": backend,
            "row_uid": df["row_uid"].astype(str).to_numpy(),
            "recording_key": df["recording_key"].astype(str).to_numpy(),
            "study_set": df["study_set"].astype(str).to_numpy(),
            "true": y_true,
            "pred": pred,
            "abs_error": np.abs(pred - y_true),
        }
    )


__all__ = [
    "DomainWeightResult",
    "FittedTreeRegressor",
    "KMeansSubgroupModel",
    "PCASupportProjector",
    "QuantileTransportFeatureMap",
    "QuantileTransportModel",
    "apply_quantile_transport_model",
    "assign_kmeans_subgroups",
    "build_stack_frame",
    "build_stack_set",
    "compute_importance_weights",
    "filter_target_rows",
    "fit_kmeans_subgroup_model",
    "fit_pca_support_projector",
    "fit_quantile_transport_model",
    "fit_tree_regressor",
    "make_latent_summary_frame",
    "make_local_regression_mixup",
    "make_metric_row",
    "make_prediction_frame",
    "make_study_indicator_frame",
    "mask_feature_columns",
    "pick_r2_first_recommendation",
    "predict_tree_regressor",
    "score_predictions",
    "select_shift_filtered_features",
    "select_shift_protected_features",
]
