from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .paths import THESIS_ROOT


RETAINED_LOFO_VARIANTS = (
    "reference_anchor_stack_xgboost",
    "wf_embed_anchor_stack_xgboost",
)


@dataclass(frozen=True)
class LimitationsBundle:
    label_budget: pd.DataFrame
    lofo_aggregate: pd.DataFrame
    lofo_family: pd.DataFrame


def get_limitations_bundle() -> LimitationsBundle:
    base = THESIS_ROOT / "data/frozen_inputs/benchmarks"
    aggregate_base = THESIS_ROOT / "artifacts" / "aggregates" / "family_held_out" / "fpos"
    lofo_aggregate_path = aggregate_base / "leave_one_family_aggregate.csv"
    lofo_family_path = aggregate_base / "leave_one_family_by_family.csv"
    lofo_aggregate = pd.read_csv(lofo_aggregate_path if lofo_aggregate_path.exists() else (base / "fpos_leave_one_family_aggregate.csv"))
    lofo_family = pd.read_csv(lofo_family_path if lofo_family_path.exists() else (base / "fpos_leave_one_family_family.csv"))
    if "variant_id" in lofo_aggregate.columns:
        lofo_aggregate = lofo_aggregate[lofo_aggregate["variant_id"].isin(RETAINED_LOFO_VARIANTS)].copy()
    if "variant_id" in lofo_family.columns:
        lofo_family = lofo_family[lofo_family["variant_id"].isin(RETAINED_LOFO_VARIANTS)].copy()
    return LimitationsBundle(
        label_budget=pd.read_csv(base / "label_budget_summary.csv"),
        lofo_aggregate=lofo_aggregate.reset_index(drop=True),
        lofo_family=lofo_family.reset_index(drop=True),
    )


def load_leave_one_family_out_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = get_limitations_bundle()
    return bundle.lofo_aggregate.copy(), bundle.lofo_family.copy()
