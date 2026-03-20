from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .paths import THESIS_ROOT


@dataclass(frozen=True)
class LimitationsBundle:
    label_budget: pd.DataFrame
    lofo_aggregate: pd.DataFrame
    lofo_family: pd.DataFrame


def get_limitations_bundle() -> LimitationsBundle:
    base = THESIS_ROOT / "data/frozen_inputs/benchmarks"
    return LimitationsBundle(
        label_budget=pd.read_csv(base / "label_budget_summary.csv"),
        lofo_aggregate=pd.read_csv(base / "fpos_leave_one_family_aggregate.csv"),
        lofo_family=pd.read_csv(base / "fpos_leave_one_family_family.csv"),
    )


def load_leave_one_family_out_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = get_limitations_bundle()
    return bundle.lofo_aggregate.copy(), bundle.lofo_family.copy()
