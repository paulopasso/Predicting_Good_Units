from pathlib import Path
import subprocess
from textwrap import dedent, indent

import nbformat


NB_PATH = Path("/Users/paulruiz/Documents/Predicting_Good_Units/Spike_UNIT_Quality.ipynb")
REPO_ROOT = NB_PATH.parent
NB_GIT_PATH = NB_PATH.name


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found for replacement:\n{old[:200]}")
    return text.replace(old, new, 1)


def replace_once_or_skip(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    return replace_once(text, old, new)


def insert_after(text: str, anchor: str, addition: str) -> str:
    if anchor not in text:
        raise RuntimeError(f"Expected anchor not found for insertion:\n{anchor[:200]}")
    return text.replace(anchor, anchor + addition, 1)


CELL5_POLICY_OLD = dedent(
    """
    RESUME_FROM_CHECKPOINTS      = True
    REUSE_EXISTING_TRAIN_FEATURES = True
    REUSE_EXISTING_TEST_FEATURES  = True
    FORCE_REBUILD_TRAIN_FEATURES  = False
    FORCE_REBUILD_TEST_FEATURES   = False
    FORCE_RETRAIN_MODELS          = False
    CLEAN_SCRATCH_AFTER_RUN       = True
    PHASE1_MAX_ITEMS              = None  # set e.g. 3 for debug batches
    PHASE1_TARGET_ROWS            = 10_000   # set None for full corpus extraction
    PHASE1_TARGET_ROWS_MARGIN     = 750      # allow slight overrun before stopping
    PHASE1_TARGET_OVERFETCH_RATIO = 1.10     # shortlist a bit above the target to avoid undershooting
    PHASE1_FALLBACK_UNITS_PER_ITEM = 24      # used when preflight unit counts are unavailable
    PHASE1_MAX_RUNTIME_HOURS      = 10.0     # set None to disable runtime budget
    RUN_PHASE1_PREFLIGHT          = False
    PHASE1_PRIORITY_STUDY_SET     = "SYNTH_JANELIA"
    PHASE1_ROUND_ROBIN_SORTERS    = True
    PREFLIGHT_STRICT              = True
    PREFLIGHT_FAIL_RATIO_THRESHOLD = 0.35  # fail-fast if too many rows are structurally bad
    PREFLIGHT_MAX_ROWS            = None
    PREFLIGHT_READY_TARGET_ROWS   = int(PHASE1_TARGET_ROWS * 1.6) if PHASE1_TARGET_ROWS else None
    ENFORCE_STUDYSET_SOURCE_CONTRACT = True
    PHASE1_SELF_CHECK_STRICT      = True
    PHASE1_MIN_SUCCESS_RATIO      = 0.55
    PHASE1_SCHEMA_AUDIT_ENABLED   = True
    PHASE1_SCHEMA_AUDIT_SAMPLE_PER_STUDY_SET = 1
    PHASE1_COMPARISON_DEBUG_MAX_ITEMS = 6
    PHASE1_COMPARISON_DEBUG_PRIORITY_STUDY_SET = "HYBRID_JANELIA"
    PHASE1_CONSTANT_TARGET_STRICT = True
    PHASE1_DEGENERATE_MAX_FRACTION = 0.50
    PHASE1_VALIDATE_JANELIA      = True
    """
).strip()


CELL5_POLICY_NEW = dedent(
    """
    RESUME_FROM_CHECKPOINTS       = True
    REUSE_EXISTING_TRAIN_FEATURES = True
    REUSE_EXISTING_TEST_FEATURES  = True
    FORCE_REBUILD_TRAIN_FEATURES  = False
    FORCE_REBUILD_TEST_FEATURES   = False
    FORCE_RETRAIN_MODELS          = False
    CLEAN_SCRATCH_AFTER_RUN       = True
    PHASE1_MAX_ITEMS              = None   # keep None for full-corpus extraction
    PHASE1_TARGET_ROWS            = None   # full corpus extraction; do not cap rows
    PHASE1_TARGET_ROWS_MARGIN     = 0
    PHASE1_TARGET_OVERFETCH_RATIO = 1.00
    PHASE1_FALLBACK_UNITS_PER_ITEM = 24
    PHASE1_MAX_RUNTIME_HOURS      = None   # let checkpointing/resume handle long runs
    RUN_PHASE1_PREFLIGHT          = False  # keep extraction semantics; avoid long preflight on every resume
    PHASE1_PRIORITY_STUDY_SET     = None
    PHASE1_ROUND_ROBIN_SORTERS    = True
    PREFLIGHT_STRICT              = True
    PREFLIGHT_FAIL_RATIO_THRESHOLD = 0.35
    PREFLIGHT_MAX_ROWS            = None
    PREFLIGHT_READY_TARGET_ROWS   = None
    ENFORCE_STUDYSET_SOURCE_CONTRACT = True
    PHASE1_SELF_CHECK_STRICT      = True
    PHASE1_MIN_SUCCESS_RATIO      = 0.55
    PHASE1_SCHEMA_AUDIT_ENABLED   = True
    PHASE1_SCHEMA_AUDIT_SAMPLE_PER_STUDY_SET = 1
    PHASE1_COMPARISON_DEBUG_MAX_ITEMS = 0
    PHASE1_COMPARISON_DEBUG_PRIORITY_STUDY_SET = "HYBRID_JANELIA"
    PHASE1_CONSTANT_TARGET_STRICT = True
    PHASE1_DEGENERATE_MAX_FRACTION = 0.50
    PHASE1_VALIDATE_JANELIA       = True
    PHASE1_SNAPSHOT_INTERVAL_MIN  = 30
    PHASE1_SANITY_CHECK_INTERVAL_MIN = 60
    PHASE1_SANITY_MIN_ROWS        = 200
    PHASE1_SANITY_MAX_MEAN_FEATURE_MISSING = 0.20
    PHASE1_SANITY_MAX_ALL_NULL_FEATURES = 32
    PHASE1_SANITY_MAX_DEGENERATE_FRACTION = 0.90
    PHASE1_SANITY_STRICT          = True
    """
).strip()


CELL5_PATHS_OLD = 'COMPARISON_DEBUG_SUMMARY_JSON = AUDIT_DIR / "comparison_debug_summary.json"\nDATA_CONTRACTS_JSON         = MANIFEST_DIR / "data_contracts.json"'
CELL5_PATHS_NEW = (
    'COMPARISON_DEBUG_SUMMARY_JSON = AUDIT_DIR / "comparison_debug_summary.json"\n'
    'PHASE1_PERIODIC_CHECK_JSON    = AUDIT_DIR / "phase1_periodic_check.json"\n'
    'PHASE1_PERIODIC_CHECK_JSONL   = AUDIT_DIR / "phase1_periodic_check_history.jsonl"\n'
    'DATA_CONTRACTS_JSON         = MANIFEST_DIR / "data_contracts.json"'
)


CELL5_PRINT_OLD = 'print(f"  Phase 1 max runtime (hours): {PHASE1_MAX_RUNTIME_HOURS}")'
CELL5_PRINT_NEW = (
    'print(f"  Phase 1 max runtime (hours): {PHASE1_MAX_RUNTIME_HOURS}")\n'
    'print(f"  Phase 1 snapshot interval (min): {PHASE1_SNAPSHOT_INTERVAL_MIN}")\n'
    'print(f"  Phase 1 sanity interval (min): {PHASE1_SANITY_CHECK_INTERVAL_MIN}")'
)


CELL5_PRINT2_OLD = 'print(f"  Comparison debug max items: {PHASE1_COMPARISON_DEBUG_MAX_ITEMS}")'
CELL5_PRINT2_NEW = (
    'print(f"  Comparison debug max items: {PHASE1_COMPARISON_DEBUG_MAX_ITEMS}")\n'
    'print(f"  Phase 1 sanity min rows: {PHASE1_SANITY_MIN_ROWS}")\n'
    'print(f"  Phase 1 sanity max mean feature missing: {PHASE1_SANITY_MAX_MEAN_FEATURE_MISSING}")\n'
    'print(f"  Phase 1 sanity max all-null features: {PHASE1_SANITY_MAX_ALL_NULL_FEATURES}")\n'
    'print(f"  Phase 1 sanity max degenerate fraction: {PHASE1_SANITY_MAX_DEGENERATE_FRACTION}")'
)


CELL12_REBUILD_OLD = dedent(
    """
    if FORCE_REBUILD_TRAIN_FEATURES:
        reset_path(TRAIN_PARQUET)
        reset_path(TRAIN_FEATURE_STATE_JSON)
        reset_path(TRAIN_PARTS_DIR)
        TRAIN_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    """
).strip()


CELL12_REBUILD_NEW = dedent(
    """
    if FORCE_REBUILD_TRAIN_FEATURES:
        reset_path(TRAIN_PARQUET)
        reset_path(TRAIN_FEATURE_STATE_JSON)
        reset_path(TRAIN_PARTS_DIR)
        reset_path(PHASE1_PERIODIC_CHECK_JSON)
        reset_path(PHASE1_PERIODIC_CHECK_JSONL)
        TRAIN_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    """
).strip()


CELL12_SKIP_OLD = dedent(
    """
    if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES:
        df_train = safe_read_parquet(TRAIN_PARQUET)
        log_status(f"Phase 1 skipped. Reusing training features from {TRAIN_PARQUET}")
    elif not SPIKEFOREST_AVAILABLE:
    """
).strip()


CELL12_SKIP_NEW = dedent(
    """
    _existing_train_state = read_json(TRAIN_FEATURE_STATE_JSON, default={})
    _current_phase1_keys = sorted(
        {
            sf_output_key(_row["study_set"], _row["study_name"], _row["recording_name"], _row["sorter_name"])
            for _row in ordered_selected_rows
        }
    )
    _existing_completed_keys = set(_existing_train_state.get("completed_keys", []))
    _missing_current_phase1_keys = sorted(set(_current_phase1_keys) - _existing_completed_keys)
    _reuse_train_complete = bool(
        _existing_train_state.get("complete", False) and not _missing_current_phase1_keys
    )
    if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES and not _reuse_train_complete:
        log_status(
            f"Partial training parquet found at {TRAIN_PARQUET}; "
            "Phase 1 state is incomplete or does not cover the current selection, "
            "so resume will continue from checkpointed part files."
        )
    if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES and _reuse_train_complete:
        df_train = safe_read_parquet(TRAIN_PARQUET)
        log_status(f"Phase 1 skipped. Reusing training features from {TRAIN_PARQUET}")
    elif not SPIKEFOREST_AVAILABLE:
    """
).strip()


CELL12_HELPER_ANCHOR = (
    '    if target_row_cap is not None:\n'
    '        log_status(\n'
    '            "Phase 1 row budget: "\n'
    '            f"existing_rows={rows_collected}, target={target_rows}, cap={target_row_cap}"\n'
    '        )\n\n'
)


CELL12_HELPER_INSERTION = indent(
    dedent(
        """

        snapshot_interval_sec = max(60.0, float(PHASE1_SNAPSHOT_INTERVAL_MIN) * 60.0)
        sanity_interval_sec = max(snapshot_interval_sec, float(PHASE1_SANITY_CHECK_INTERVAL_MIN) * 60.0)
        phase1_timers = {"snapshot_at": time.time(), "sanity_at": time.time()}

        def _phase1_collect_part_paths(rows):
            part_paths = []
            for _item in rows:
                try:
                    _row = _manifest_row_with_defaults(_item)
                except Exception:
                    continue
                _key = sf_output_key(_row["study_set"], _row["study_name"], _row["recording_name"], _row["sorter_name"])
                _part_path = TRAIN_PARTS_DIR / f"{safe_stem(_key)}.parquet"
                if _part_path.exists():
                    part_paths.append(_part_path)
            return part_paths

        def _phase1_build_snapshot_df(part_paths):
            if not part_paths:
                return pd.DataFrame()
            _df = pd.concat([safe_read_parquet(pp) for pp in part_paths], ignore_index=True)
            _df = _df.drop_duplicates(
                subset=["study_set", "study_name", "recording_name", "sorter_name", "unit_id"]
            ).reset_index(drop=True)
            if "group_key" not in _df.columns:
                _df["group_key"] = _df.apply(
                    lambda _row: recording_group_key(_row["study_set"], _row["study_name"], _row["recording_name"]),
                    axis=1,
                )
            if "row_uid" not in _df.columns:
                _df["row_uid"] = _df.apply(
                    lambda _row: make_row_uid(
                        _row["study_set"],
                        _row["study_name"],
                        _row["recording_name"],
                        _row["sorter_name"],
                        _row["unit_id"],
                    ),
                    axis=1,
                )
            if target_row_cap is not None and len(_df) > target_row_cap:
                _before_trim = int(len(_df))
                _df["_target_rank"] = _df["row_uid"].astype(str).map(
                    lambda v: int(hashlib.sha1(v.encode("utf-8")).hexdigest()[:12], 16)
                )
                _df = (
                    _df
                    .sort_values(["_target_rank", "row_uid"], kind="mergesort")
                    .head(int(target_row_cap))
                    .drop(columns=["_target_rank"])
                    .reset_index(drop=True)
                )
                log_status(
                    f"Phase 1 target cap applied during snapshot: kept {len(_df)} rows from {_before_trim}"
                )
            return stable_sort_frame(_df, ["group_key", "sorter_name", "unit_id", "row_uid"])

        def _phase1_atomic_write_snapshot(df_snapshot, reason):
            _tmp_path = TRAIN_PARQUET.with_suffix(".tmp.parquet")
            df_snapshot.to_parquet(_tmp_path, index=False)
            _tmp_path.replace(TRAIN_PARQUET)
            state["last_snapshot_at"] = datetime.now().isoformat()
            state["last_snapshot_reason"] = str(reason)
            state["last_snapshot_rows"] = int(len(df_snapshot))
            write_json(TRAIN_FEATURE_STATE_JSON, state)
            log_status(
                f"Phase 1 snapshot [{reason}] saved to {TRAIN_PARQUET} "
                f"| rows={len(df_snapshot):,} | parts={len(_phase1_collect_part_paths(selected_to_process))}"
            )

        def _phase1_snapshot_health(df_snapshot, part_paths, reason):
            meta_cols = {
                "study_set", "study_name", "recording_name", "sorter_name",
                "unit_id", "group_key", "row_uid", "match_status",
                "matched_gt_unit_id", "hungarian_gt_unit_id",
            }
            target_cols = {
                "fmiss", "fpos", "accuracy", "agreement_score",
                "tp", "fn", "fp", "num_gt", "num_tested", "recall", "precision",
            }
            feature_cols = [c for c in df_snapshot.columns if c not in meta_cols and c not in target_cols]
            missing_mean = (
                float(df_snapshot[feature_cols].isna().mean().mean())
                if feature_cols and len(df_snapshot) > 0
                else np.nan
            )
            all_null_features = sorted([c for c in feature_cols if df_snapshot[c].isna().all()]) if feature_cols else []
            duplicate_row_uid_count = (
                int(df_snapshot["row_uid"].duplicated().sum())
                if "row_uid" in df_snapshot.columns
                else int(df_snapshot.duplicated().sum())
            )
            target_null_fraction = {
                col: float(df_snapshot[col].isna().mean()) if col in df_snapshot.columns and len(df_snapshot) > 0 else 1.0
                for col in ["fmiss", "fpos", "accuracy"]
            }
            if {"accuracy", "fpos"}.issubset(df_snapshot.columns):
                if "matched_gt_unit_id" in df_snapshot.columns:
                    _unmatched_mask = df_snapshot["matched_gt_unit_id"].isna()
                else:
                    _unmatched_mask = df_snapshot["fmiss"].isna() if "fmiss" in df_snapshot.columns else pd.Series(False, index=df_snapshot.index)
                _degenerate_mask = (
                    df_snapshot["accuracy"].eq(0.0)
                    & df_snapshot["fpos"].eq(1.0)
                    & (_unmatched_mask | df_snapshot["fmiss"].eq(1.0))
                ) if "fmiss" in df_snapshot.columns else pd.Series(False, index=df_snapshot.index)
                degenerate_fraction = float(_degenerate_mask.mean()) if len(df_snapshot) > 0 else 0.0
            else:
                degenerate_fraction = np.nan

            major_issues = []
            enough_rows = len(df_snapshot) >= int(PHASE1_SANITY_MIN_ROWS)
            if (time.time() - start_time) >= 3600.0 and len(part_paths) == 0:
                major_issues.append("no_part_files_after_1h")
            if enough_rows:
                for _col, _frac in target_null_fraction.items():
                    if _frac >= 0.95:
                        major_issues.append(f"target_mostly_null:{_col}={_frac:.1%}")
                if pd.notna(missing_mean) and missing_mean > float(PHASE1_SANITY_MAX_MEAN_FEATURE_MISSING):
                    major_issues.append(
                        f"mean_feature_missing={missing_mean:.1%} (> {PHASE1_SANITY_MAX_MEAN_FEATURE_MISSING:.1%})"
                    )
                if len(all_null_features) > int(PHASE1_SANITY_MAX_ALL_NULL_FEATURES):
                    major_issues.append(
                        f"all_null_features={len(all_null_features)} (> {PHASE1_SANITY_MAX_ALL_NULL_FEATURES})"
                    )
                if duplicate_row_uid_count > 0:
                    major_issues.append(f"duplicate_row_uid_count={duplicate_row_uid_count}")
                if pd.notna(degenerate_fraction) and degenerate_fraction > float(PHASE1_SANITY_MAX_DEGENERATE_FRACTION):
                    major_issues.append(
                        f"degenerate_target_fraction={degenerate_fraction:.1%} (> {PHASE1_SANITY_MAX_DEGENERATE_FRACTION:.1%})"
                    )

            payload = {
                "generated_at": datetime.now().isoformat(),
                "reason": str(reason),
                "n_rows_snapshot": int(len(df_snapshot)),
                "n_parts_snapshot": int(len(part_paths)),
                "n_completed_keys": int(len(completed)),
                "n_selected_to_process": int(len(selected_to_process)),
                "snapshot_success_ratio": float(len(part_paths) / max(len(selected_to_process), 1)),
                "feature_count": int(len(feature_cols)),
                "mean_feature_missing": None if pd.isna(missing_mean) else float(missing_mean),
                "all_null_feature_count": int(len(all_null_features)),
                "all_null_feature_preview": all_null_features[:20],
                "duplicate_row_uid_count": int(duplicate_row_uid_count),
                "target_null_fraction": target_null_fraction,
                "degenerate_target_fraction": None if pd.isna(degenerate_fraction) else float(degenerate_fraction),
                "major_issues": list(major_issues),
                "train_parquet": str(TRAIN_PARQUET),
            }
            write_json(PHASE1_PERIODIC_CHECK_JSON, payload)
            with open(PHASE1_PERIODIC_CHECK_JSONL, "a", encoding="utf-8") as _fh:
                _fh.write(json.dumps(payload) + "\\n")
            state["last_sanity_check_at"] = payload["generated_at"]
            state["last_sanity_check_rows"] = int(len(df_snapshot))
            state["last_sanity_major_issues"] = list(major_issues)
            write_json(TRAIN_FEATURE_STATE_JSON, state)
            log_status(
                f"Phase 1 sanity [{reason}] rows={len(df_snapshot):,} "
                f"| mean_feature_missing={payload['mean_feature_missing']} "
                f"| all_null_features={len(all_null_features)} "
                f"| major_issues={len(major_issues)}"
            )
            if PHASE1_SANITY_STRICT and major_issues:
                raise RuntimeError(
                    "Phase 1 periodic sanity check failed. "
                    + " | ".join(major_issues)
                    + f". Inspect {PHASE1_PERIODIC_CHECK_JSON}."
                )
            return payload

        def _maybe_phase1_snapshot_and_check(force=False, reason="periodic"):
            _now = time.time()
            _need_snapshot = bool(force or (_now - phase1_timers["snapshot_at"] >= snapshot_interval_sec))
            _need_sanity = bool(force or (_now - phase1_timers["sanity_at"] >= sanity_interval_sec))
            if not _need_snapshot and not _need_sanity:
                return
            _part_paths = _phase1_collect_part_paths(selected_to_process)
            if not _part_paths:
                return
            _df_snapshot = _phase1_build_snapshot_df(_part_paths)
            if _need_snapshot:
                _phase1_atomic_write_snapshot(_df_snapshot, reason=reason)
                phase1_timers["snapshot_at"] = _now
            if _need_sanity:
                _phase1_snapshot_health(_df_snapshot, _part_paths, reason=reason)
                phase1_timers["sanity_at"] = _now
        """
    ).lstrip("\n"),
    "    ",
)


CELL12_HELPER_BAD = (
    'def _phase1_collect_part_paths(rows):\n'
    '    _maybe_phase1_snapshot_and_check(force=True, reason="loop_end")\n\n'
    '    part_paths = []\n'
)


CELL12_HELPER_FIXED = (
    'def _phase1_collect_part_paths(rows):\n'
    '    part_paths = []\n'
)


CELL12_LOOP_OLD = (
    '\n'
    '            if last_exc is not None:\n'
    '                state["failed_keys"][key] = str(last_exc)\n'
    '                state["heartbeat_at"] = datetime.now().isoformat()\n'
    '                state["last_status"] = "processed_failed"\n'
    '                write_json(TRAIN_FEATURE_STATE_JSON, state)\n'
    '                log_status(f"SKIP {key}: {last_exc}")\n\n'
    '            progress.advance(task)\n\n'
)


CELL12_LOOP_NEW = (
    '\n'
    '            if last_exc is not None:\n'
    '                state["failed_keys"][key] = str(last_exc)\n'
    '                state["heartbeat_at"] = datetime.now().isoformat()\n'
    '                state["last_status"] = "processed_failed"\n'
    '                write_json(TRAIN_FEATURE_STATE_JSON, state)\n'
    '                log_status(f"SKIP {key}: {last_exc}")\n\n'
    '            progress.advance(task)\n'
    '            _maybe_phase1_snapshot_and_check(reason=f"after:{key}")\n\n'
)


CELL12_RESUME_OLD = (
    '            if RESUME_FROM_CHECKPOINTS and part_path.exists():\n'
    '                if key not in completed:\n'
    '                    completed.add(key)\n'
    '                    state["completed_keys"] = sorted(completed)\n'
    '                progress.advance(task)\n'
    '                continue\n'
)


CELL12_RESUME_NEW = (
    '            if RESUME_FROM_CHECKPOINTS and part_path.exists():\n'
    '                if key not in completed:\n'
    '                    completed.add(key)\n'
    '                    state["completed_keys"] = sorted(completed)\n'
    '                progress.advance(task)\n'
    '                _maybe_phase1_snapshot_and_check(reason=f"resume:{key}")\n'
    '                continue\n'
)


CELL12_LOOP_END_OLD = '    part_paths = []\n    for item in selected_to_process:\n'
CELL12_LOOP_END_NEW = (
    '    _maybe_phase1_snapshot_and_check(force=True, reason="loop_end")\n\n'
    '    part_paths = []\n'
    '    for item in selected_to_process:\n'
)


def patch_cell5(source: str) -> str:
    out = source
    out = replace_once_or_skip(out, CELL5_PATHS_OLD, CELL5_PATHS_NEW)
    out = replace_once_or_skip(out, CELL5_POLICY_OLD, CELL5_POLICY_NEW)
    out = replace_once_or_skip(out, CELL5_PRINT_OLD, CELL5_PRINT_NEW)
    out = replace_once_or_skip(out, CELL5_PRINT2_OLD, CELL5_PRINT2_NEW)
    return out


def patch_cell12(source: str) -> str:
    out = source
    out = replace_once_or_skip(out, CELL12_REBUILD_OLD, CELL12_REBUILD_NEW)
    out = replace_once_or_skip(out, CELL12_SKIP_OLD, CELL12_SKIP_NEW)
    if CELL12_HELPER_INSERTION not in out:
        out = replace_once(out, CELL12_HELPER_ANCHOR, CELL12_HELPER_ANCHOR + CELL12_HELPER_INSERTION)
    out = replace_once_or_skip(out, CELL12_RESUME_OLD, CELL12_RESUME_NEW)
    out = replace_once_or_skip(out, CELL12_LOOP_OLD, CELL12_LOOP_NEW)
    out = out.replace(CELL12_HELPER_BAD, CELL12_HELPER_FIXED, 1)
    out = replace_once_or_skip(out, CELL12_LOOP_END_OLD, CELL12_LOOP_END_NEW)
    return out


def main():
    base_text = subprocess.check_output(
        ["git", "show", f"HEAD:{NB_GIT_PATH}"],
        cwd=REPO_ROOT,
        text=True,
    )
    nb = nbformat.reads(base_text, as_version=4)
    nb.cells[5].source = patch_cell5(nb.cells[5].source)
    nb.cells[12].source = patch_cell12(nb.cells[12].source)
    nbformat.write(nb, NB_PATH)
    print(f"Patched notebook: {NB_PATH}")


if __name__ == "__main__":
    main()
