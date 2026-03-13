from pathlib import Path
from textwrap import dedent

import nbformat


NB_PATH = Path("/Users/paulruiz/Documents/Predicting_Good_Units/notebooks/Units_Spikeforest_Extraction_Colab.ipynb")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found for replacement:\n{old[:240]}")
    return text.replace(old, new, 1)


def patch_config_cell(src: str) -> str:
    src = replace_once(
        src,
        'KACHERY_CLOUD_DIR = PROTOTYPE_ROOT / ".kachery-cloud"\n',
        'KACHERY_CLOUD_DIR = None  # assigned after scratch paths are defined\n',
    )
    src = replace_once(
        src,
        'SORTER_SCRATCH_DIR = SCRATCH_ROOT / "sorters"\n',
        'SORTER_SCRATCH_DIR = SCRATCH_ROOT / "sorters"\n'
        'KACHERY_CLOUD_DIR = SCRATCH_ROOT / ".kachery-cloud"\n',
    )

    old_study_sets = dedent(
        """
        SPIKEFOREST_STUDY_SETS = [
            # ── Priority synthetic set for first-pass extraction ──────────────
            "SYNTH_JANELIA",
            # ── Hybrid (real recording + injected spikes) — needs SPIKEFOREST_EXTRA_URI_PAIRS
            # ⚠ leakage guard below excludes any recording matching TEST_RECORDING_SPECS
            "HYBRID_JANELIA",
            # ── Paired (real) ground-truth recordings ───────────────────────
            "PAIRED_BOYDEN",
            "PAIRED_CRCNS_HC1",
            "PAIRED_ENGLISH",
            "PAIRED_KAMPFF",
            "PAIRED_MEA64C_YGER",
            # ── Synthetic study sets (ground truth available) ────────────────
            # SYNTH_JANELIA:  60 recordings, ~50 units, 4–64 ch, NP-noise, drift variants
            # SYNTH_BIONET:   36 recordings, ~40 units, 60 ch, biophysically realistic
            "SYNTH_BIONET",
            # SYNTH_MEAREC_NEURONEX: 60 recordings, ~10 units, 32 ch, MEArec, varying SNR
            "SYNTH_MEAREC_NEURONEX",
            # SYNTH_MAGLAND:  80 recordings, 10 units, 4–8 ch, simple noise ("K10")
            "SYNTH_MAGLAND",
            # SYNTH_VISAPY:   6 recordings, ~10 units, 30 ch; KS/KS2 crash on these —
            #   per-key failures are already caught and logged to failed_keys, so safe.
            "SYNTH_VISAPY",
            # ── Low-yield synthetics (optional; comment out to save time) ────
            # SYNTH_MEAREC_TETRODE: 40 recordings, 4-ch tetrode — limited channel diversity
            # "SYNTH_MEAREC_TETRODE",
            # SYNTH_MONOTRODE: 1 recording, 1 channel (enabled to mirror examples SHA1 source list)
            "SYNTH_MONOTRODE",
        ]
        """
    ).strip()
    new_study_sets = dedent(
        """
        SPIKEFOREST_STUDY_SETS = [
            # Paired ground-truth recordings first.
            "PAIRED_BOYDEN",
            "PAIRED_CRCNS_HC1",
            "PAIRED_ENGLISH",
            "PAIRED_KAMPFF",
            "PAIRED_MEA64C_YGER",
            # Keep hybrid after paired; exclude synthetic study sets entirely.
            # ⚠ leakage guard below excludes any recording matching TEST_RECORDING_SPECS
            "HYBRID_JANELIA",
        ]
        """
    ).strip()
    src = replace_once(src, old_study_sets, new_study_sets)
    return src


def patch_phase0_cell(src: str) -> str:
    old_load_cached = dedent(
        """
        def _load_cached_manifest_df():
            if SPIKEFOREST_MANIFEST.exists():
                with open(SPIKEFOREST_MANIFEST) as f:
                    return pd.DataFrame(json.load(f))
            return pd.DataFrame()
        """
    ).strip()
    new_load_cached = dedent(
        """
        def _requested_study_set_order():
            return [str(ss) for ss in SPIKEFOREST_STUDY_SETS if ss]


        def _study_set_requested_rank(study_set):
            order = _requested_study_set_order()
            order_map = {ss: idx for idx, ss in enumerate(order)}
            return int(order_map.get(str(study_set or ""), len(order) + 999))


        def _load_cached_manifest_df():
            if not SPIKEFOREST_MANIFEST.exists():
                return pd.DataFrame()
            with open(SPIKEFOREST_MANIFEST) as f:
                df = pd.DataFrame(json.load(f))
            if df.empty or "study_set" not in df.columns:
                return df

            requested = _requested_study_set_order()
            cached_sets = sorted({str(v) for v in df["study_set"].dropna().astype(str)})
            missing_requested = [ss for ss in requested if ss not in cached_sets]
            extra_cached = [ss for ss in cached_sets if ss not in requested]

            if extra_cached:
                print(
                    "⚠ Filtering cached manifest to current study-set request; "
                    f"dropping cached study sets: {extra_cached}"
                )
            if missing_requested:
                print(
                    "⚠ Cached manifest does not fully cover the current study-set request; "
                    f"missing: {missing_requested}"
                )

            df = df.loc[df["study_set"].astype(str).isin(requested)].copy()
            if df.empty:
                return df

            df["_study_rank"] = df["study_set"].astype(str).map(_study_set_requested_rank)
            sort_cols = ["_study_rank"] + [c for c in ["study_name", "recording_name", "sorter_name"] if c in df.columns]
            df = (
                df.sort_values(sort_cols)
                .drop(columns="_study_rank")
                .reset_index(drop=True)
            )
            return df


        def _cached_manifest_covers_requested_study_sets(df):
            if df.empty or "study_set" not in df.columns:
                return False
            requested = set(_requested_study_set_order())
            cached = {str(v) for v in df["study_set"].dropna().astype(str)}
            return bool(requested) and requested.issubset(cached)
        """
    ).strip()
    src = replace_once(src, old_load_cached, new_load_cached)

    old_reuse_branch = dedent(
        """
        if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES:
            print("Using existing training parquet; SpikeForest enumeration is optional and skipped.")
            df_manifest = _load_cached_manifest_df()
            if not df_manifest.empty:
                selected = df_manifest.to_dict(orient="records")
            phase0_mode = "skipped_reuse_train"
        elif not SPIKEFOREST_AVAILABLE:
        """
    ).strip()
    new_reuse_branch = dedent(
        """
        if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES:
            df_manifest = _load_cached_manifest_df()
            if _cached_manifest_covers_requested_study_sets(df_manifest):
                print("Using existing training parquet; cached SpikeForest manifest filtered to the current study-set request.")
                selected = df_manifest.to_dict(orient="records")
                phase0_mode = "skipped_reuse_train"
            else:
                print("Cached manifest does not satisfy the current study-set request; re-enumerating SpikeForest sorting outputs.")
                selected, available_study_sets, phase0_counters, phase0_source_summary, phase0_study_set_source_map = _enumerate_sf_manifest_rows()
                write_json(SPIKEFOREST_MANIFEST, selected)
                df_manifest = pd.DataFrame(selected)
                phase0_mode = "re_enumerated_after_cached_manifest_refresh"
        elif not SPIKEFOREST_AVAILABLE:
        """
    ).strip()
    src = replace_once(src, old_reuse_branch, new_reuse_branch)
    return src


def patch_phase1_cell(src: str) -> str:
    old_priority_block = dedent(
        """
        def _round_robin_by_sorter(rows):
            if not rows:
                return []
            from collections import deque

            by_sorter = {}
            for row in rows:
                sorter = str(row.get("sorter_name", ""))
                by_sorter.setdefault(sorter, []).append(row)
            queues = {
                sorter: deque(sorted(items, key=lambda r: sf_output_key(r["study_set"], r["study_name"], r["recording_name"], r["sorter_name"])))
                for sorter, items in by_sorter.items()
            }
            ordered = []
            sorter_order = sorted(queues.keys())
            while True:
                progressed = False
                for sorter in sorter_order:
                    q = queues.get(sorter)
                    if q:
                        ordered.append(q.popleft())
                        progressed = True
                if not progressed:
                    break
            return ordered


        def _prioritize_phase1_rows(rows):
            priority_set = str(PHASE1_PRIORITY_STUDY_SET or "").strip()
            rows_in = list(rows)
            if not rows_in:
                return []

            if priority_set:
                priority_rows = [r for r in rows_in if str(r.get("study_set", "")) == priority_set]
                other_rows = [r for r in rows_in if str(r.get("study_set", "")) != priority_set]
            else:
                priority_rows = []
                other_rows = rows_in

            if PHASE1_ROUND_ROBIN_SORTERS:
                priority_rows = _round_robin_by_sorter(priority_rows)
                other_rows = _round_robin_by_sorter(other_rows)
            else:
                priority_rows = sorted(priority_rows, key=lambda r: sf_output_key(r["study_set"], r["study_name"], r["recording_name"], r["sorter_name"]))
                other_rows = sorted(other_rows, key=lambda r: sf_output_key(r["study_set"], r["study_name"], r["recording_name"], r["sorter_name"]))

            ordered = priority_rows + other_rows
            if priority_set:
                log_status(
                    "Phase 1 ordering: "
                    f"{len(priority_rows)} rows from {priority_set} first, then {len(other_rows)} rows from other study sets."
                )
            return ordered
        """
    ).strip()
    new_priority_block = dedent(
        """
        def _sort_rows_by_requested_study_order(rows):
            return sorted(
                rows,
                key=lambda r: (
                    _study_set_requested_rank(r.get("study_set", "")),
                    sf_output_key(r["study_set"], r["study_name"], r["recording_name"], r["sorter_name"]),
                ),
            )


        def _round_robin_by_sorter(rows):
            if not rows:
                return []
            from collections import deque

            by_sorter = {}
            for row in _sort_rows_by_requested_study_order(rows):
                sorter = str(row.get("sorter_name", ""))
                by_sorter.setdefault(sorter, []).append(row)
            queues = {
                sorter: deque(_sort_rows_by_requested_study_order(items))
                for sorter, items in by_sorter.items()
            }
            ordered = []
            sorter_order = sorted(queues.keys())
            while True:
                progressed = False
                for sorter in sorter_order:
                    q = queues.get(sorter)
                    if q:
                        ordered.append(q.popleft())
                        progressed = True
                if not progressed:
                    break
            return ordered


        def _prioritize_phase1_rows(rows):
            rows_in = _sort_rows_by_requested_study_order(list(rows))
            if not rows_in:
                return []

            ordered = []
            consumed = set()
            priority_set = str(PHASE1_PRIORITY_STUDY_SET or "").strip()
            requested_sets = list(_requested_study_set_order())

            def _row_key(_row):
                return sf_output_key(_row["study_set"], _row["study_name"], _row["recording_name"], _row["sorter_name"])

            def _emit_bucket(bucket_rows):
                bucket_rows = _sort_rows_by_requested_study_order(bucket_rows)
                if PHASE1_ROUND_ROBIN_SORTERS:
                    bucket_rows = _round_robin_by_sorter(bucket_rows)
                return bucket_rows

            if priority_set:
                priority_rows = [r for r in rows_in if str(r.get("study_set", "")) == priority_set]
                for row in _emit_bucket(priority_rows):
                    ordered.append(row)
                    consumed.add(_row_key(row))
                requested_sets = [ss for ss in requested_sets if ss != priority_set]
                log_status(
                    "Phase 1 ordering: "
                    f"explicit priority study set {priority_set} first, then remaining requested study sets."
                )
            else:
                log_status(
                    "Phase 1 ordering by requested study-set list: "
                    + " -> ".join(requested_sets)
                )

            for study_set in requested_sets:
                bucket = [r for r in rows_in if str(r.get("study_set", "")) == study_set and _row_key(r) not in consumed]
                for row in _emit_bucket(bucket):
                    ordered.append(row)
                    consumed.add(_row_key(row))

            remaining = [r for r in rows_in if _row_key(r) not in consumed]
            ordered.extend(_emit_bucket(remaining))
            return ordered
        """
    ).strip()
    src = replace_once(src, old_priority_block, new_priority_block)

    old_reuse_complete = dedent(
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
    new_reuse_complete = dedent(
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
        _extra_completed_keys = sorted(_existing_completed_keys - set(_current_phase1_keys))
        _reuse_train_complete = bool(
            _existing_train_state.get("complete", False)
            and not _missing_current_phase1_keys
            and not _extra_completed_keys
        )
        if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES and not _reuse_train_complete:
            reason_bits = []
            if _missing_current_phase1_keys:
                reason_bits.append("state is missing keys from the current selection")
            if _extra_completed_keys:
                reason_bits.append("state contains completed keys outside the current selection")
            reason_text = "; ".join(reason_bits) if reason_bits else "state is incomplete"
            log_status(
                f"Partial training parquet found at {TRAIN_PARQUET}; "
                f"{reason_text}, so resume will continue from checkpointed part files."
            )
        if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES and _reuse_train_complete:
            df_train = safe_read_parquet(TRAIN_PARQUET)
            log_status(f"Phase 1 skipped. Reusing training features from {TRAIN_PARQUET}")
        elif not SPIKEFOREST_AVAILABLE:
        """
    ).strip()
    src = replace_once(src, old_reuse_complete, new_reuse_complete)
    return src


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)

    patched = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        original = src
        if "SPIKEFOREST_STUDY_SETS = [" in src and "KACHERY_CLOUD_DIR" in src:
            src = patch_config_cell(src)
        if "def _load_cached_manifest_df():" in src and "def _enumerate_sf_manifest_rows():" in src:
            src = patch_phase0_cell(src)
        if "def _prioritize_phase1_rows(rows):" in src and "_existing_train_state = read_json" in src:
            src = patch_phase1_cell(src)
        if src != original:
            cell["source"] = src
            patched += 1

    if patched == 0:
        raise RuntimeError("No notebook cells were patched.")

    nbformat.write(nb, NB_PATH)
    print(f"Patched {patched} notebook cell(s) in {NB_PATH}")


if __name__ == "__main__":
    main()
