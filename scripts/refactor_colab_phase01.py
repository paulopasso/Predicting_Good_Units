from pathlib import Path
from textwrap import dedent

import nbformat


NB_PATH = Path("/Users/paulruiz/Documents/Predicting_Good_Units/Spike_UNIT_Quality.ipynb")


CELL11_SOURCE = dedent(
    """
    # ── Phase 0A: Source contracts and normalization ─────────────────────────────
    # This phase builds a contract-first manifest from SHA1 payloads with strict
    # schema normalization. It writes:
    #   - spikeforest_manifest_full.json     (joined rows before final filtering)
    #   - spikeforest_manifest_selected.json (runnable rows only)
    #   - phase0_summary.json                (source + filter diagnostics)
    # Safe re-run: deterministic and idempotent for same source URIs.

    PHASE0_SCHEMA_VERSION = "phase0_v2_contract_20260307"
    SPIKEFOREST_MANIFEST_FULL = (
        SPIKEFOREST_MANIFEST_FULL_JSON
        if "SPIKEFOREST_MANIFEST_FULL_JSON" in globals()
        else MANIFEST_DIR / "spikeforest_manifest_full.json"
    )
    SPIKEFOREST_MANIFEST_SELECTED = (
        SPIKEFOREST_MANIFEST_SELECTED_JSON
        if "SPIKEFOREST_MANIFEST_SELECTED_JSON" in globals()
        else MANIFEST_DIR / "spikeforest_manifest_selected.json"
    )
    PHASE0_SUMMARY_PATH = (
        PHASE0_SUMMARY_JSON if "PHASE0_SUMMARY_JSON" in globals() else MANIFEST_DIR / "phase0_summary.json"
    )
    SPIKEFOREST_MANIFEST = SPIKEFOREST_MANIFEST_SELECTED

    _KACHERY_CONNECT_TIMEOUT = 60
    _KACHERY_READ_TIMEOUT = 300
    apply_kachery_requests_timeout_patch(
        connect_timeout=_KACHERY_CONNECT_TIMEOUT,
        read_timeout=_KACHERY_READ_TIMEOUT,
    )


    def _phase0_local_bootstrap_kachery_client_keys():
        try:
            import kachery_cloud._client_keys as ck
            ck._get_client_keys_hex(generate_if_missing=True)
            pub, priv = ck._get_client_keys_hex(generate_if_missing=False)
            return pub is not None and priv is not None
        except Exception:
            return False


    def _phase0_as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y"}
        return False


    def _phase0_to_int(value):
        try:
            return int(value)
        except Exception:
            return None


    def _phase0_to_float(value):
        try:
            return float(value)
        except Exception:
            return np.nan


    def _phase0_get(d, *keys, default=None):
        if not isinstance(d, dict):
            return default
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
        return default


    def _phase0_extract_list(payload, preferred_keys):
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return []

        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        list_values = [v for v in payload.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        if len(list_values) > 1:
            best = max(list_values, key=lambda x: len(x))
            return best
        return []


    def _phase0_sorting_object_supported(sorting_object):
        if sorting_object is None:
            return False
        if not isinstance(sorting_object, dict):
            return False
        if "firings" in sorting_object:
            return True
        data = sorting_object.get("data", {})
        if not isinstance(data, dict):
            return False
        if "firings" not in data and "firingsUri" not in data:
            return False
        sorting_format = str(sorting_object.get("sorting_format", "mda")).lower()
        return sorting_format in {"mda", "npz"}


    def _phase0_study_priority(study_set):
        study_set = str(study_set or "")
        synthetic = {
            "SYNTH_MONOTRODE",
            "SYNTH_MAGLAND",
            "SYNTH_JANELIA",
            "SYNTH_BIONET",
            "SYNTH_MEAREC_NEURONEX",
            "SYNTH_VISAPY",
            "SYNTH_MEAREC_TETRODE",
        }
        paired = {
            "PAIRED_ENGLISH",
            "PAIRED_MEA64C_YGER",
            "PAIRED_BOYDEN",
            "PAIRED_KAMPFF",
            "PAIRED_CRCNS_HC1",
        }
        if study_set in synthetic:
            return 3
        if study_set in paired:
            return 2
        if study_set == "HYBRID_JANELIA":
            return 1
        return 0


    def _phase0_historical_sorter_success():
        state = read_json(TRAIN_FEATURE_STATE_JSON, default={})
        rows_state = state.get("rows", {})
        if not isinstance(rows_state, dict):
            return {}
        counts = {}
        for key, payload in rows_state.items():
            if not isinstance(payload, dict):
                continue
            sorter = str(payload.get("sorter_name") or "")
            if not sorter and isinstance(key, str):
                parts = key.split("/")
                if len(parts) >= 4:
                    sorter = parts[3]
            if not sorter:
                continue
            c = counts.setdefault(sorter, {"ok": 0, "total": 0})
            c["total"] += 1
            if str(payload.get("status")) == "ready":
                c["ok"] += 1
        out = {}
        for sorter, c in counts.items():
            out[sorter] = float(c["ok"] / max(c["total"], 1))
        return out


    def _phase0_priority_score(row, sorter_stability_map):
        true_units = row.get("recording_num_true_units")
        if true_units is None:
            true_units = 0
        try:
            true_units = int(true_units)
        except Exception:
            true_units = 0
        study_priority = _phase0_study_priority(row.get("study_set"))
        sorter = str(row.get("sorter_name", ""))
        stability = float(sorter_stability_map.get(sorter, 0.0))
        cpu_time = _phase0_to_float(row.get("cpu_time_sec", np.nan))
        cpu_bonus = 0.0 if not np.isfinite(cpu_time) else max(0.0, 1.0 - min(cpu_time, 600.0) / 600.0)
        return float(true_units * 1000.0 + study_priority * 100.0 + stability * 10.0 + cpu_bonus)


    def _phase0_build_sources():
        default_sorting_uri = SPIKEFOREST_SORTING_OUTPUTS_URI or SPIKEFOREST_DEFAULT_SORTING_OUTPUTS_URI
        default_recordings_uri = SPIKEFOREST_RECORDINGS_URI or SPIKEFOREST_DEFAULT_RECORDINGS_URI
        sources = [
            {
                "name": "default",
                "sorting_outputs_uri": str(default_sorting_uri),
                "recordings_uri": str(default_recordings_uri),
            }
        ]
        for idx, pair in enumerate(SPIKEFOREST_EXTRA_URI_PAIRS or []):
            if not pair:
                continue
            out_uri = str(pair[0]) if len(pair) >= 1 else ""
            rec_uri = str(pair[1]) if len(pair) >= 2 else str(default_recordings_uri)
            if not out_uri:
                continue
            sources.append(
                {
                    "name": f"extra_{idx}",
                    "sorting_outputs_uri": out_uri,
                    "recordings_uri": rec_uri,
                }
            )
        dedup = []
        seen = set()
        for src in sources:
            key = (src["sorting_outputs_uri"], src["recordings_uri"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(src)
        return dedup


    def _phase0_load_json_uri(uri):
        if kcl is None:
            raise RuntimeError("kachery_cloud is not available in this runtime.")
        payload = kcl.load_json(uri)
        if payload is None:
            raise RuntimeError(f"Failed to load JSON from {uri}")
        return payload


    def _phase0_normalize_recordings(recording_payload, source_recordings_uri):
        rows = []
        records = _phase0_extract_list(
            recording_payload,
            preferred_keys=["recordings", "items", "data"],
        )
        for rec in records:
            if not isinstance(rec, dict):
                continue
            study_name = str(_phase0_get(rec, "studyName", "study_name", default="") or "")
            recording_name = str(
                _phase0_get(rec, "recordingName", "recording_name", "name", default="") or ""
            )
            if not study_name or not recording_name:
                continue

            study_set = str(
                _phase0_get(
                    rec,
                    "studySetName",
                    "study_set_name",
                    "studySet",
                    "study_set",
                    default="",
                )
                or ""
            )
            recording_object = _phase0_get(rec, "recordingObject", "recording_object", default={})
            if not isinstance(recording_object, dict):
                recording_object = {}
            params = recording_object.get("params", {})
            if not isinstance(params, dict):
                params = {}
            geom = recording_object.get("geom", [])
            if not isinstance(geom, list):
                geom = []
            raw_uri = _phase0_get(rec, "recordingRawUri", default=None)
            if raw_uri is None:
                raw_uri = recording_object.get("raw")

            n_channels = _phase0_get(rec, "numChannels", "num_channels", default=None)
            if n_channels is None and len(geom) > 0:
                n_channels = len(geom)
            n_channels = _phase0_to_int(n_channels)

            fs = _phase0_get(rec, "sampleRateHz", "sample_rate_hz", "sample_rate", default=None)
            if fs is None:
                fs = _phase0_get(params, "samplerate", "sampling_frequency", default=None)
            fs = _phase0_to_float(fs)

            duration_sec = _phase0_to_float(
                _phase0_get(rec, "durationSec", "duration_sec", default=np.nan)
            )
            if not np.isfinite(duration_sec):
                duration_sec = _phase0_to_float(_phase0_get(params, "duration_sec", default=np.nan))

            rows.append(
                {
                    "study_name": study_name,
                    "recording_name": recording_name,
                    "study_set": study_set,
                    "recording_num_true_units": _phase0_to_int(
                        _phase0_get(rec, "numTrueUnits", "num_true_units", default=None)
                    ),
                    "recording_num_channels": n_channels,
                    "recording_sample_rate": fs,
                    "recording_duration_sec": duration_sec,
                    "recording_raw_uri": str(raw_uri) if raw_uri else None,
                    "source_recordings_uri": str(source_recordings_uri),
                }
            )
        return rows


    def _phase0_normalize_outputs(output_payload, source_sorting_outputs_uri):
        rows = []
        outputs = _phase0_extract_list(
            output_payload,
            preferred_keys=["sorting_outputs", "sortingOutputs", "items", "data", "outputs"],
        )
        for out in outputs:
            if not isinstance(out, dict):
                continue
            study_name = str(_phase0_get(out, "studyName", "study_name", default="") or "")
            recording_name = str(
                _phase0_get(out, "recordingName", "recording_name", default="") or ""
            )
            sorter_name = str(_phase0_get(out, "sorterName", "sorter_name", default="") or "")
            if not study_name or not recording_name or not sorter_name:
                continue
            sorting_object = _phase0_get(out, "sortingObject", "sorting_object", default=None)
            timed_out = _phase0_as_bool(_phase0_get(out, "timedOut", "timed_out", default=False))
            return_code = _phase0_to_int(_phase0_get(out, "returnCode", "return_code", default=None))
            cpu_time_sec = _phase0_to_float(_phase0_get(out, "cpuTimeSec", "cpu_time_sec", default=np.nan))
            rows.append(
                {
                    "study_name": study_name,
                    "recording_name": recording_name,
                    "sorter_name": sorter_name,
                    "source_sorting_outputs_uri": str(source_sorting_outputs_uri),
                    "cpu_time_sec": cpu_time_sec,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "has_sorting_object": sorting_object is not None,
                    "sorting_object_supported": _phase0_sorting_object_supported(sorting_object),
                    "sorting_object": sorting_object,
                }
            )
        return rows


    def _phase0_contract_row(row):
        required = [
            "study_set",
            "study_name",
            "recording_name",
            "sorter_name",
            "source_sorting_outputs_uri",
            "source_recordings_uri",
            "phase0_schema_version",
        ]
        missing = [k for k in required if not row.get(k)]
        return missing


    def _phase0_run_enumeration():
        sources = _phase0_build_sources()
        sorter_stability = _phase0_historical_sorter_success()

        full_rows = []
        selected_rows = []
        recording_lookup_global = {}
        phase0_counts = {
            "sources": len(sources),
            "recording_rows": 0,
            "output_rows": 0,
            "join_missing": 0,
            "skipped_filter_study_set": 0,
            "skipped_leakage": 0,
            "skipped_failed_output": 0,
            "selected_rows": 0,
            "selected_rows_dedup": 0,
        }
        source_summary = {}
        study_set_source_map = {}
        available_study_sets = {}

        for src in sources:
            out_uri = src["sorting_outputs_uri"]
            rec_uri = src["recordings_uri"]
            summary_row = {
                "sorting_outputs_uri": out_uri,
                "recordings_uri": rec_uri,
                "n_recordings": 0,
                "n_outputs": 0,
                "n_join_miss": 0,
                "n_selected": 0,
            }
            source_summary[out_uri] = summary_row

            rec_payload = _phase0_load_json_uri(rec_uri)
            out_payload = _phase0_load_json_uri(out_uri)
            rec_rows = _phase0_normalize_recordings(rec_payload, rec_uri)
            out_rows = _phase0_normalize_outputs(out_payload, out_uri)

            summary_row["n_recordings"] = len(rec_rows)
            summary_row["n_outputs"] = len(out_rows)
            phase0_counts["recording_rows"] += len(rec_rows)
            phase0_counts["output_rows"] += len(out_rows)

            local_lookup = {}
            for rr in rec_rows:
                key = (rr["study_name"], rr["recording_name"])
                local_lookup[key] = rr
                recording_lookup_global[key] = rr
                ss = rr.get("study_set")
                if ss:
                    available_study_sets[ss] = available_study_sets.get(ss, 0) + 1

            for out in out_rows:
                key = (out["study_name"], out["recording_name"])
                rec = local_lookup.get(key) or recording_lookup_global.get(key)
                row = dict(out)
                row["source_recordings_uri"] = rec_uri
                row["phase0_schema_version"] = PHASE0_SCHEMA_VERSION
                row["join_key"] = f"{out['study_name']}::{out['recording_name']}"
                row["join_found"] = rec is not None
                row["study_set"] = rec.get("study_set") if rec else None
                row["recording_num_true_units"] = rec.get("recording_num_true_units") if rec else None
                row["recording_num_channels"] = rec.get("recording_num_channels") if rec else None
                row["recording_sample_rate"] = rec.get("recording_sample_rate") if rec else np.nan
                row["recording_duration_sec"] = rec.get("recording_duration_sec") if rec else np.nan
                row["recording_raw_uri"] = rec.get("recording_raw_uri") if rec else None
                row["skip_reason"] = None
                row["selected"] = False

                if rec is None:
                    phase0_counts["join_missing"] += 1
                    summary_row["n_join_miss"] += 1
                    row["skip_reason"] = "join_missing_recording"
                    full_rows.append(row)
                    continue

                if row["study_set"] not in SPIKEFOREST_STUDY_SETS:
                    phase0_counts["skipped_filter_study_set"] += 1
                    row["skip_reason"] = "filtered_study_set"
                    full_rows.append(row)
                    continue

                if row["recording_name"] in (TRAIN_EXCLUDE_RECORDING_NAMES or set()):
                    phase0_counts["skipped_leakage"] += 1
                    row["skip_reason"] = "leakage_excluded_recording"
                    full_rows.append(row)
                    continue

                timed_out = _phase0_as_bool(row.get("timed_out"))
                rc = row.get("return_code")
                has_obj = bool(row.get("has_sorting_object"))
                obj_ok = bool(row.get("sorting_object_supported"))
                if SKIP_FAILED_SORTING_OUTPUTS and (
                    timed_out or (rc is not None and rc != 0) or (not has_obj) or (not obj_ok)
                ):
                    phase0_counts["skipped_failed_output"] += 1
                    row["skip_reason"] = "failed_sorting_output"
                    full_rows.append(row)
                    continue

                row["priority_score"] = _phase0_priority_score(row, sorter_stability)
                row["sorter_stability"] = float(sorter_stability.get(row["sorter_name"], 0.0))
                row["selected"] = True
                row["skip_reason"] = None
                summary_row["n_selected"] += 1
                selected_rows.append(row)
                full_rows.append(row)
                study_set_source_map.setdefault(row["study_set"], set()).add(str(rec_uri))

        dedup = []
        seen = set()
        for row in selected_rows:
            key = sf_output_key(
                row["study_set"],
                row["study_name"],
                row["recording_name"],
                row["sorter_name"],
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(row)

        dedup = sorted(
            dedup,
            key=lambda r: (
                -int(r.get("recording_num_true_units") or 0),
                -int(_phase0_study_priority(r.get("study_set"))),
                -float(r.get("sorter_stability", 0.0)),
                -float(r.get("priority_score", 0.0)),
                str(r.get("study_set", "")),
                str(r.get("study_name", "")),
                str(r.get("recording_name", "")),
                str(r.get("sorter_name", "")),
            ),
        )

        phase0_counts["selected_rows"] = len(selected_rows)
        phase0_counts["selected_rows_dedup"] = len(dedup)
        study_set_source_map = {k: sorted(list(v)) for k, v in study_set_source_map.items()}

        summary = {
            "generated_at": datetime.now().isoformat(),
            "schema_version": PHASE0_SCHEMA_VERSION,
            "phase0_counts": phase0_counts,
            "sources": source_summary,
            "available_study_sets": dict(sorted(available_study_sets.items(), key=lambda x: (-x[1], x[0]))),
            "study_set_to_recordings_uri": study_set_source_map,
            "requested_study_sets": list(SPIKEFOREST_STUDY_SETS),
            "manifest_full_path": str(SPIKEFOREST_MANIFEST_FULL),
            "manifest_selected_path": str(SPIKEFOREST_MANIFEST_SELECTED),
        }
        return full_rows, dedup, summary


    # ── Phase 0B/C execution + persistence ──────────────────────────────────────
    selected = []
    phase0_mode = "none"
    df_manifest = pd.DataFrame()
    phase0_summary = {}

    if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES:
        phase0_mode = "skipped_reuse_train"
        if SPIKEFOREST_MANIFEST_SELECTED.exists():
            cached = read_json(SPIKEFOREST_MANIFEST_SELECTED, default=[])
            if isinstance(cached, list):
                selected = cached
                df_manifest = pd.DataFrame(selected)
        print("Using existing training parquet; Phase 0 skipped.")
    else:
        try:
            full_rows, selected_rows, phase0_summary = _phase0_run_enumeration()
            write_json(SPIKEFOREST_MANIFEST_FULL, full_rows)
            write_json(SPIKEFOREST_MANIFEST_SELECTED, selected_rows)
            write_json(SPIKEFOREST_MANIFEST, selected_rows)
            write_json(PHASE0_SUMMARY_PATH, phase0_summary)
            write_json(
                DATA_CONTRACTS_JSON,
                {
                    "generated_at": datetime.now().isoformat(),
                    "schema_version": PHASE0_SCHEMA_VERSION,
                    "study_set_to_recordings_uri": phase0_summary.get("study_set_to_recordings_uri", {}),
                    "sources": phase0_summary.get("sources", {}),
                    "phase0_counts": phase0_summary.get("phase0_counts", {}),
                    "manifest_selected_path": str(SPIKEFOREST_MANIFEST_SELECTED),
                },
            )
            selected = selected_rows
            df_manifest = pd.DataFrame(selected)
            phase0_mode = "enumerated"
        except Exception as exc:
            msg = str(exc).lower()
            key_error = "client keys" in msg or "client key" in msg
            if key_error:
                print("⚠ Phase 0 requires kachery client keys. Attempting bootstrap...")
                ok = _phase0_local_bootstrap_kachery_client_keys()
                if ok:
                    full_rows, selected_rows, phase0_summary = _phase0_run_enumeration()
                    write_json(SPIKEFOREST_MANIFEST_FULL, full_rows)
                    write_json(SPIKEFOREST_MANIFEST_SELECTED, selected_rows)
                    write_json(SPIKEFOREST_MANIFEST, selected_rows)
                    write_json(PHASE0_SUMMARY_PATH, phase0_summary)
                    selected = selected_rows
                    df_manifest = pd.DataFrame(selected)
                    phase0_mode = "enumerated_after_key_bootstrap"
                else:
                    raise RuntimeError("Phase 0 bootstrap for kachery keys failed.") from exc
            else:
                raise

    # ── Manifest contract checks ────────────────────────────────────────────────
    if isinstance(selected, list) and len(selected) > 0:
        bad_rows = []
        for i, row in enumerate(selected):
            missing = _phase0_contract_row(row)
            if missing:
                bad_rows.append({"index": i, "missing": missing})
        if bad_rows:
            preview = bad_rows[:5]
            raise RuntimeError(f"Phase 0 manifest contract failed for {len(bad_rows)} rows: {preview}")

    if df_manifest.empty:
        requested = sorted(list(SPIKEFOREST_STUDY_SETS))
        print(f"✓ Phase 0 ({phase0_mode}) did not produce a manifest. requested={requested}")
    else:
        required_cols = [
            "study_set",
            "study_name",
            "recording_name",
            "sorter_name",
            "source_sorting_outputs_uri",
            "source_recordings_uri",
            "phase0_schema_version",
            "priority_score",
        ]
        missing_cols = [c for c in required_cols if c not in df_manifest.columns]
        if missing_cols:
            raise RuntimeError(f"Phase 0 selected manifest missing columns: {missing_cols}")

        if df_manifest[["study_set", "study_name", "recording_name", "sorter_name"]].isna().any().any():
            raise RuntimeError("Phase 0 selected manifest has null identity keys.")

        study_set_counts = (
            df_manifest.groupby("study_set").size().sort_values(ascending=False).head(8).to_dict()
        )
        print("Enumerating SpikeForest sorting outputs from SHA1 manifests...")
        print(f"Available study sets (top): {list(study_set_counts.items())}")
        if phase0_summary:
            print(f"Phase 0 counters: {phase0_summary.get('phase0_counts', {})}")

        table = Table(title="SpikeForest Manifest")
        table.add_column("Study Set")
        table.add_column("Sorter")
        table.add_column("N recordings", justify="right")
        for (ss, sorter), grp in df_manifest.groupby(["study_set", "sorter_name"], dropna=False):
            table.add_row(str(ss), str(sorter), str(int(len(grp))))
        console.print(table)
    """
).strip()


CELL12_SOURCE = dedent(
    """
    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1A — Runtime guards + paths
    # ══════════════════════════════════════════════════════════════════
    # This phase performs timeout-safe sequential extraction and writes:
    #   - train_feature_state.json
    #   - train_failures.parquet
    #   - train_success_manifest.parquet
    #   - extract_progress.jsonl
    #   - features_train.parquet
    #   - data_audit_phase1.json
    #   - feature_coverage_by_block.json
    # Safe rerun behavior:
    #   - completed rows with existing part files are skipped when RESUME_FROM_CHECKPOINTS=True
    #   - full rebuild when FORCE_REBUILD_TRAIN_FEATURES=True

    import multiprocessing as mp

    TRAIN_FAILURES_PARQUET = (
        TRAIN_FAILURES_PARQUET if "TRAIN_FAILURES_PARQUET" in globals() else AUDIT_DIR / "train_failures.parquet"
    )
    TRAIN_SUCCESS_MANIFEST_PARQUET = (
        TRAIN_SUCCESS_MANIFEST_PARQUET
        if "TRAIN_SUCCESS_MANIFEST_PARQUET" in globals()
        else AUDIT_DIR / "train_success_manifest.parquet"
    )
    EXTRACT_PROGRESS_JSONL = (
        EXTRACT_PROGRESS_JSONL if "EXTRACT_PROGRESS_JSONL" in globals() else LOG_DIR / "extract_progress.jsonl"
    )
    DATA_AUDIT_PHASE1_JSON = (
        DATA_AUDIT_PHASE1_JSON if "DATA_AUDIT_PHASE1_JSON" in globals() else AUDIT_DIR / "data_audit_phase1.json"
    )
    FEATURE_COVERAGE_BY_BLOCK_JSON = (
        FEATURE_COVERAGE_BY_BLOCK_JSON
        if "FEATURE_COVERAGE_BY_BLOCK_JSON" in globals()
        else AUDIT_DIR / "feature_coverage_by_block.json"
    )
    PHASE1_ROW_TIMEOUT_SEC = (
        PHASE1_ROW_TIMEOUT_SEC if "PHASE1_ROW_TIMEOUT_SEC" in globals() else 240
    )
    PHASE1_ROW_TIMEOUT_HYBRID_SEC = (
        PHASE1_ROW_TIMEOUT_HYBRID_SEC if "PHASE1_ROW_TIMEOUT_HYBRID_SEC" in globals() else 420
    )
    PHASE1_MAX_RETRIES = PHASE1_MAX_RETRIES if "PHASE1_MAX_RETRIES" in globals() else 3

    _KACHERY_CONNECT_TIMEOUT = 60
    _KACHERY_READ_TIMEOUT = 300
    _KACHERY_MAX_RETRIES = max(1, int(PHASE1_MAX_RETRIES))
    apply_kachery_requests_timeout_patch(
        connect_timeout=_KACHERY_CONNECT_TIMEOUT,
        read_timeout=_KACHERY_READ_TIMEOUT,
    )
    log_status(
        "kachery requests.get patched with download timeout "
        f"(connect={_KACHERY_CONNECT_TIMEOUT}s, read={_KACHERY_READ_TIMEOUT}s, "
        f"max_retries={_KACHERY_MAX_RETRIES})"
    )

    try:
        import numba as _numba
        _numba_version = str(_numba.__version__)
    except Exception as exc:
        raise RuntimeError(
            "numba is required for Phase 1 extraction. "
            "Re-run install cell (Cell 0), restart runtime, then continue."
        ) from exc

    if Version is not None:
        if Version(_numba_version) < Version("0.59.0"):
            raise RuntimeError(f"numba {_numba_version} detected, but Phase 1 requires >=0.59.0")

    BLOCK_A_DEFAULTS = {
        "snr": np.nan,
        "peak_to_trough_uv": np.nan,
        "half_width_ms": np.nan,
        "repolarization_slope": np.nan,
        "pre_trough_peak_uv": np.nan,
        "post_trough_peak_uv": np.nan,
        "trough_time_ms": np.nan,
        "wf_energy": np.nan,
        "wf_ptp_ratio": np.nan,
        "wf_asymmetry": np.nan,
        "wf_prepeak_ratio": np.nan,
        "wf_rebound_ratio": np.nan,
        "wf_peak_trough_ms": np.nan,
        "wf_prepeak_to_trough_ms": np.nan,
        "wf_trough_to_peak_ms": np.nan,
        "wf_zero_cross_pre_ms": np.nan,
        "wf_zero_cross_post_ms": np.nan,
        "wf_trough_width_25_ms": np.nan,
        "wf_trough_width_50_ms": np.nan,
        "wf_trough_width_75_ms": np.nan,
        "wf_neg_area_uv_ms": np.nan,
        "wf_pos_area_uv_ms": np.nan,
        "wf_pos_neg_area_ratio": np.nan,
        "wf_line_length_norm": np.nan,
        "wf_curvature_norm": np.nan,
        "wf_trough_sharpness": np.nan,
        "wf_pre_trough_slope": np.nan,
        "wf_post_peak_slope": np.nan,
        "template_norm": np.nan,
        "n_active_channels": np.nan,
        "spread_um": np.nan,
        "center_of_mass_um": np.nan,
        "spread_weighted_um": np.nan,
        "peak_channel_depth_um": np.nan,
        "n_channels": np.nan,
    }
    for _i in range(30):
        BLOCK_A_DEFAULTS[f"wf_bin_{_i:02d}"] = np.nan
    BLOCK_B_DEFAULTS = {
        "amp_mean": np.nan,
        "amp_std": np.nan,
        "amp_cv": np.nan,
        "amp_p5": np.nan,
        "amp_p50": np.nan,
        "amp_p95": np.nan,
        "amp_iqr": np.nan,
        "amp_skew": np.nan,
        "amp_kurtosis": np.nan,
        "amp_drift_slope": np.nan,
        "amp_drift_r2": np.nan,
        "amp_bimodality": np.nan,
        "amp_outlier_pct": np.nan,
        "amp_early_p50": np.nan,
        "amp_mid_p50": np.nan,
        "amp_late_p50": np.nan,
        "amp_early_late_ratio": np.nan,
        "amp_first_half_slope": np.nan,
        "amp_second_half_slope": np.nan,
        "amp_trend_spearman": np.nan,
        "amp_bin_cv_mean": np.nan,
        "amp_bin_cv_std": np.nan,
        "amp_bin_spread_ratio": np.nan,
    }
    _acg_bins = int(round(float(ACG_MAX_LAG_MS) / float(ACG_BIN_SIZE_MS)))
    BLOCK_C_DEFAULTS = {
        "firing_rate_hz": np.nan,
        "isi_mean_ms": np.nan,
        "isi_std_ms": np.nan,
        "isi_cv": np.nan,
        "isi_skew": np.nan,
        "presence_ratio": np.nan,
        "isi_violation_rate": np.nan,
        "burst_index": np.nan,
        "spike_count": np.nan,
    }
    for _i in range(_acg_bins):
        BLOCK_C_DEFAULTS[f"acg_{_i:02d}"] = np.nan
    BLOCK_D_DEFAULTS = {
        "d_rate_change_ratio": np.nan,
        "d_amp_change_ratio": np.nan,
        "d_waveform_corr_early_late": np.nan,
        "d_peak_channel_switch": np.nan,
    }
    BLOCK_E_DEFAULTS = {
        "rec_noise_mean_uv": np.nan,
        "rec_noise_std_uv": np.nan,
        "rec_duration_sec": np.nan,
        "rec_n_channels": np.nan,
        "rec_n_units": np.nan,
        "rec_sampling_rate": np.nan,
    }
    BLOCK_F_DEFAULTS = {
        "max_cosine": np.nan,
        "mean_cosine_top3": np.nan,
        "mean_cosine_top5": np.nan,
        "median_cosine": np.nan,
        "std_cosine": np.nan,
        "cosine_gap_top2": np.nan,
        "n_confusable": np.nan,
        "isolation_score": np.nan,
        "max_waveform_cosine": np.nan,
        "mean_waveform_cosine_top3": np.nan,
        "median_waveform_cosine": np.nan,
        "std_waveform_cosine": np.nan,
        "n_waveform_confusable": np.nan,
        "min_neighbor_dist_um": np.nan,
        "mean_neighbor_dist_um": np.nan,
    }


    def _manifest_row_with_defaults_phase1(item):
        if not isinstance(item, dict):
            raise RuntimeError("Phase 1 expects manifest rows as dictionaries from Phase 0.")
        row = dict(item)
        row["study_set"] = str(row.get("study_set", ""))
        row["study_name"] = str(row.get("study_name", ""))
        row["recording_name"] = str(row.get("recording_name", ""))
        row["sorter_name"] = str(row.get("sorter_name", ""))
        row["source_sorting_outputs_uri"] = str(row.get("source_sorting_outputs_uri", ""))
        row["source_recordings_uri"] = str(row.get("source_recordings_uri", ""))
        return row


    def load_recording_from_manifest_row(row):
        try:
            return sf.load_spikeforest_recording(
                study_name=row["study_name"],
                recording_name=row["recording_name"],
                uri=row.get("source_recordings_uri"),
            )
        except TypeError:
            return sf.load_spikeforest_recording(
                study_name=row["study_name"],
                recording_name=row["recording_name"],
            )


    def load_ground_truth_from_recording(recording_obj):
        gt = recording_obj.get_sorting_true_extractor()
        if gt is None or not hasattr(gt, "get_unit_ids"):
            raise RuntimeError("Ground truth sorting extractor is missing/invalid.")
        return gt


    def load_sorting_output_from_sorting_object(row, sampling_frequency):
        sorting_object = row.get("sorting_object")
        if sorting_object is None:
            raise RuntimeError("sorting_object_missing")
        if not isinstance(sorting_object, dict):
            raise RuntimeError(f"sorting_object_invalid_type:{type(sorting_object)}")
        if not _sorting_object_shape_supported(sorting_object):
            raise RuntimeError("sorting_format_unsupported")
        return load_sf_sorting_extractor(
            {
                "sorting_object": sorting_object,
                "return_code": row.get("return_code"),
                "timed_out": row.get("timed_out"),
                "recording_name": row.get("recording_name"),
                "sorter_name": row.get("sorter_name"),
            },
            sampling_frequency=sampling_frequency,
        )


    def _extract_firings_uri(sorting_object):
        if not isinstance(sorting_object, dict):
            return None
        if "firings" in sorting_object:
            return sorting_object.get("firings")
        data = sorting_object.get("data", {})
        if not isinstance(data, dict):
            return None
        return data.get("firings") or data.get("firingsUri")


    def _phase1_error_category(exc_or_msg):
        msg = str(exc_or_msg).lower()
        if "row_timeout_hard" in msg:
            return "row_timeout_hard"
        if "sorting_object_missing" in msg:
            return "sorting_object_missing"
        if "sorting_format_unsupported" in msg:
            return "sorting_format_unsupported"
        if "ground truth sorting extractor is missing" in msg:
            return "gt_load_failed"
        if "compare_sorter_to_ground_truth failed" in msg:
            return "compare_failed_canonical"
        if "maximum recursion depth exceeded" in msg:
            return "recursion_depth"
        if "gateway timeout" in msg or "function_invocation_timeout" in msg:
            return "gateway_timeout"
        if "timed out" in msg or "timeout" in msg:
            return "network_timeout"
        if "could not load sorting extractor" in msg:
            return "sorting_extractor_load_failed"
        if "ufunc 'equal'" in msg and "int64dtype" in msg and "strdtype" in msg:
            return "dtype_mismatch_int_vs_str"
        if "recording not found" in msg:
            return "recording_not_found"
        return "other"


    def _phase1_is_transient(category):
        transient = {
            "row_timeout_hard",
            "network_timeout",
            "gateway_timeout",
            "recursion_depth",
        }
        return category in transient


    def _safe_block_update(defaults, fn, *args, **kwargs):
        out = dict(defaults)
        try:
            got = fn(*args, **kwargs)
            if isinstance(got, dict):
                out.update(got)
        except Exception:
            pass
        return out


    def compare_and_extract_rows_phase1(recording, sorting_gt, sorting_out, metadata):
        if sorting_out is None or not hasattr(sorting_out, "get_unit_ids"):
            raise RuntimeError("Sorting output extractor is None or invalid for this item.")
        if sorting_gt is None or not hasattr(sorting_gt, "get_unit_ids"):
            raise RuntimeError("Ground-truth sorting extractor is None or invalid for this item.")

        sorting_gt_i, gt_id_map = _reindex_sorting_units_to_int(sorting_gt)
        sorting_out_i, out_id_map = _reindex_sorting_units_to_int(sorting_out)

        fs = float(recording.get_sampling_frequency())
        duration_sec = float(recording.get_num_samples() / fs) if fs > 0 else np.nan
        ch_locs = safe_channel_locations(recording)
        unit_ids = list(sorting_out_i.get_unit_ids())
        if len(unit_ids) == 0:
            return []

        try:
            cmp = sc.compare_sorter_to_ground_truth(
                sorting_gt_i,
                sorting_out_i,
                exhaustive_gt=True,
                match_score=0.5,
                chance_score=0.1,
            )
        except Exception as exc:
            gt_dtype = str(np.array(list(sorting_gt_i.get_unit_ids())).dtype)
            out_dtype = str(np.array(list(sorting_out_i.get_unit_ids())).dtype)
            raise RuntimeError(
                "compare_sorter_to_ground_truth failed "
                f"(gt_dtype={gt_dtype}, out_dtype={out_dtype}, n_gt={len(sorting_gt_i.get_unit_ids())}, "
                f"n_out={len(sorting_out_i.get_unit_ids())}): {exc}"
            ) from exc

        def _normalize_cmp_unit_id(value):
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if value == "" or value == -1:
                return None
            try:
                return int(value)
            except Exception:
                return value

        well_detected_ids = set(_normalize_cmp_unit_id(v) for v in cmp.get_well_detected_units())
        false_positive_ids = set(_normalize_cmp_unit_id(v) for v in cmp.get_false_positive_units())
        redundant_ids = set(_normalize_cmp_unit_id(v) for v in cmp.get_redundant_units())
        overmerged_ids = set(_normalize_cmp_unit_id(v) for v in cmp.get_overmerged_units())

        tested_targets_by_unit = {}
        for uid in unit_ids:
            best_gt_uid = _normalize_cmp_unit_id(cmp.best_match_21.get(uid, None))
            hungarian_gt_uid = _normalize_cmp_unit_id(cmp.hungarian_match_21.get(uid, None))
            num_tested = int(cmp.event_counts2.get(uid, 0))

            if best_gt_uid is None:
                tp = 0
                fn = np.nan
                fp = num_tested
                num_gt = np.nan
                recall = np.nan
                precision = 0.0 if num_tested > 0 else np.nan
                fmiss = np.nan
                fpos = 1.0 if num_tested > 0 else np.nan
                accuracy = 0.0 if num_tested > 0 else np.nan
                agreement_score = 0.0
            else:
                tp = int(cmp.match_event_count.at[best_gt_uid, uid])
                num_gt = int(cmp.event_counts1.at[best_gt_uid])
                fn = num_gt - tp
                fp = num_tested - tp
                recall = tp / num_gt if num_gt > 0 else np.nan
                precision = tp / num_tested if num_tested > 0 else np.nan
                fmiss = fn / num_gt if num_gt > 0 else np.nan
                fpos = fp / num_tested if num_tested > 0 else np.nan
                denom = tp + fn + fp
                accuracy = tp / denom if denom > 0 else np.nan
                agreement_score = float(cmp.agreement_scores.at[best_gt_uid, uid])

            if uid in well_detected_ids:
                match_status = "well_detected"
            elif uid in false_positive_ids:
                match_status = "false_positive"
            elif uid in redundant_ids:
                match_status = "redundant"
            elif uid in overmerged_ids:
                match_status = "overmerged"
            elif best_gt_uid is not None:
                match_status = "matched_best"
            else:
                match_status = "unclassified"

            tested_targets_by_unit[uid] = {
                "matched_gt_unit_id": None if best_gt_uid is None else str(gt_id_map.get(best_gt_uid, best_gt_uid)),
                "hungarian_gt_unit_id": None if hungarian_gt_uid is None else str(gt_id_map.get(hungarian_gt_uid, hungarian_gt_uid)),
                "match_status": match_status,
                "agreement_score": safe_float(agreement_score),
                "tp": int(tp),
                "fn": None if pd.isna(fn) else int(fn),
                "fp": int(fp),
                "num_gt": None if pd.isna(num_gt) else int(num_gt),
                "num_tested": int(num_tested),
                "recall": safe_float(recall),
                "precision": safe_float(precision),
                "fmiss": safe_float(fmiss),
                "fpos": safe_float(fpos),
                "accuracy": safe_float(accuracy),
            }

        block_e = _safe_block_update(
            BLOCK_E_DEFAULTS,
            extract_block_e,
            recording,
            len(unit_ids),
            metadata.get("study_set", ""),
        )
        templates = build_templates(
            recording,
            sorting_out_i,
            unit_ids,
            fs,
            metadata["recording_name"],
            metadata["sorter_name"],
        )
        si_metrics_by_unit = compute_spikeinterface_unit_metrics(recording, sorting_out_i, unit_ids)

        rows = []
        for uid in unit_ids:
            if uid not in templates:
                continue
            target_row = tested_targets_by_unit.get(uid)
            if target_row is None:
                continue
            tmpl = templates[uid].astype(np.float32)
            st_unit = sorting_out_i.get_unit_spike_train(uid, segment_index=0)
            peak_ch = int(np.argmax(np.ptp(tmpl, axis=1)))
            amps, amp_time_fraction = extract_unit_amplitudes(
                recording,
                st_unit,
                peak_ch,
                metadata["recording_name"],
                metadata["sorter_name"],
                uid,
            )

            original_uid = out_id_map.get(uid, uid)
            original_uid_str = str(original_uid)
            row = dict(metadata)
            row.update(
                {
                    "unit_id": original_uid_str,
                    "group_key": recording_group_key(
                        metadata["study_set"],
                        metadata["study_name"],
                        metadata["recording_name"],
                    ),
                    "row_uid": make_row_uid(
                        metadata["study_set"],
                        metadata["study_name"],
                        metadata["recording_name"],
                        metadata["sorter_name"],
                        original_uid_str,
                    ),
                    "fmiss": target_row["fmiss"],
                    "fpos": target_row["fpos"],
                    "accuracy": target_row["accuracy"],
                    "matched_gt_unit_id": target_row["matched_gt_unit_id"],
                    "hungarian_gt_unit_id": target_row["hungarian_gt_unit_id"],
                    "match_status": target_row["match_status"],
                    "agreement_score": target_row["agreement_score"],
                    "tp": target_row["tp"],
                    "fn": target_row["fn"],
                    "fp": target_row["fp"],
                    "num_gt": target_row["num_gt"],
                    "num_tested": target_row["num_tested"],
                    "recall": target_row["recall"],
                    "precision": target_row["precision"],
                }
            )

            row.update(BLOCK_A_DEFAULTS)
            row.update(BLOCK_B_DEFAULTS)
            row.update(BLOCK_C_DEFAULTS)
            row.update(BLOCK_D_DEFAULTS)
            row.update(BLOCK_E_DEFAULTS)
            row.update(BLOCK_F_DEFAULTS)

            row.update(si_metrics_by_unit.get(uid, {}))
            row.update(_safe_block_update(BLOCK_A_DEFAULTS, extract_block_a, tmpl, ch_locs, fs))
            row.update(_safe_block_update(BLOCK_B_DEFAULTS, extract_block_b, amps, amp_time_fraction))
            row.update(_safe_block_update(BLOCK_C_DEFAULTS, extract_block_c, st_unit, fs, duration_sec))
            row.update(
                _safe_block_update(
                    BLOCK_D_DEFAULTS,
                    extract_block_d,
                    recording=recording,
                    spike_train=st_unit,
                    mean_template=tmpl,
                    fs=fs,
                    duration_sec=duration_sec,
                    amplitudes=amps,
                    amp_time_fraction=amp_time_fraction,
                    recording_name=metadata["recording_name"],
                    sorter_name=metadata["sorter_name"],
                    unit_id=uid,
                )
            )
            row.update(block_e)
            row.update(_safe_block_update(BLOCK_F_DEFAULTS, extract_block_f, uid, templates, ch_locs))
            rows.append(row)
        return rows


    def _append_progress(payload):
        payload = dict(payload)
        payload.setdefault("ts", datetime.now().isoformat())
        EXTRACT_PROGRESS_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(EXTRACT_PROGRESS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\\n")


    def _row_timeout_seconds(row):
        if str(row.get("study_set", "")) == "HYBRID_JANELIA":
            return int(PHASE1_ROW_TIMEOUT_HYBRID_SEC)
        return int(PHASE1_ROW_TIMEOUT_SEC)


    def _phase1_worker_entry(row, part_path_str, result_path_str):
        t0 = time.time()
        out = {
            "status": "error",
            "skip_category": "other",
            "error_message": "",
            "attempts": 1,
            "n_rows": 0,
            "part_path": None,
            "elapsed_sec": np.nan,
        }
        try:
            row = _manifest_row_with_defaults_phase1(row)
            if row.get("sorting_object") is None:
                raise RuntimeError("sorting_object_missing")

            firings_uri = _extract_firings_uri(row.get("sorting_object"))
            if firings_uri and kcl is not None:
                # Warm a small remote object first; hard timeout handled by parent process.
                _ = kcl.load_file(firings_uri)

            R = load_recording_from_manifest_row(row)
            recording = R.get_recording_extractor()
            sorting_gt = load_ground_truth_from_recording(R)
            sorting_out = load_sorting_output_from_sorting_object(
                row,
                sampling_frequency=float(recording.get_sampling_frequency()),
            )

            rows = compare_and_extract_rows_phase1(
                recording,
                sorting_gt,
                sorting_out,
                metadata={
                    "study_set": row["study_set"],
                    "study_name": row["study_name"],
                    "recording_name": row["recording_name"],
                    "sorter_name": row["sorter_name"],
                    "source_sorting_outputs_uri": row.get("source_sorting_outputs_uri"),
                    "source_recordings_uri": row.get("source_recordings_uri"),
                },
            )
            if not rows:
                raise RuntimeError("No comparable units produced for this row.")
            pd.DataFrame(rows).to_parquet(part_path_str, index=False)
            out.update(
                {
                    "status": "ready",
                    "skip_category": "",
                    "error_message": "",
                    "n_rows": int(len(rows)),
                    "part_path": str(part_path_str),
                }
            )
        except Exception as exc:
            out.update(
                {
                    "status": "skip",
                    "skip_category": _phase1_error_category(exc),
                    "error_message": str(exc),
                    "n_rows": 0,
                    "part_path": None,
                }
            )
        finally:
            out["elapsed_sec"] = float(time.time() - t0)
            Path(result_path_str).write_text(json.dumps(out, ensure_ascii=True))


    def _phase1_run_row_attempt(row, part_path, timeout_sec):
        result_path = DIAGNOSTIC_DIR / f"{safe_stem(sf_output_key(row['study_set'], row['study_name'], row['recording_name'], row['sorter_name']))}.result.json"
        if result_path.exists():
            result_path.unlink()

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        proc = ctx.Process(
            target=_phase1_worker_entry,
            args=(dict(row), str(part_path), str(result_path)),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=timeout_sec)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
            if part_path.exists():
                try:
                    part_path.unlink()
                except Exception:
                    pass
            return {
                "status": "timeout",
                "skip_category": "row_timeout_hard",
                "error_message": f"row_timeout_hard ({timeout_sec}s)",
                "n_rows": 0,
                "part_path": None,
                "elapsed_sec": float(timeout_sec),
            }

        if not result_path.exists():
            return {
                "status": "error",
                "skip_category": "other",
                "error_message": "Worker exited without result payload.",
                "n_rows": 0,
                "part_path": None,
                "elapsed_sec": np.nan,
            }
        payload = read_json(result_path, default={})
        if result_path.exists():
            try:
                result_path.unlink()
            except Exception:
                pass
        return payload


    def _phase1_stability_map_from_state(state):
        rows_state = state.get("rows", {})
        if not isinstance(rows_state, dict):
            return {}
        counts = {}
        for _, item in rows_state.items():
            if not isinstance(item, dict):
                continue
            sorter = str(item.get("sorter_name", ""))
            if not sorter:
                continue
            c = counts.setdefault(sorter, {"ok": 0, "total": 0})
            c["total"] += 1
            if str(item.get("status")) == "ready":
                c["ok"] += 1
        out = {}
        for sorter, c in counts.items():
            out[sorter] = float(c["ok"] / max(c["total"], 1))
        return out


    def _phase1_dataset_priority(study_set):
        synthetic = {
            "SYNTH_MONOTRODE",
            "SYNTH_MAGLAND",
            "SYNTH_JANELIA",
            "SYNTH_BIONET",
            "SYNTH_MEAREC_NEURONEX",
            "SYNTH_VISAPY",
            "SYNTH_MEAREC_TETRODE",
        }
        paired = {
            "PAIRED_ENGLISH",
            "PAIRED_MEA64C_YGER",
            "PAIRED_BOYDEN",
            "PAIRED_KAMPFF",
            "PAIRED_CRCNS_HC1",
        }
        if study_set in synthetic:
            return 3
        if study_set in paired:
            return 2
        if study_set == "HYBRID_JANELIA":
            return 1
        return 0


    def _phase1_sort_manifest(selected_rows, state):
        sorter_stability = _phase1_stability_map_from_state(state)
        out = list(selected_rows)
        out.sort(
            key=lambda r: (
                -int(r.get("recording_num_true_units") or 0),
                -int(_phase1_dataset_priority(str(r.get("study_set", "")))),
                -float(sorter_stability.get(str(r.get("sorter_name", "")), 0.0)),
                -float(r.get("priority_score", 0.0)),
                str(r.get("study_set", "")),
                str(r.get("study_name", "")),
                str(r.get("recording_name", "")),
                str(r.get("sorter_name", "")),
            )
        )
        return out


    def _phase1_default_state(total_rows):
        return {
            "started_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
            "total_rows": int(total_rows),
            "completed_count": 0,
            "failed_count": 0,
            "running_key": None,
            "running_stage": "init",
            "complete": False,
            "rows": {},
        }


    # ── Phase 1B/C — Timeout-safe sequential orchestration ─────────────────────
    if FORCE_REBUILD_TRAIN_FEATURES:
        reset_path(TRAIN_PARQUET)
        reset_path(TRAIN_FEATURE_STATE_JSON)
        reset_path(TRAIN_PARTS_DIR)
        reset_path(TRAIN_FAILURES_PARQUET)
        reset_path(TRAIN_SUCCESS_MANIFEST_PARQUET)
        reset_path(EXTRACT_PROGRESS_JSONL)
        TRAIN_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_PARQUET.exists() and REUSE_EXISTING_TRAIN_FEATURES and not FORCE_REBUILD_TRAIN_FEATURES:
        df_train = safe_read_parquet(TRAIN_PARQUET)
        log_status(f"Phase 1 skipped. Reusing training features from {TRAIN_PARQUET}")
    elif not SPIKEFOREST_AVAILABLE:
        raise RuntimeError(
            "SpikeForest is required to build the training parquet from scratch. "
            "Re-run the install cell, then resume."
        )
    elif "selected" not in globals() or len(selected) == 0:
        raise RuntimeError(
            "Run Phase 0 first. If Phase 0 produced zero rows, check SPIKEFOREST_STUDY_SETS and SHA1 manifest URIs."
        )
    else:
        TRAIN_PARTS_DIR.mkdir(parents=True, exist_ok=True)
        selected_to_process = list(selected)
        if PHASE1_MAX_ITEMS is not None:
            selected_to_process = selected_to_process[: int(PHASE1_MAX_ITEMS)]
            log_status(f"Phase 1 debug limit active: processing first {len(selected_to_process)} items")

        state = read_json(TRAIN_FEATURE_STATE_JSON, default=None)
        if not isinstance(state, dict) or "rows" not in state:
            state = _phase1_default_state(total_rows=len(selected_to_process))
        state["total_rows"] = int(len(selected_to_process))
        selected_to_process = _phase1_sort_manifest(selected_to_process, state)

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Extracting features...", total=len(selected_to_process))

            for idx, item in enumerate(selected_to_process, start=1):
                row = _manifest_row_with_defaults_phase1(item)
                key = sf_output_key(row["study_set"], row["study_name"], row["recording_name"], row["sorter_name"])
                part_path = TRAIN_PARTS_DIR / f"{safe_stem(key)}.parquet"
                progress.update(task, description=f"[cyan]{key[:72]}")

                row_state = state["rows"].get(key, {})
                if (
                    RESUME_FROM_CHECKPOINTS
                    and str(row_state.get("status")) == "ready"
                    and part_path.exists()
                ):
                    progress.advance(task)
                    continue

                state["running_key"] = key
                state["running_stage"] = "start"
                state["last_heartbeat"] = datetime.now().isoformat()
                write_json(TRAIN_FEATURE_STATE_JSON, state)

                _append_progress(
                    {
                        "row_index": idx,
                        "row_total": len(selected_to_process),
                        "key": key,
                        "stage": "start",
                        "attempt": 1,
                        "completed": int(state.get("completed_count", 0)),
                        "failed": int(state.get("failed_count", 0)),
                    }
                )

                final_outcome = None
                for attempt in range(1, _KACHERY_MAX_RETRIES + 1):
                    timeout_sec = _row_timeout_seconds(row)
                    outcome = _phase1_run_row_attempt(row, part_path, timeout_sec=timeout_sec)
                    category = str(outcome.get("skip_category", "") or "")
                    if outcome.get("status") == "ready":
                        final_outcome = dict(outcome)
                        final_outcome["attempts"] = attempt
                        _append_progress(
                            {
                                "key": key,
                                "status": "ready",
                                "attempt": attempt,
                                "elapsed_sec": float(outcome.get("elapsed_sec", np.nan)),
                                "n_units": int(outcome.get("n_rows", 0)),
                                "stage": "write_part",
                            }
                        )
                        break

                    _append_progress(
                        {
                            "key": key,
                            "status": "error",
                            "attempt": attempt,
                            "stage": "row_attempt",
                            "error": str(outcome.get("error_message", "")),
                            "category": category,
                            "transient": bool(_phase1_is_transient(category)),
                        }
                    )

                    if attempt < _KACHERY_MAX_RETRIES and _phase1_is_transient(category):
                        sleep_s = min(30, 2 + attempt * 3 + int(np.random.default_rng().integers(0, 2)))
                        log_status(
                            f"  attempt {attempt}/{_KACHERY_MAX_RETRIES} transient error for "
                            f"{key[:70]}: {outcome.get('error_message', '')} | retry in {sleep_s}s"
                        )
                        time.sleep(sleep_s)
                        gc.collect()
                        continue

                    final_outcome = dict(outcome)
                    final_outcome["attempts"] = attempt
                    break

                if final_outcome is None:
                    final_outcome = {
                        "status": "error",
                        "skip_category": "other",
                        "error_message": "No extraction outcome captured.",
                        "attempts": _KACHERY_MAX_RETRIES,
                        "elapsed_sec": np.nan,
                        "n_rows": 0,
                        "part_path": None,
                    }

                row_outcome = {
                    "key": key,
                    "study_set": row["study_set"],
                    "study_name": row["study_name"],
                    "recording_name": row["recording_name"],
                    "sorter_name": row["sorter_name"],
                    "status": "ready" if final_outcome.get("status") == "ready" else str(final_outcome.get("status")),
                    "skip_category": str(final_outcome.get("skip_category", "")),
                    "error_message": str(final_outcome.get("error_message", "")),
                    "elapsed_sec": float(final_outcome.get("elapsed_sec", np.nan)),
                    "attempts": int(final_outcome.get("attempts", 1)),
                    "part_path": final_outcome.get("part_path"),
                    "updated_at": datetime.now().isoformat(),
                }
                state["rows"][key] = row_outcome
                state["completed_count"] = int(sum(1 for v in state["rows"].values() if v.get("status") == "ready"))
                state["failed_count"] = int(sum(1 for v in state["rows"].values() if v.get("status") != "ready"))
                state["last_heartbeat"] = datetime.now().isoformat()
                state["running_stage"] = "row_done"
                write_json(TRAIN_FEATURE_STATE_JSON, state)

                if row_outcome["status"] != "ready":
                    log_status(f"SKIP {key}: {row_outcome['error_message']}")

                progress.advance(task)

        rows_outcomes = list(state.get("rows", {}).values())
        success_rows = [r for r in rows_outcomes if str(r.get("status")) == "ready"]
        failure_rows = [r for r in rows_outcomes if str(r.get("status")) != "ready"]

        if failure_rows:
            pd.DataFrame(failure_rows).to_parquet(TRAIN_FAILURES_PARQUET, index=False)
        else:
            if TRAIN_FAILURES_PARQUET.exists():
                reset_path(TRAIN_FAILURES_PARQUET)

        if success_rows:
            pd.DataFrame(success_rows).to_parquet(TRAIN_SUCCESS_MANIFEST_PARQUET, index=False)
        else:
            if TRAIN_SUCCESS_MANIFEST_PARQUET.exists():
                reset_path(TRAIN_SUCCESS_MANIFEST_PARQUET)

        part_paths = []
        for r in success_rows:
            p = r.get("part_path")
            if not p:
                continue
            p = Path(p)
            if p.exists():
                part_paths.append(p)

        if not part_paths:
            state["complete"] = False
            state["running_key"] = None
            state["running_stage"] = "failed_no_parts"
            state["last_heartbeat"] = datetime.now().isoformat()
            write_json(TRAIN_FEATURE_STATE_JSON, state)
            summary = {
                "generated_at": datetime.now().isoformat(),
                "n_manifest_rows": len(selected_to_process),
                "n_completed": int(state.get("completed_count", 0)),
                "n_failed": int(state.get("failed_count", 0)),
                "top_failed_categories": dict(
                    pd.Series([r.get("skip_category", "other") for r in failure_rows]).value_counts().head(20)
                )
                if failure_rows
                else {},
                "state_path": str(TRAIN_FEATURE_STATE_JSON),
                "failure_parquet": str(TRAIN_FAILURES_PARQUET),
                "progress_jsonl": str(EXTRACT_PROGRESS_JSONL),
            }
            write_json(DATA_AUDIT_PHASE1_JSON, summary)
            raise RuntimeError(
                "No training rows were extracted. Check train_failures.parquet and extract_progress.jsonl."
            )

        df_train = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
        identity_cols = ["study_set", "study_name", "recording_name", "sorter_name", "unit_id"]
        df_train = df_train.drop_duplicates(subset=identity_cols).reset_index(drop=True)

        if "group_key" not in df_train.columns:
            df_train["group_key"] = df_train.apply(
                lambda row: recording_group_key(row["study_set"], row["study_name"], row["recording_name"]),
                axis=1,
            )
        if "row_uid" not in df_train.columns:
            df_train["row_uid"] = df_train.apply(
                lambda row: make_row_uid(
                    row["study_set"],
                    row["study_name"],
                    row["recording_name"],
                    row["sorter_name"],
                    row["unit_id"],
                ),
                axis=1,
            )

        required_identity = ["study_set", "study_name", "recording_name", "sorter_name", "unit_id", "group_key", "row_uid"]
        required_targets = ["fmiss", "fpos", "accuracy"]
        missing_required = [c for c in required_identity + required_targets if c not in df_train.columns]
        if missing_required:
            raise RuntimeError(f"Phase 1 output missing required columns: {missing_required}")

        for target_col in required_targets:
            df_train[target_col] = pd.to_numeric(df_train[target_col], errors="coerce")
        target_range_mask = (
            (df_train["fmiss"].isna() | df_train["fmiss"].between(0.0, 1.0, inclusive="both"))
            & df_train["fpos"].between(0.0, 1.0, inclusive="both")
            & df_train["accuracy"].between(0.0, 1.0, inclusive="both")
        )
        invalid_target_rows = int((~target_range_mask).sum())
        if invalid_target_rows > 0:
            df_train = df_train.loc[target_range_mask].reset_index(drop=True)

        df_train = stable_sort_frame(df_train, ["group_key", "sorter_name", "unit_id", "row_uid"])
        df_train.to_parquet(TRAIN_PARQUET, index=False)

        # ── Phase 1D — Dataset audits ───────────────────────────────────────────
        row_uid_dupes = int(df_train["row_uid"].duplicated().sum())
        id_dupes = int(df_train.duplicated(subset=identity_cols).sum())

        block_cols = {
            "A": [c for c in BLOCK_A_DEFAULTS if c in df_train.columns],
            "B": [c for c in BLOCK_B_DEFAULTS if c in df_train.columns],
            "C": [c for c in BLOCK_C_DEFAULTS if c in df_train.columns],
            "D": [c for c in BLOCK_D_DEFAULTS if c in df_train.columns],
            "E": [c for c in BLOCK_E_DEFAULTS if c in df_train.columns],
            "F": [c for c in BLOCK_F_DEFAULTS if c in df_train.columns],
        }
        coverage = {
            "generated_at": datetime.now().isoformat(),
            "n_rows": int(len(df_train)),
            "coverage_pct_by_block": {},
            "coverage_pct_by_column": {},
        }
        for block, cols in block_cols.items():
            if not cols:
                coverage["coverage_pct_by_block"][block] = np.nan
                continue
            col_cov = {}
            for c in cols:
                col_cov[c] = float(df_train[c].notna().mean() * 100.0)
            coverage["coverage_pct_by_column"][block] = col_cov
            coverage["coverage_pct_by_block"][block] = float(np.mean(list(col_cov.values())))
        write_json(FEATURE_COVERAGE_BY_BLOCK_JSON, coverage)

        audit = {
            "generated_at": datetime.now().isoformat(),
            "n_rows": int(len(df_train)),
            "n_groups": int(df_train["group_key"].nunique()),
            "n_recordings": int(df_train["recording_name"].nunique()),
            "n_sorters": int(df_train["sorter_name"].nunique()),
            "n_study_sets": int(df_train["study_set"].nunique()),
            "invalid_target_rows_dropped": invalid_target_rows,
            "row_uid_duplicates": row_uid_dupes,
            "identity_duplicates": id_dupes,
            "rows_by_study_set": {str(k): int(v) for k, v in df_train.groupby("study_set").size().items()},
            "rows_by_sorter": {str(k): int(v) for k, v in df_train.groupby("sorter_name").size().items()},
            "missing_targets": {c: int(df_train[c].isna().sum()) for c in required_targets},
            "failure_category_histogram": dict(
                pd.Series([r.get("skip_category", "other") for r in failure_rows]).value_counts()
            )
            if failure_rows
            else {},
            "train_parquet": str(TRAIN_PARQUET),
            "failure_parquet": str(TRAIN_FAILURES_PARQUET),
            "success_parquet": str(TRAIN_SUCCESS_MANIFEST_PARQUET),
            "progress_jsonl": str(EXTRACT_PROGRESS_JSONL),
        }
        write_json(DATA_AUDIT_PHASE1_JSON, audit)

        state["complete"] = True
        state["running_key"] = None
        state["running_stage"] = "complete"
        state["last_heartbeat"] = datetime.now().isoformat()
        state["finished_at"] = datetime.now().isoformat()
        state["n_rows"] = int(len(df_train))
        state["train_parquet"] = str(TRAIN_PARQUET)
        state["failure_parquet"] = str(TRAIN_FAILURES_PARQUET)
        state["success_manifest_parquet"] = str(TRAIN_SUCCESS_MANIFEST_PARQUET)
        state["feature_coverage_json"] = str(FEATURE_COVERAGE_BY_BLOCK_JSON)
        state["data_audit_json"] = str(DATA_AUDIT_PHASE1_JSON)
        write_json(TRAIN_FEATURE_STATE_JSON, state)

        elapsed = time.time() - start_time
        log_status(
            f"Phase 1 complete. {len(df_train)} rows | {df_train['group_key'].nunique()} groups | "
            f"elapsed: {elapsed / 60:.1f} min"
        )
    """
).strip()


BLOCK_D_SNIPPET = dedent(
    """

    # ════════════════════════════════════════════════════
    #  BLOCK D — Temporal Stability
    # ════════════════════════════════════════════════════

    def extract_block_d(
        recording,
        spike_train,
        mean_template,
        fs=30000,
        duration_sec=600.0,
        amplitudes=None,
        amp_time_fraction=None,
        recording_name="",
        sorter_name="",
        unit_id="",
    ):
        st = np.asarray(spike_train, dtype=np.int64).ravel()
        if st.size < 5:
            return {
                "d_rate_change_ratio": np.nan,
                "d_amp_change_ratio": np.nan,
                "d_waveform_corr_early_late": np.nan,
                "d_peak_channel_switch": np.nan,
            }

        n_samples = int(recording.get_num_samples()) if recording is not None else int(duration_sec * fs)
        n_samples = max(n_samples, 1)
        t1 = int(n_samples / 3)
        t2 = int(2 * n_samples / 3)
        early_st = st[st < t1]
        late_st = st[st >= t2]

        third_sec = max(float(duration_sec) / 3.0, 1e-8)
        early_rate = float(len(early_st) / third_sec)
        late_rate = float(len(late_st) / third_sec)
        rate_ratio = float(late_rate / (early_rate + 1e-8))

        amp_ratio = np.nan
        if amplitudes is not None and amp_time_fraction is not None:
            amps = np.asarray(amplitudes, dtype=np.float32).ravel()
            fracs = np.asarray(amp_time_fraction, dtype=np.float32).ravel()
            n = min(len(amps), len(fracs))
            if n > 0:
                amps = amps[:n]
                fracs = fracs[:n]
                early_amp = amps[fracs <= (1.0 / 3.0)]
                late_amp = amps[fracs >= (2.0 / 3.0)]
                if len(early_amp) > 0 and len(late_amp) > 0:
                    amp_ratio = float(np.median(late_amp) / (np.median(early_amp) + 1e-8))

        corr = np.nan
        peak_switch = np.nan
        try:
            n_before = int(round((TEMPLATE_WINDOW_MS / 1000.0) * fs))
            n_after = n_before
            tmpl_early = build_unit_template(recording, early_st, n_before, n_after)
            tmpl_late = build_unit_template(recording, late_st, n_before, n_after)
            if tmpl_early is not None and tmpl_late is not None:
                a = tmpl_early.astype(np.float32).ravel()
                b = tmpl_late.astype(np.float32).ravel()
                if a.size == b.size and a.size > 0:
                    if np.std(a) > 0 and np.std(b) > 0:
                        corr = float(np.corrcoef(a, b)[0, 1])
                    else:
                        corr = 1.0
                p1 = int(np.argmax(np.ptp(tmpl_early.astype(np.float32), axis=1)))
                p2 = int(np.argmax(np.ptp(tmpl_late.astype(np.float32), axis=1)))
                peak_switch = float(1.0 if p1 != p2 else 0.0)
        except Exception:
            corr = np.nan
            peak_switch = np.nan

        return {
            "d_rate_change_ratio": rate_ratio,
            "d_amp_change_ratio": amp_ratio,
            "d_waveform_corr_early_late": corr,
            "d_peak_channel_switch": peak_switch,
        }
    """
).rstrip()


BLOCK_A_REPLACEMENT = dedent(
    """
    # ════════════════════════════════════════════════════
    #  BLOCK A — Waveform Morphology
    # ════════════════════════════════════════════════════

    def _waveform_bin_features(wf_norm, trough_idx, ms_per_sample, n_bins=30):
        wf_norm = np.asarray(wf_norm, dtype=np.float32).ravel()
        if wf_norm.size == 0 or n_bins <= 0:
            return {}

        positions_ms = (np.arange(wf_norm.size, dtype=np.float32) - float(trough_idx)) * float(ms_per_sample)
        span_ms = float(max(abs(positions_ms[0]), abs(positions_ms[-1]), float(ms_per_sample)))
        target_ms = np.linspace(-span_ms, span_ms, int(n_bins), dtype=np.float32)
        bins = np.interp(
            target_ms,
            positions_ms,
            wf_norm,
            left=float(wf_norm[0]),
            right=float(wf_norm[-1]),
        ).astype(np.float32)
        return {f"wf_bin_{i:02d}": float(v) for i, v in enumerate(bins)}


    def extract_block_a(mean_template, channel_locations, fs=30000, n_waveform_bins=30):
        \"\"\"mean_template: (n_ch, n_samples) float32. Returns dict of features.\"\"\"
        n_ch, n_t = mean_template.shape
        ptp_per_ch = np.ptp(mean_template, axis=1)
        peak_ch = int(np.argmax(ptp_per_ch))
        wf = mean_template[peak_ch].astype(np.float32)
        ms_per_sample = 1000.0 / fs

        trough_idx = int(np.argmin(wf))
        peak_idx = int(np.argmax(wf))
        ptp_uv = float(ptp_per_ch[peak_ch])
        wf_std = float(np.std(wf))
        abs_max = float(np.max(np.abs(wf))) + 1e-8
        wf_norm = (wf / abs_max).astype(np.float32)

        def _contiguous_width_ms(center_idx, threshold):
            left = int(center_idx)
            right = int(center_idx)
            while left > 0 and wf[left - 1] <= threshold:
                left -= 1
            while right < n_t - 1 and wf[right + 1] <= threshold:
                right += 1
            return float(right - left + 1) * ms_per_sample

        def _time_to_last_nonnegative_before(idx):
            if idx <= 0:
                return np.nan
            candidates = np.where(wf[:idx] >= 0)[0]
            if candidates.size == 0:
                return np.nan
            return float(idx - int(candidates[-1])) * ms_per_sample

        def _time_to_first_nonnegative_after(idx):
            if idx >= n_t - 1:
                return np.nan
            candidates = np.where(wf[idx + 1:] >= 0)[0]
            if candidates.size == 0:
                return np.nan
            return float(int(candidates[0]) + 1) * ms_per_sample

        # Half-width: legacy global count below half trough amplitude.
        half_val = wf[trough_idx] / 2.0
        below = np.where(wf < half_val)[0]
        hw_ms = float(len(below)) * ms_per_sample if len(below) > 0 else 0.0

        if trough_idx < n_t - 1:
            repol_slope = float(wf[trough_idx + 1] - wf[trough_idx]) / ms_per_sample
        else:
            repol_slope = 0.0

        pre_seg = wf[:trough_idx]
        post_seg = wf[trough_idx + 1:] if trough_idx < n_t - 1 else np.array([], dtype=np.float32)

        pre_peak_uv = 0.0
        pre_peak_idx = None
        if pre_seg.size > 0:
            pre_peak_raw = float(np.max(pre_seg))
            if pre_peak_raw > 0:
                pre_peak_uv = pre_peak_raw
                pre_peak_idx = int(np.argmax(pre_seg))

        post_peak_uv = 0.0
        post_peak_idx = None
        if post_seg.size > 0:
            post_peak_rel = int(np.argmax(post_seg))
            post_peak_raw = float(post_seg[post_peak_rel])
            if post_peak_raw > 0:
                post_peak_uv = post_peak_raw
                post_peak_idx = trough_idx + 1 + post_peak_rel

        trough_val = float(wf[trough_idx])
        trough_mag = abs(trough_val)

        # Redefined as bounded pre/post-positive-lobe asymmetry around the trough.
        asymmetry = float((post_peak_uv - pre_peak_uv) / (post_peak_uv + pre_peak_uv + 1e-8))
        prepeak_ratio = float(pre_peak_uv / (trough_mag + 1e-8))
        rebound_ratio = float(post_peak_uv / (trough_mag + 1e-8))

        peak_trough_ms = float(abs(peak_idx - trough_idx)) * ms_per_sample
        prepeak_to_trough_ms = (
            float(trough_idx - pre_peak_idx) * ms_per_sample if pre_peak_idx is not None else np.nan
        )
        trough_to_peak_ms = (
            float(post_peak_idx - trough_idx) * ms_per_sample if post_peak_idx is not None else np.nan
        )

        trough_width_25 = _contiguous_width_ms(trough_idx, 0.25 * trough_val)
        trough_width_50 = _contiguous_width_ms(trough_idx, 0.50 * trough_val)
        trough_width_75 = _contiguous_width_ms(trough_idx, 0.75 * trough_val)

        neg_area = float(np.sum(np.maximum(-wf, 0.0)) * ms_per_sample)
        pos_area = float(np.sum(np.maximum(wf, 0.0)) * ms_per_sample)
        pos_neg_area_ratio = float(pos_area / (neg_area + 1e-8))

        line_length_norm = float(np.sum(np.abs(np.diff(wf_norm)))) if n_t > 1 else 0.0
        curvature_norm = float(np.sum(np.abs(np.diff(wf_norm, n=2)))) if n_t > 2 else 0.0
        trough_sharpness = (
            float(abs(wf_norm[trough_idx - 1] - 2.0 * wf_norm[trough_idx] + wf_norm[trough_idx + 1]))
            if 0 < trough_idx < n_t - 1
            else np.nan
        )
        pre_trough_slope = (
            float(wf[trough_idx] - wf[trough_idx - 1]) / ms_per_sample if trough_idx > 0 else np.nan
        )
        post_peak_slope = (
            float(wf[post_peak_idx + 1] - wf[post_peak_idx]) / ms_per_sample
            if post_peak_idx is not None and post_peak_idx < n_t - 1
            else np.nan
        )

        max_ptp = float(np.max(ptp_per_ch)) + 1e-8
        active_mask = ptp_per_ch > AMPLITUDE_THRESHOLD * max_ptp
        n_active = int(np.sum(active_mask))
        locs = np.asarray(channel_locations) if channel_locations is not None else None

        if (
            n_active > 0
            and locs is not None
            and locs.ndim == 2
            and locs.shape[0] >= n_ch
            and locs.shape[1] >= 2
        ):
            depths = locs[active_mask, 1]
            weights = ptp_per_ch[active_mask]
            com_um = float(np.average(depths, weights=weights))
            spread_um = float(np.max(depths) - np.min(depths)) if n_active > 1 else 0.0
            spread_w = float(np.sqrt(np.average((depths - com_um) ** 2, weights=weights)))
            peak_depth = float(locs[peak_ch, 1])
        else:
            com_um = spread_um = spread_w = peak_depth = 0.0

        feats = {
            'snr':                     ptp_uv / (wf_std + 1e-8),
            'peak_to_trough_uv':       ptp_uv,
            'half_width_ms':           hw_ms,
            'repolarization_slope':    repol_slope,
            'pre_trough_peak_uv':      pre_peak_uv,
            'post_trough_peak_uv':     post_peak_uv,
            'trough_time_ms':          float(trough_idx) * ms_per_sample,
            'wf_energy':               float(np.sum(wf**2)),
            'wf_ptp_ratio':            ptp_uv / (wf_std + 1e-8),
            'wf_asymmetry':            asymmetry,
            'wf_prepeak_ratio':        prepeak_ratio,
            'wf_rebound_ratio':        rebound_ratio,
            'wf_peak_trough_ms':       peak_trough_ms,
            'wf_prepeak_to_trough_ms': prepeak_to_trough_ms,
            'wf_trough_to_peak_ms':    trough_to_peak_ms,
            'wf_zero_cross_pre_ms':    _time_to_last_nonnegative_before(trough_idx),
            'wf_zero_cross_post_ms':   _time_to_first_nonnegative_after(trough_idx),
            'wf_trough_width_25_ms':   trough_width_25,
            'wf_trough_width_50_ms':   trough_width_50,
            'wf_trough_width_75_ms':   trough_width_75,
            'wf_neg_area_uv_ms':       neg_area,
            'wf_pos_area_uv_ms':       pos_area,
            'wf_pos_neg_area_ratio':   pos_neg_area_ratio,
            'wf_line_length_norm':     line_length_norm,
            'wf_curvature_norm':       curvature_norm,
            'wf_trough_sharpness':     trough_sharpness,
            'wf_pre_trough_slope':     pre_trough_slope,
            'wf_post_peak_slope':      post_peak_slope,
            'template_norm':           float(np.linalg.norm(wf)),
            'n_active_channels':       n_active,
            'spread_um':               spread_um,
            'center_of_mass_um':       com_um,
            'spread_weighted_um':      spread_w,
            'peak_channel_depth_um':   peak_depth,
            'n_channels':              n_ch,
        }
        feats.update(_waveform_bin_features(wf_norm, trough_idx, ms_per_sample, n_bins=n_waveform_bins))
        return feats
    """
).strip()


BLOCK_F_REPLACEMENT = dedent(
    """
    # ════════════════════════════════════════════════════
    #  BLOCK F — Pairwise Relational (only when n_units ≥ 2)
    # ════════════════════════════════════════════════════

    def cosine_amp_normalised(tA, tB):
        \"\"\"Amplitude-normalised cosine — measures shape, not amplitude.\"\"\"
        pA = float(np.max(np.abs(tA))) + 1e-8
        pB = float(np.max(np.abs(tB))) + 1e-8
        A = (tA / pA).ravel().astype(np.float32)
        B = (tB / pB).ravel().astype(np.float32)
        val = float(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8))
        return float(np.clip(val, -1.0, 1.0))

    def cosine_peak_waveform(tA, tB):
        \"\"\"Peak-channel waveform cosine after per-waveform amplitude normalisation.\"\"\"
        wA = tA[int(np.argmax(np.ptp(tA, axis=1)))].astype(np.float32)
        wB = tB[int(np.argmax(np.ptp(tB, axis=1)))].astype(np.float32)
        pA = float(np.max(np.abs(wA))) + 1e-8
        pB = float(np.max(np.abs(wB))) + 1e-8
        A = (wA / pA).ravel()
        B = (wB / pB).ravel()
        val = float(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8))
        return float(np.clip(val, -1.0, 1.0))

    def _topk_mean(values, k):
        arr = np.sort(np.asarray(values, dtype=np.float32))
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr[-min(k, arr.size):]))

    def extract_block_f(unit_id, templates_dict, channel_locations):
        \"\"\"templates_dict: {unit_id: (n_ch, n_t) float16}.\"\"\"
        empty = {
            'max_cosine':                 0.0,
            'mean_cosine_top3':           0.0,
            'mean_cosine_top5':           0.0,
            'median_cosine':              0.0,
            'std_cosine':                 0.0,
            'cosine_gap_top2':            np.nan,
            'n_confusable':               0,
            'isolation_score':            1.0,
            'max_waveform_cosine':        0.0,
            'mean_waveform_cosine_top3':  0.0,
            'median_waveform_cosine':     0.0,
            'std_waveform_cosine':        0.0,
            'n_waveform_confusable':      0,
            'min_neighbor_dist_um':       np.nan,
            'mean_neighbor_dist_um':      np.nan,
        }
        if len(templates_dict) < 2:
            return empty

        tA = templates_dict[unit_id].astype(np.float32)
        cos_vals = []
        wf_cos_vals = []
        dists = []
        peak_A = int(np.argmax(np.ptp(tA, axis=1)))
        locs = np.asarray(channel_locations) if channel_locations is not None else None

        for uid_B, tB_f16 in templates_dict.items():
            if uid_B == unit_id:
                continue
            tB = tB_f16.astype(np.float32)
            cos_vals.append(cosine_amp_normalised(tA, tB))
            wf_cos_vals.append(cosine_peak_waveform(tA, tB))
            if locs is not None and locs.ndim == 2 and locs.shape[0] > peak_A and locs.shape[1] >= 2:
                peak_B = int(np.argmax(np.ptp(tB, axis=1)))
                if locs.shape[0] > peak_B:
                    dists.append(float(np.linalg.norm(locs[peak_A] - locs[peak_B])))
            del tB

        cos_vals = np.asarray(cos_vals, dtype=np.float32)
        wf_cos_vals = np.asarray(wf_cos_vals, dtype=np.float32)
        sorted_cos = np.sort(cos_vals)
        max_cos = float(sorted_cos[-1])
        second_cos = float(sorted_cos[-2]) if sorted_cos.size >= 2 else np.nan

        out = dict(empty)
        out.update(
            {
                'max_cosine':                 max_cos,
                'mean_cosine_top3':           _topk_mean(cos_vals, 3),
                'mean_cosine_top5':           _topk_mean(cos_vals, 5),
                'median_cosine':              float(np.median(cos_vals)),
                'std_cosine':                 float(np.std(cos_vals)),
                'cosine_gap_top2':            float(max_cos - second_cos) if np.isfinite(second_cos) else np.nan,
                'n_confusable':               int(np.sum(cos_vals > COSINE_THRESHOLD)),
                'isolation_score':            float(1.0 - max_cos),
                'max_waveform_cosine':        float(np.max(wf_cos_vals)),
                'mean_waveform_cosine_top3':  _topk_mean(wf_cos_vals, 3),
                'median_waveform_cosine':     float(np.median(wf_cos_vals)),
                'std_waveform_cosine':        float(np.std(wf_cos_vals)),
                'n_waveform_confusable':      int(np.sum(wf_cos_vals > COSINE_THRESHOLD)),
                'min_neighbor_dist_um':       float(np.min(dists)) if dists else np.nan,
                'mean_neighbor_dist_um':      float(np.mean(dists)) if dists else np.nan,
            }
        )
        return out
    """
).strip()


def _replace_section(source: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = source.find(start_marker)
    if start == -1:
        raise RuntimeError(f"Could not find start marker: {start_marker!r}")
    end = source.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"Could not find end marker: {end_marker!r}")
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def patch_waveform_sections_only(source: str) -> str:
    out = _replace_section(
        source,
        "# ════════════════════════════════════════════════════\n#  BLOCK A — Waveform Morphology\n# ════════════════════════════════════════════════════",
        "# ════════════════════════════════════════════════════\n#  BLOCK B — Amplitude Statistics\n# ════════════════════════════════════════════════════",
        BLOCK_A_REPLACEMENT,
    )
    out = _replace_section(
        out,
        "# ════════════════════════════════════════════════════\n#  BLOCK F — Pairwise Relational (only when n_units ≥ 2)\n# ════════════════════════════════════════════════════",
        "print('✓ Feature functions defined",
        BLOCK_F_REPLACEMENT,
    )
    return out


def patch_cell5(source: str) -> str:
    out = source
    if "SPIKEFOREST_MANIFEST_FULL_JSON" not in out:
        out = out.replace(
            'DATA_CONTRACTS_JSON         = MANIFEST_DIR / "data_contracts.json"',
            'DATA_CONTRACTS_JSON         = MANIFEST_DIR / "data_contracts.json"\n'
            'SPIKEFOREST_MANIFEST_FULL_JSON = MANIFEST_DIR / "spikeforest_manifest_full.json"\n'
            'SPIKEFOREST_MANIFEST_SELECTED_JSON = MANIFEST_DIR / "spikeforest_manifest_selected.json"\n'
            'PHASE0_SUMMARY_JSON         = MANIFEST_DIR / "phase0_summary.json"\n'
            'TRAIN_FAILURES_PARQUET      = AUDIT_DIR / "train_failures.parquet"\n'
            'TRAIN_SUCCESS_MANIFEST_PARQUET = AUDIT_DIR / "train_success_manifest.parquet"\n'
            'EXTRACT_PROGRESS_JSONL      = LOG_DIR / "extract_progress.jsonl"\n'
            'DATA_AUDIT_PHASE1_JSON      = AUDIT_DIR / "data_audit_phase1.json"\n'
            'FEATURE_COVERAGE_BY_BLOCK_JSON = AUDIT_DIR / "feature_coverage_by_block.json"',
        )
    if "PHASE1_ROW_TIMEOUT_SEC" not in out:
        out = out.replace(
            "PHASE1_MIN_SUCCESS_RATIO      = 0.55",
            "PHASE1_MIN_SUCCESS_RATIO      = 0.55\n"
            "PHASE1_ROW_TIMEOUT_SEC        = 240\n"
            "PHASE1_ROW_TIMEOUT_HYBRID_SEC = 420\n"
            "PHASE1_MAX_RETRIES            = 3",
        )
    return out


def patch_cell9(source: str) -> str:
    out = patch_waveform_sections_only(source)
    if "def extract_block_d(" not in out:
        marker = "# ════════════════════════════════════════════════════\n#  BLOCK E — Recording Context (one per recording)\n# ════════════════════════════════════════════════════"
        out = out.replace(marker, BLOCK_D_SNIPPET + "\n\n" + marker)
    out = out.replace(
        "print('✓ Feature functions defined (Blocks A, B, C, E, F)')",
        "print('✓ Feature functions defined (Blocks A, B, C, D, E, F)')",
    )
    return out


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.cells[5].source = patch_cell5(nb.cells[5].source)
    nb.cells[9].source = patch_cell9(nb.cells[9].source)
    nb.cells[11].source = CELL11_SOURCE
    nb.cells[12].source = CELL12_SOURCE
    nbformat.write(nb, NB_PATH)
    print(f"Patched notebook: {NB_PATH}")


if __name__ == "__main__":
    main()
