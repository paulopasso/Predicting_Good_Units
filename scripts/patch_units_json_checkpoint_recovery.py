from pathlib import Path
from textwrap import dedent

import nbformat


NB_PATH = Path("/Users/paulruiz/Documents/Predicting_Good_Units/notebooks/Units_Spikeforest_Extraction_Colab.ipynb")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found for replacement:\n{old[:240]}")
    return text.replace(old, new, 1)


def patch_runtime_cell(src: str) -> str:
    old_read_json = dedent(
        """
        def read_json(path, default=None):

            path = Path(path)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
            return default
        """
    ).strip()
    new_read_json = dedent(
        """
        def _quarantine_corrupt_json(path, reason):
            path = Path(path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_path = path.with_name(f"{path.stem}.corrupt_{stamp}{path.suffix}")
            try:
                path.replace(quarantine_path)
                print(f"⚠ JSON recovery: moved {path} -> {quarantine_path} ({reason})")
            except Exception as exc:
                print(f"⚠ JSON recovery: could not quarantine {path} ({reason}): {exc}")
            return quarantine_path


        def read_json(path, default=None):

            path = Path(path)
            if not path.exists():
                return default

            try:
                raw_text = path.read_text()
            except Exception as exc:
                print(f"⚠ JSON read warning for {path}: {exc}. Using default.")
                return default

            if not raw_text.strip():
                _quarantine_corrupt_json(path, "empty file")
                return default

            try:
                return json.loads(raw_text)
            except json.JSONDecodeError as exc:
                _quarantine_corrupt_json(path, f"invalid JSON: {exc}")
                return default
            except Exception as exc:
                print(f"⚠ JSON read warning for {path}: {exc}. Using default.")
                return default
        """
    ).strip()
    if old_read_json in src:
        src = replace_once(src, old_read_json, new_read_json)

    old_write_json = dedent(
        """
        def write_json(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        """
    ).strip()
    new_write_json = dedent(
        """
        def write_json(path, payload):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}")
            try:
                with open(tmp_path, "w") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        """
    ).strip()
    if old_write_json in src:
        src = replace_once(src, old_write_json, new_write_json)

    return src


def patch_phase0_cell(src: str) -> str:
    old_cached_manifest = dedent(
        """
        def _load_cached_manifest_df():
            if not SPIKEFOREST_MANIFEST.exists():
                return pd.DataFrame()
            with open(SPIKEFOREST_MANIFEST) as f:
                df = pd.DataFrame(json.load(f))
            if df.empty or "study_set" not in df.columns:
                return df
        """
    ).strip()
    new_cached_manifest = dedent(
        """
        def _load_cached_manifest_df():
            if not SPIKEFOREST_MANIFEST.exists():
                return pd.DataFrame()
            payload = read_json(SPIKEFOREST_MANIFEST, default=[])
            df = pd.DataFrame(payload)
            if df.empty or "study_set" not in df.columns:
                return df
        """
    ).strip()
    if old_cached_manifest in src:
        src = replace_once(src, old_cached_manifest, new_cached_manifest)
    return src


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)

    patched = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        original = src
        if "def read_json(path, default=None):" in src and "def write_json(path, payload):" in src:
            src = patch_runtime_cell(src)
        if "def _load_cached_manifest_df():" in src and "SPIKEFOREST_MANIFEST" in src:
            src = patch_phase0_cell(src)
        if src != original:
            cell["source"] = src
            patched += 1

    if patched == 0:
        raise RuntimeError("No notebook cells were patched.")

    nbformat.write(nb, NB_PATH)
    print(f"Patched {patched} notebook cell(s) in {NB_PATH}")


if __name__ == "__main__":
    main()
