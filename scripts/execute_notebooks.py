#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient
try:
    from nbformat import MissingIDFieldWarning
except Exception:  # pragma: no cover
    class MissingIDFieldWarning(Warning):
        pass


NOTEBOOKS = [
    "01_data_extraction_and_audit.ipynb",
    "02_domain_shift_hybrid_vs_paired.ipynb",
    "03_paired_family_structure.ipynb",
    "04_fpos_model_progression.ipynb",
    "05_fmiss_model_progression.ipynb",
    "06_fpos_vs_fmiss_xai.ipynb",
    "07_limitations_and_data_budget.ipynb",
    "08_appendix_additional_ablations.ipynb",
]


def main() -> int:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    thesis_root = Path(__file__).resolve().parents[1]
    notebooks_dir = thesis_root / "notebooks"
    out_dir = thesis_root / "artifacts/executed_notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.only or NOTEBOOKS

    for name in targets:
        src = notebooks_dir / name
        nb = nbformat.read(src, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3")
        client.execute(cwd=str(notebooks_dir))
        dst = out_dir / name
        nbformat.write(nb, dst)
        print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
