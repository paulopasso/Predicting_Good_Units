from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from crash_tests.domain_shift_audit.config import DomainShiftAuditConfig
from crash_tests.domain_shift_audit.report import run_with_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid-vs-paired domain shift audit")
    parser.add_argument("--parquet", default="data/raw/33000_ROWS.parquet", help="Path to the main QC parquet")
    parser.add_argument("--output-root", default="crash_tests/domain_shift_audit/outputs", help="Root folder for outputs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--quiet", action="store_true", help="Reduce runner logging")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> DomainShiftAuditConfig:
    return DomainShiftAuditConfig(
        parquet_path=REPO_ROOT / args.parquet,
        output_root=REPO_ROOT / args.output_root,
        run_name=args.run_name,
        verbose=not args.quiet,
    )


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    run_with_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
