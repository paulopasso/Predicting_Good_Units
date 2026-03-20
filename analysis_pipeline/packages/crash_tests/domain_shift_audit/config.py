from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DomainShiftAuditConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    output_root: Path = Path("crash_tests/domain_shift_audit/outputs")
    run_name: str | None = None
    random_state: int = 42
    paired_final_holdout_rows: int = 100
    top_n_features: int = 30
    max_domain_rows_for_classifier: int = 5000
    pca_rows_per_domain: int = 1500
    include_pca_3d: bool = True
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"domain_shift_audit_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload
