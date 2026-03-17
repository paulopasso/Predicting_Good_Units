from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SpikeForestAuditConfig:
    parquet_path: Path = Path("data/raw/33000_ROWS.parquet")
    extraction_notebook: Path = Path("notebooks/Units_Spikeforest_Extraction_Colab.ipynb")
    output_root: Path = Path("crash_tests/spikeforest_audit/outputs")
    run_name: str | None = None
    paired_prefix: str = "PAIRED_"
    common_sorter_fraction: float = 0.5
    verbose: bool = True

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"spikeforest_audit_{stamp}"

    def resolved_output_dir(self) -> Path:
        return self.output_root / self.resolved_run_name()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parquet_path"] = str(self.parquet_path)
        payload["extraction_notebook"] = str(self.extraction_notebook)
        payload["output_root"] = str(self.output_root)
        payload["resolved_output_dir"] = str(self.resolved_output_dir())
        return payload
