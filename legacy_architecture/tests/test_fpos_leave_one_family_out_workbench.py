from __future__ import annotations

from crash_tests.ml_statistics_workbench.fpos_leave_one_family_out import (
    FPosLeaveOneFamilyOutConfig,
)


def test_leave_one_family_config_resolves_output_dir() -> None:
    config = FPosLeaveOneFamilyOutConfig(run_name="lofo_unit_test")
    assert str(config.resolved_output_dir()).endswith("lofo_unit_test")


def test_leave_one_family_defaults_are_positive() -> None:
    config = FPosLeaveOneFamilyOutConfig()
    assert config.conv_embed_dim > 0
    assert config.conv_hidden_channels > 0
