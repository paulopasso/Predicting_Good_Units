from __future__ import annotations

from crash_tests.ml_statistics_workbench.fpos_waveform_multiscale_stack import (
    FPosWaveformMultiscaleStackConfig,
)


def test_multiscale_config_resolves_output_dir() -> None:
    config = FPosWaveformMultiscaleStackConfig(run_name="multiscale_waveform_unit_test")
    assert str(config.resolved_output_dir()).endswith("multiscale_waveform_unit_test")


def test_multiscale_defaults_are_positive() -> None:
    config = FPosWaveformMultiscaleStackConfig()
    assert config.multiscale_embed_dim > 0
    assert config.multiscale_hidden_channels > 0
    assert config.denoise_std >= 0.0
    assert config.derivative_loss_weight >= 0.0
