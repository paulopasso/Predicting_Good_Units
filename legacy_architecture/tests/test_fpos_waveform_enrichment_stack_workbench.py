from __future__ import annotations

from crash_tests.ml_statistics_workbench.fpos_waveform_enrichment_stack import (
    FPosWaveformEnrichmentStackConfig,
)


def test_waveform_enrichment_config_resolves_output_dir() -> None:
    config = FPosWaveformEnrichmentStackConfig(run_name="waveform_enrichment_unit_test")
    assert str(config.resolved_output_dir()).endswith("waveform_enrichment_unit_test")


def test_waveform_enrichment_defaults_are_positive() -> None:
    config = FPosWaveformEnrichmentStackConfig()
    assert config.dct_components > 0
    assert config.conv_embed_dim > 0
    assert config.conv_hidden_channels > 0
