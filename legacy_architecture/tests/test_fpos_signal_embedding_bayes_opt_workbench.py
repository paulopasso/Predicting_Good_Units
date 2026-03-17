from __future__ import annotations

from crash_tests.ml_statistics_workbench.fpos_signal_embedding_bayes_opt import (
    FPosSignalEmbeddingBayesOptConfig,
)


def test_bayes_opt_config_resolves_output_dir() -> None:
    config = FPosSignalEmbeddingBayesOptConfig(run_name="bayes_opt_unit_test")
    assert str(config.resolved_output_dir()).endswith("bayes_opt_unit_test")


def test_bayes_opt_defaults_are_valid() -> None:
    config = FPosSignalEmbeddingBayesOptConfig()
    assert config.n_trials > 0
    assert config.timeout_seconds is None or config.timeout_seconds > 0
