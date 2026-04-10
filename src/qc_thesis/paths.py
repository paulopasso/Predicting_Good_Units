from __future__ import annotations

from pathlib import Path


def get_thesis_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (
            (candidate / "src" / "qc_thesis" / "__init__.py").exists()
            and (candidate / "README.md").exists()
        ):
            return candidate
    raise FileNotFoundError("Could not find thesis root containing src/qc_thesis and README.md")


THESIS_ROOT = Path(__file__).resolve().parents[2]

PRE_WAVEFORM_LABEL = "Trust-filtered unlabeled stack"
WAVEFORM_AUGMENTED_LABEL = "Waveform-augmented stack"
REDUCED_LATENT_CONTEXT_LABEL = "Reduced-latent context"

LEGACY_MODEL_LABELS = {
    "Pre-waveform unlabeled structure": PRE_WAVEFORM_LABEL,
    "Pre-waveform unlabeled stack": PRE_WAVEFORM_LABEL,
    "Pseudo+anchor": "Trust-filtered",
    "Pre-wave": "Trust-filtered",
    "Pre-waveform": "Trust-filtered",
    "Waveform winner": WAVEFORM_AUGMENTED_LABEL,
    "Waveform-augmented structure": WAVEFORM_AUGMENTED_LABEL,
    "Reduced-latent winner": REDUCED_LATENT_CONTEXT_LABEL,
}

FEATURE_GROUP_DISPLAY = {
    "recording_context": "Recording context (session / site)",
    "spikeinterface_qc": "SpikeInterface QC (quality metrics)",
    "waveform": "Waveform / template shape",
    "source_transfer": "Source-transfer signals",
    "acg": "Autocorrelogram (temporal structure)",
    "amplitude": "Amplitude / scale",
    "other": "Other unit descriptors",
    "waveform_bins": "Waveform bins (template shape)",
    "confusability": "Confusability (overlap risk)",
    "spikeinterface": "SpikeInterface QC (quality metrics)",
    "recording_relative": "Recording-relative normalization",
}

FEATURE_GROUP_SHORT = {
    "recording_context": "Recording context",
    "spikeinterface_qc": "SpikeInterface QC",
    "waveform": "Waveform",
    "source_transfer": "Source transfer",
    "acg": "ACG",
    "amplitude": "Amplitude",
    "other": "Other",
    "waveform_bins": "Waveform bins",
    "confusability": "Confusability",
    "spikeinterface": "SpikeInterface QC",
    "recording_relative": "Recording-relative",
}


def normalize_model_label(label: object) -> str:
    text = str(label)
    return LEGACY_MODEL_LABELS.get(text, text)


def feature_group_display(group: object, *, short: bool = False) -> str:
    text = str(group)
    mapping = FEATURE_GROUP_SHORT if short else FEATURE_GROUP_DISPLAY
    return mapping.get(text, text.replace("_", " ").title())
