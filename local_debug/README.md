# Local Debug Pipeline

This folder contains a simplified, local-first version of the Spike QC workflow.

## Goal
- Run a smaller but high-signal pipeline on your own machine.
- Keep code easier to debug than the Colab production notebook.
- Preserve core flow: manifest enumeration -> preflight/extraction -> audit -> model training.

## Included
- `notebooks/Spike_QC_Local_Debug.ipynb`: simplified end-to-end notebook.
- `environment.yml`: pinned conda environment for local reproducibility.
- `artifacts/`: generated outputs (parquets, models, reports).
- `cache/`: kachery/spikeforest temporary downloads.
- `logs/`: run logs.
- `data/`: optional local inputs.

## Quick Start
1. Create and activate env:
```bash
conda env create -f local_debug/environment.yml
conda activate spike-qc-local
```

2. Register kernel (optional):
```bash
python -m ipykernel install --user --name spike-qc-local --display-name "spike-qc-local"
```

3. Launch Jupyter and open:
- `local_debug/notebooks/Spike_QC_Local_Debug.ipynb`

## Default Scope (smaller than full Colab)
- Focused on hybrid-heavy + paired sets.
- Limits rows/sorters for debug speed and stability.
- Writes checkpointed parts so reruns skip completed work.

## Notes
- This notebook is intentionally simpler for debugging, not feature-complete parity with `Spike_UNIT_Quality.ipynb`.
- Once stable, fixes can be promoted back to the main notebook.
