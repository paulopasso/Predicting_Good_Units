# Research Modeling Index

This file is a package map, not a historical path manifest.

## Capability Layout

- `qc_thesis.modeling.shared`
  - shared stack abstractions and feature-shift utilities
- `qc_thesis.modeling.specs`
  - compact dataclasses and benchmark configs migrating out of subpackages
- `qc_thesis.modeling.augmenters`
  - reusable feature engineering and recording-context augmentation
- `qc_thesis.modeling.backends`
  - reusable trainers, model registry, and evaluation helpers
- `qc_thesis.modeling.runners`
  - top-level experiment and family entrypoints for transfer and neural methods
- `qc_thesis.modeling.transfer`
  - classical transfer baselines and benchmark runner
- `qc_thesis.modeling.stacking`
  - context-first, pseudo-label, anchor, waveform, and robustness families
- `qc_thesis.modeling.neural`
  - SSL and MMD experiment families
- `qc_thesis.modeling.recipes`
  - recipe catalog and runtime API used by notebooks

## Main Recipe Families

- `legacy_transfer`
  - `qc_thesis.modeling.runners`
- `context_stack`
  - `qc_thesis.modeling.stacking.recipes`
- `anchor_stack`
  - `qc_thesis.modeling.stacking.recipes`
- `waveform_stack`
  - `qc_thesis.modeling.stacking.recipes`
- `robustness_stack`
  - `qc_thesis.modeling.stacking.recipes`
- `waveform_transfer`
  - `qc_thesis.modeling.stacking.recipes`
- `ssl_neural`
  - `qc_thesis.modeling.runners`
- `mmd_neural`
  - `qc_thesis.modeling.runners`

## Recipe Inventory

Use `build_recipe_inventory()` or `list_recipes()` from `qc_thesis` to inspect the live recipe catalog.
