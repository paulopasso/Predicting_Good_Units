# Written Thesis

This directory is the source of truth for the bachelor thesis manuscript.

## Build

```bash
cd /Users/paulruiz/Documents/Predicting_Good_Units/written_thesis
./build_thesis.sh
```

The compiled PDF is copied to:

```text
/Users/paulruiz/Documents/Predicting_Good_Units/docs/thesis.pdf
```

## Structure

- `main.tex`: manuscript entrypoint
- `sections/title_page.tex`: title page
- `sections/abstract.tex`: abstract
- `sections/introduction.tex`: thesis motivation, problem statement, hypotheses, contributions
- `sections/litterature_review.tex`: problem space, background, and literature review
- `sections/methodology.tex`: dataset taxonomy, tools, split protocols, model families, reproducibility, robustness
- `sections/results.tex`: empirical benchmark results
- `sections/discussion.tex`: interpretation, implications, limitations, future work
- `sections/conclusions.tex`: final hypothesis resolution and takeaway
- `sections/appendix.tex`: annex material
- `references.bib`: bibliography

## Thesis-Facing Inputs and Outputs

- The manuscript cites tracked assets from `figures/` and `tables/`.
- Reproducibility metadata lives in `reproducibility/`.
- Core benchmark code lives in `src/qc_thesis/`.
- Build and audit helpers live in `scripts/`.
