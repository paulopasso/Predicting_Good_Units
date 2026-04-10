#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(cd "${ROOT_DIR}/../docs" && pwd)"

cd "${ROOT_DIR}"
latexmk -pdf main_results_onward.tex
cp "build/main_results_onward.pdf" "${DOCS_DIR}/thesis_results_onward.pdf"
echo "Built ${DOCS_DIR}/thesis_results_onward.pdf"
