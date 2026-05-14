#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CHECKPOINT="${CHECKPOINT:-checkpoints/ps_7.pt}"
IMAGE="${IMAGE:-data/0844x2.png}"
OUTPUT="${OUTPUT:-output}"
CONDA_ENV="${CONDA_ENV:-AIR}"
USE_CONDA="${USE_CONDA:-1}"

usage() {
    echo "Usage: $0 [--checkpoint PATH] [--image PATH] [--output DIR] [--no-conda]"
    echo
    echo "Defaults:"
    echo "  --checkpoint ${CHECKPOINT}"
    echo "  --image      ${IMAGE}"
    echo "  --output     ${OUTPUT}"
    echo
    echo "Environment overrides:"
    echo "  CONDA_ENV=<env>   Conda environment name, default: AIR"
    echo "  USE_CONDA=0       Run with the current python instead of conda run"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --image|--input)
            IMAGE="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --no-conda)
            USE_CONDA=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${USE_CONDA}" == "1" ]] && command -v conda >/dev/null 2>&1; then
    conda run -n "${CONDA_ENV}" python test/infer.py \
        --checkpoint "${CHECKPOINT}" \
        --image "${IMAGE}" \
        --output "${OUTPUT}"
else
    python test/infer.py \
        --checkpoint "${CHECKPOINT}" \
        --image "${IMAGE}" \
        --output "${OUTPUT}"
fi
