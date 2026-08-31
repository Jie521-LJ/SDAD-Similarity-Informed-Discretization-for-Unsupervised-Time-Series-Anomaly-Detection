#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check-only|-c|checkonly|CheckOnly)
      CHECK_ONLY=1
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash run_reproduce.sh [--check-only]"
      exit 2
      ;;
  esac
done

MACHINES=(
  machine-1-1
  machine-1-2
  machine-1-3
  machine-1-4
  machine-1-5
  machine-1-6
  machine-1-7
  machine-2-1
  machine-2-2
  machine-2-3
)

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("${PYTHON_BIN}")
elif [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n "${CONDA_ENV_NAME}" python)
elif [[ -n "${CONDA_DEFAULT_ENV:-}" || -n "${VIRTUAL_ENV:-}" ]]; then
  PY_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY_CMD=(python3)
elif command -v conda >/dev/null 2>&1; then
  PY_CMD=(conda run -n base python)
else
  PY_CMD=(python)
fi

echo "[1/6] Checking Python dependencies..."
"${PY_CMD[@]}" -c "import importlib.util; mods=['numpy','pandas','matplotlib','sklearn','torch','tabulate','PIL']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; assert not missing, 'missing dependencies: '+', '.join(missing)+'; install with: python -m pip install -r requirements.txt'; print('dependencies_ok')"

echo "[2/6] Checking included SMD data..."
"${PY_CMD[@]}" -c "from pathlib import Path; machines=['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-2-1','machine-2-2','machine-2-3']; base=Path('repos_paper/TranAD/data/SMD'); missing=[str(base/s/(m+'.txt')) for s in ['train','test','labels'] for m in machines if not (base/s/(m+'.txt')).exists()]; assert not missing, 'missing data: '+str(missing); print('data_ok')"

DEVICE=$("${PY_CMD[@]}" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" | tail -n 1 | tr -d '\r')
echo "Selected device: ${DEVICE}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Check-only mode finished. Full reproduction was not started."
  exit 0
fi

echo "[3/6] Running SDAD VQ anomaly detection pipeline..."
"${PY_CMD[@]}" ./experiments/run_sdad_vq_pipeline.py \
  --root . \
  --machines "${MACHINES[@]}" \
  --prefix sdad_vq_light_cuda \
  --device "${DEVICE}"

echo "[4/6] Running lightweight deep baselines..."
"${PY_CMD[@]}" ./experiments/smd_light_deep_baselines.py \
  --root . \
  --machines "${MACHINES[@]}" \
  --epochs 8 \
  --device "${DEVICE}" \
  --output-dir ./experiments/results/light_deep_baselines_cuda

echo "[5/6] Running traditional baselines..."
"${PY_CMD[@]}" ./experiments/smd_baseline_experiment.py \
  --root . \
  --machines "${MACHINES[@]}" \
  --output-dir ./experiments/results

echo "[6/6] Building patent-oriented materials..."
"${PY_CMD[@]}" ./experiments/build_final_materials.py --output-dir ./deliverables/patent_materials

echo "Reproduction finished."
echo "Results:   ./experiments/results"
echo "Materials: ./deliverables/patent_materials"
