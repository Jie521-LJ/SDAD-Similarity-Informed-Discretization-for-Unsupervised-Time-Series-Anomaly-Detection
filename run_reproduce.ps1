param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Machines = @(
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-1-4",
    "machine-1-5",
    "machine-1-6",
    "machine-1-7",
    "machine-2-1",
    "machine-2-2",
    "machine-2-3"
)

Write-Host "[1/6] Checking conda base Python and dependencies..."
conda run -n base python -c "import numpy, pandas, matplotlib, sklearn, torch, tabulate, PIL; print('dependencies_ok')"

Write-Host "[2/6] Checking included SMD data..."
conda run -n base python -c "from pathlib import Path; machines=['machine-1-1','machine-1-2','machine-1-3','machine-1-4','machine-1-5','machine-1-6','machine-1-7','machine-2-1','machine-2-2','machine-2-3']; base=Path('repos_paper/TranAD/data/SMD'); missing=[str(base/s/(m+'.txt')) for s in ['train','test','labels'] for m in machines if not (base/s/(m+'.txt')).exists()]; assert not missing, 'missing data: '+str(missing); print('data_ok')"

$device = (conda run -n base python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')").Trim()
Write-Host "Selected device: $device"

if ($CheckOnly) {
    Write-Host "CheckOnly mode finished. Full reproduction was not started."
    exit 0
}

Write-Host "[3/6] Running SDAD VQ anomaly detection pipeline..."
conda run -n base python .\experiments\run_sdad_vq_pipeline.py --root . --machines $Machines --prefix sdad_vq_light_cuda --device $device

Write-Host "[4/6] Running lightweight deep baselines..."
conda run -n base python .\experiments\smd_light_deep_baselines.py --root . --machines $Machines --epochs 8 --device $device --output-dir .\experiments\results\light_deep_baselines_cuda

Write-Host "[5/6] Running traditional baselines..."
conda run -n base python .\experiments\smd_baseline_experiment.py --root . --machines $Machines --output-dir .\experiments\results

Write-Host "[6/6] Building patent-oriented materials..."
conda run -n base python .\experiments\build_final_materials.py --output-dir .\deliverables\patent_materials

Write-Host "Reproduction finished."
Write-Host "Results:      .\experiments\results"
Write-Host "Materials:    .\deliverables\patent_materials"
