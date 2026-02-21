#!/bin/bash
### ============================================================
### LSF Batch Script — Plot loss curve (one-off)
### Submit with:  bsub < batch/plot_loss_curve.sh
### ============================================================

#BSUB -J DAS_LossCurve
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=2GB]"
#BSUB -W 0:10
#BSUB -o /work3/s214374/DASProject/DASProject/logs/loss_curve.out
#BSUB -e /work3/s214374/DASProject/DASProject/logs/loss_curve.err

module load python3/3.11.11
export PYTHONUNBUFFERED=1

cd /work3/s214374/DASProject/DASProject
source .venv/bin/activate
mkdir -p logs reports/figures

> /work3/s214374/DASProject/DASProject/logs/loss_curve.out
> /work3/s214374/DASProject/DASProject/logs/loss_curve.err


python3 scripts/plot_loss_curve.py
