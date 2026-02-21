#!/bin/bash
### ============================================================
### LSF Batch Script — One-off plotting job
### 
### Usage:
###   bsub < batch/plot.sh
###
### Edit START_FILE and END_FILE below before submitting.
### ============================================================

#BSUB -J DAS_Plot
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -o /work3/s214374/DASProject/DASProject/logs/plot.out
#BSUB -e /work3/s214374/DASProject/DASProject/logs/plot.err

### ── EDIT THESE ─────────────────────────────────────────────
START_FILE="150005.hdf5"
END_FILE="150605.hdf5"
### ───────────────────────────────────────────────────────────

### --- Environment ---
module load python3/3.11.11
export PYTHONUNBUFFERED=1

cd /work3/s214374/DASProject/DASProject
source .venv/bin/activate
mkdir -p logs reports/figures

> /work3/s214374/DASProject/DASProject/logs/plot.out
> /work3/s214374/DASProject/DASProject/logs/plot.err


### --- Run ---
echo "=== Plotting $START_FILE → $END_FILE ==="
python3 scripts/plot_reconstructions.py "$START_FILE" "$END_FILE"
echo "=== Done ==="
