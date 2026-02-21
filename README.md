# DAS_Project

Distributed Acoustic Sensing (DAS) Pipeline for Unsupervised Learning and Transfer Learning

## Overview
This repository contains a modular pipeline for processing, training, and evaluating models on DAS data, with a focus on unsupervised learning (e.g., convolutional autoencoders) and transfer learning across different cable installations.

## Features
- **Preprocessing:** Channel baseline removal, log normalization, patch extraction
- **Training:** Hyperparameter search, model training (CAE)
- **Inference:** Residual calculation and reconstruction
- **Metadata Tracking:** Automatic extraction and storage of key cable and acquisition parameters
- **Batch Scripts:** Easy-to-use scripts for running the full pipeline on HPC clusters

## Structure
- `src/` — Core source code
- `scripts/` — Utility and plotting scripts
- `models/` — Saved models
- `reports/` — Figures and results
- `data/`, `logs/`, `notebooks/`, `batch/`, `config/` — Project scaffolding (contents ignored in git)

## Usage
1. Configure your experiment in `config/config.yaml`
2. Run the pipeline: `python3 tasks.py` or submit the batch script
3. Find results in `models/`, `reports/`, and metadata in `data/das_metadata.json`

## Requirements
- Python 3.11+
- PyTorch, NumPy, h5py, and other dependencies (see `requirements.txt`)

## License
MIT License
