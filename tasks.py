
import logging
import torch
from pathlib import Path
from src.dasproject.utils import load_config, setup_logging
from src.dasproject.train import run_hyperparameter_search
from src.dasproject.evaluate import run_inference_and_save
from src.dasproject.model import ConvAutoencoder

def run_pipeline():
    # 1. Setup
    config = load_config("config/config.yaml")
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("--- Pipeline Started ---")

    # 2. Train (Optuna or Grid Search)
    logger.info("Step 1: Hyperparameter Search & Training")
    best_model, best_params = run_hyperparameter_search(config)

    # 3. Inference (Save residuals to HDF5)
    logger.info("Step 2: Running Inference & Stitching")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(config["paths"]["model_path"]) / "best_cae.pth"
    loaded_model = ConvAutoencoder(config, best_params["latent_dim"])
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))

    run_inference_and_save(
        loaded_model,
        config,
        config["paths"]["inference_data_path"],
        config["paths"]["reconstruction_path"],
    )

    logger.info("Pipeline Finished Successfully.")

if __name__ == "__main__":
    run_pipeline()

