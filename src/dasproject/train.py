import json
import copy
import logging
import itertools
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.exceptions import TrialPruned

from src.dasproject.data import DASDataset
from src.dasproject.model import ConvAutoencoder
from src.dasproject.loss import get_loss_function
from src.dasproject.utils import get_device


# ── Epoch helpers ───────────────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one epoch of training."""
    model.train()
    running_loss = 0.0
    for img, _ in dataloader:
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Evaluates model on validation set."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            running_loss += loss.item()
    return running_loss / len(dataloader)


# ── Loss history I/O ────────────────────────────────────────────────────────
def _save_loss_history(history, config):
    """Save loss history dict to JSON for later plotting."""
    path = Path(config["paths"].get("loss_history_path", "models/loss_history.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    logging.getLogger(__name__).info(f"Loss history saved to {path}")


# ── Single training run (used by both Optuna and grid) ──────────────────────
def _train_single(config, train_loader, val_loader, device, lr, latent_dim,
                  trial=None):
    """
    Train one model configuration. Returns (val_loss, model_state, epoch_log).
    
    If `trial` is an Optuna Trial, reports intermediate values and supports
    pruning of unpromising trials.
    """
    logger = logging.getLogger(__name__)
    criterion = get_loss_function(config)
    model = ConvAutoencoder(config, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 5)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    epoch_log = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        t_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss = validate(model, val_loader, criterion, device)

        epoch_log["train_loss"].append(t_loss)
        epoch_log["val_loss"].append(v_loss)

        print(f"  Epoch {epoch + 1}/{epochs}: Train={t_loss:.6f}  Val={v_loss:.6f}")

        # Early stopping
        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Optuna pruning
        if trial is not None:
            trial.report(v_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

    return best_val, best_state, epoch_log


# ── Optuna objective ────────────────────────────────────────────────────────
def _make_objective(config, train_loader, val_loader, device, results_container):
    """Returns an Optuna objective function (closure over data loaders)."""

    def objective(trial):
        hp = config["hyperparameter_search"]
        lr = trial.suggest_float("lr", hp["lr_min"], hp["lr_max"], log=True)
        latent_dim = trial.suggest_int(
            "latent_dim", hp["latent_dim_min"], hp["latent_dim_max"], step=32
        )

        print(f"\n--- Trial {trial.number}: LR={lr:.6f}, Latent={latent_dim} ---")

        val_loss, state, epoch_log = _train_single(
            config, train_loader, val_loader, device, lr, latent_dim, trial=trial
        )

        # Keep track of best
        if val_loss < results_container["best_val"]:
            results_container["best_val"] = val_loss
            results_container["best_state"] = state
            results_container["best_params"] = {"lr": lr, "latent_dim": latent_dim}
            results_container["best_epoch_log"] = epoch_log
            print(f"--> New best (Val Loss: {val_loss:.6f})")

        return val_loss

    return objective


# ── Public entry point ──────────────────────────────────────────────────────
def run_hyperparameter_search(config):
    """
    Orchestrates training with either Optuna or Grid Search.

    Returns:
        best_model: The model with lowest validation loss.
        best_params: Dict with 'lr' and 'latent_dim'.
    """
    logger = logging.getLogger(__name__)
    device = get_device()
    print(f"Device: {device}")

    # 1. Prepare Data — FILE-LEVEL split (no data leakage)
    raw_path = Path(config["paths"]["raw_data_path"])
    all_files = sorted(list(raw_path.glob("*.hdf5")) + list(raw_path.glob("*.h5")))
    print(f"Found {len(all_files)} training files.")

    # Shuffle files deterministically, then split
    rng = torch.Generator().manual_seed(config["training"]["seed"])
    indices = torch.randperm(len(all_files), generator=rng).tolist()
    split_idx = int(config["data"]["train_val_split"] * len(all_files))

    train_files = [all_files[i] for i in indices[:split_idx]]
    val_files = [all_files[i] for i in indices[split_idx:]]
    print(f"File-level split: {len(train_files)} train / {len(val_files)} val")

    train_dataset = DASDataset(train_files, config, mode="train")
    val_dataset = DASDataset(val_files, config, mode="train")

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    method = config["hyperparameter_search"].get("method", "optuna")

    # Shared results container
    results = {
        "best_val": float("inf"),
        "best_state": None,
        "best_params": {},
        "best_epoch_log": {},
    }

    # 2. Run search
    if method == "optuna":
        n_trials = config["hyperparameter_search"].get("n_trials", 20)
        objective = _make_objective(config, train_loader, val_loader, device, results)

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        )
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Optuna finished. Best trial: {study.best_trial.number}")
        logger.info(f"  Params: {study.best_trial.params}")
        logger.info(f"  Val Loss: {study.best_trial.value:.6f}")

    elif method == "grid":
        lrs = config["hyperparameter_search"]["learning_rates"]
        latents = config["hyperparameter_search"]["latent_dims"]

        for lr, latent_dim in itertools.product(lrs, latents):
            print(f"\n--- Grid: LR={lr}, Latent={latent_dim} ---")
            val_loss, state, epoch_log = _train_single(
                config, train_loader, val_loader, device, lr, latent_dim
            )
            if val_loss < results["best_val"]:
                results["best_val"] = val_loss
                results["best_state"] = state
                results["best_params"] = {"lr": lr, "latent_dim": latent_dim}
                results["best_epoch_log"] = epoch_log
                print(f"--> New best (Val Loss: {val_loss:.6f})")
    else:
        raise ValueError(f"Unknown search method: {method}")

    # 3. Save best model
    save_path = Path(config["paths"]["model_path"]) / "best_cae.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_params = results["best_params"]
    final_model = ConvAutoencoder(config, best_params["latent_dim"])
    final_model.load_state_dict(results["best_state"])
    torch.save(final_model.state_dict(), save_path)

    print(f"\nBest model saved to {save_path} with params: {best_params}")
    logger.info(f"Best model saved to {save_path} | params: {best_params}")

    # 4. Save loss history for plotting
    history = {
        "best_params": best_params,
        "best_val_loss": results["best_val"],
        "train_loss": results["best_epoch_log"].get("train_loss", []),
        "val_loss": results["best_epoch_log"].get("val_loss", []),
        "loss_function": config["training"].get("loss_function", "l1"),
        "search_method": method,
    }
    _save_loss_history(history, config)

    return final_model, best_params