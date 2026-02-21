"""
Plot training and validation loss curves from saved history.

Usage:
    python scripts/plot_loss_curve.py [--history-path models/loss_history.json] [--output-dir reports/figures]

Reads the JSON file saved by train.py after each run, no model retraining needed.
"""

import argparse
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = PROJECT_ROOT / "models" / "loss_history.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "figures"

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train/val loss curves.")
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY,
                        help="Path to loss_history.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory to save the figure")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.history_path.exists():
        sys.exit(f"ERROR: Loss history not found at {args.history_path}\n"
                 f"Run the training pipeline first (python main.py)")

    with open(args.history_path) as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    best_params = history.get("best_params", {})
    loss_fn = history.get("loss_function", "unknown")
    search_method = history.get("search_method", "unknown")

    if not train_loss or not val_loss:
        sys.exit("ERROR: Loss history is empty. No epochs were recorded.")

    epochs = list(range(1, len(train_loss) + 1))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2, markersize=4)

    # Mark best epoch
    best_epoch = val_loss.index(min(val_loss)) + 1
    best_val = min(val_loss)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_val:.6f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 1, best_val + (max(val_loss) - min(val_loss)) * 0.1),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=12, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss ({loss_fn.upper()})")
    ax.set_title(f"Training Curves  |  {search_method}  |  LR={best_params.get('lr', '?'):.5f}  "
                 f"Latent={best_params.get('latent_dim', '?')}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "loss_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"  Epochs: {len(train_loss)}")
    print(f"  Best val loss: {best_val:.6f} (epoch {best_epoch})")
    print(f"  Params: {best_params}")


if __name__ == "__main__":
    main()
