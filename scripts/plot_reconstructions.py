"""
Plot raw DAS data and reconstruction error for a range of files.

Usage:
    python scripts/plot_reconstructions.py <start_file> <end_file> [--raw-dir DIR] [--recon-dir DIR]

Examples:
    python scripts/plot_reconstructions.py 150005.hdf5 150105.hdf5
    python scripts/plot_reconstructions.py 150200.hdf5 150400.hdf5 --raw-dir data/raw/test --recon-dir data/reconstructions

Output:
    reports/figures/Original_<start>_<end>.png
    reports/figures/Reconstruction_<start>_<end>.png
"""

import argparse
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "test"
DEFAULT_RECON_DIR = PROJECT_ROOT / "data" / "reconstructions"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

TIME_DOWNSAMPLE = 10
SPATIAL_DOWNSAMPLE = 4

# Plot style
plt.rcParams.update({
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot stitched raw DAS data and reconstruction error."
    )
    parser.add_argument("start_file", help="First filename, e.g. 150005.hdf5")
    parser.add_argument("end_file", help="Last filename, e.g. 150105.hdf5")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                        help="Directory with raw HDF5 files")
    parser.add_argument("--recon-dir", type=Path, default=DEFAULT_RECON_DIR,
                        help="Directory with residual HDF5 files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory to save figures")
    return parser.parse_args()


def get_files_in_range(raw_dir: Path, recon_dir: Path, start: str, end: str):
    """Return sorted (raw, recon) file pairs between start and end (inclusive)."""
    all_raw = sorted(list(raw_dir.glob("*.h5")) + list(raw_dir.glob("*.hdf5")))
    all_recon = sorted(
        list(recon_dir.glob("residual_*.h5")) + list(recon_dir.glob("residual_*.hdf5"))
    )

    # Build a lookup: original name -> recon path
    recon_lookup = {}
    for rp in all_recon:
        orig_name = rp.name.replace("residual_", "")
        recon_lookup[orig_name] = rp

    # Filter raw files to the requested range (lexicographic on filename)
    raw_in_range = [f for f in all_raw if start <= f.name <= end]

    if not raw_in_range:
        sys.exit(f"ERROR: No raw files found between {start} and {end} in {raw_dir}")

    # Pair with reconstructions
    pairs = []
    for rf in raw_in_range:
        rec = recon_lookup.get(rf.name)
        if rec is not None:
            pairs.append((rf, rec))
        else:
            print(f"  WARNING: No reconstruction for {rf.name}, skipping.")

    if not pairs:
        sys.exit(f"ERROR: No matching reconstruction files found in {recon_dir}")

    print(f"Found {len(pairs)} file pairs in range [{start} → {end}]")
    return pairs


def load_and_stitch(pairs):
    """Load, downsample, and vertically stack all file pairs."""
    raw_chunks = []
    res_chunks = []
    start_time = None
    dt = dx = 0.0
    total_samples = 0

    for raw_path, recon_path in pairs:
        with h5py.File(raw_path, "r") as f_raw, h5py.File(recon_path, "r") as f_res:
            # Metadata (from first file)
            if start_time is None:
                start_time = f_raw["header/time"][()]
                dt = f_raw["header/dt"][()]
                dx = f_raw["header/dx"][()]

            # Raw data → absolute value, downsampled
            shape = f_raw["data"].shape
            raw = np.empty(shape, dtype=np.float32)
            f_raw["data"].read_direct(raw)
            raw_down = np.abs(raw[::TIME_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE])
            raw_chunks.append(raw_down)

            # Residual → downsampled
            res = f_res["data"][:]
            res_down = res[::TIME_DOWNSAMPLE, ::SPATIAL_DOWNSAMPLE]
            res_chunks.append(res_down)

            total_samples += raw_down.shape[0]

    big_raw = np.vstack(raw_chunks)
    big_res = np.vstack(res_chunks)

    # Time axis
    start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
    effective_dt = dt * TIME_DOWNSAMPLE
    end_dt = start_dt + timedelta(seconds=total_samples * effective_dt)

    # Spatial axis
    effective_dx = dx * SPATIAL_DOWNSAMPLE

    return big_raw, big_res, start_dt, end_dt, effective_dx


def add_colorbar(im, ax, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, rotation=270, labelpad=28, fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    return cbar


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_and_save(raw, res, t_start, t_end, dx, output_dir: Path, tag: str):
    """Generate and save the two figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    max_dist_km = (raw.shape[1] * dx) / 1000.0
    t0_num = mdates.date2num(t_start)
    t1_num = mdates.date2num(t_end)
    extent = [0, max_dist_km, t0_num, t1_num]
    date_fmt = mdates.DateFormatter("%H:%M")

    # ── Figure 1: Original raw data ──
    fig1, ax1 = plt.subplots(figsize=(16, 7))
    vmax_raw = np.percentile(raw, 99.5)
    im1 = ax1.imshow(raw, aspect="auto", cmap="jet", origin="lower",
                     extent=extent, vmin=0, vmax=vmax_raw)
    add_colorbar(im1, ax1, "|Strain Rate|")
    ax1.yaxis_date()
    ax1.yaxis.set_major_formatter(date_fmt)
    ax1.set_ylabel("Time (UTC)")
    ax1.set_xlabel("Distance (km)")
    ax1.set_title("Raw DAS Data")

    path1 = output_dir / f"Original_{tag}.png"
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── Figure 2: Reconstruction error ──
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    vmax_res = np.percentile(res, 99.5)
    im2 = ax2.imshow(res, aspect="auto", cmap="inferno", origin="lower",
                     extent=extent, vmin=0, vmax=vmax_res)
    add_colorbar(im2, ax2, "Reconstruction Error")
    ax2.yaxis_date()
    ax2.yaxis.set_major_formatter(date_fmt)
    ax2.set_ylabel("Time (UTC)")
    ax2.set_xlabel("Distance (km)")
    ax2.set_title("Reconstruction Error (Residual)")

    path2 = output_dir / f"Reconstruction_{tag}.png"
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"Raw dir:   {args.raw_dir}")
    print(f"Recon dir: {args.recon_dir}")
    print(f"Range:     {args.start_file} → {args.end_file}")

    pairs = get_files_in_range(args.raw_dir, args.recon_dir,
                               args.start_file, args.end_file)

    raw, res, t_start, t_end, dx = load_and_stitch(pairs)
    print(f"Stitched shape: {raw.shape}  |  Time: {t_start:%H:%M:%S} → {t_end:%H:%M:%S}")

    # Tag for filenames, e.g. "150005_150105"
    tag_start = Path(args.start_file).stem
    tag_end = Path(args.end_file).stem
    tag = f"{tag_start}_{tag_end}"

    plot_and_save(raw, res, t_start, t_end, dx, args.output_dir, tag)
    print("Done.")


if __name__ == "__main__":
    main()
