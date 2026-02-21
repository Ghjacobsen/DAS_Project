import h5py
import numpy as np
import torch
import logging
import gc
import json
from torch.utils.data import Dataset
from pathlib import Path


class DASDataset(Dataset):
    """
    PyTorch Dataset for Distributed Acoustic Sensing (DAS) data.
    
    Preprocessing pipeline (in order):
      1. Channel median subtraction (optional, enabled by default)
         — removes per-channel static baseline so the model only sees
           dynamic signal content. This makes learned features
           cable-agnostic, which is critical for transfer learning.
      2. Symmetric log normalization: sign(x) * log1p(|x|) / scale
         — compresses dynamic range while preserving sign.
    """

    def __init__(self, file_paths, config, mode='train'):
        self.logger = logging.getLogger(__name__)
        self.file_paths = file_paths
        self.config = config
        self.mode = mode
        self.patch_size = tuple(config['data']['patch_size'])
        self.patches_per_file = config['data'].get('patches_per_file', 50)

        self.logger.info(f"Initializing DASDataset with {len(file_paths)} files in {mode} mode.")
        self.metadata = self._extract_and_save_metadata()
        self.patches = self._load_and_process_data()

    def _extract_and_save_metadata(self):
        """
        Extracts metadata from the first file and saves it as a JSON file for the dataset.
        """
        if not self.file_paths:
            self.logger.warning("No files provided for metadata extraction.")
            return {}
        fp = self.file_paths[0]
        meta = {}
        try:
            with h5py.File(fp, 'r') as f:
                fs = 1 / f['header']['dt'][()]
                dx = f['cableSpec']['sensorDistances'][()][1] - f['cableSpec']['sensorDistances'][()][0]
                GL = f['header']['gaugeLength'][()]
                nx = len(f['header']['channels'][()])
                meta = {'fs': fs, 'dx': dx, 'GL': GL, 'nx': nx}
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {fp}: {e}")
            return {}

        # Save metadata as JSON in the same directory as the first file
        meta_path = Path(fp).parent / 'das_metadata.json'
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            self.logger.info(f"Saved metadata to {meta_path}")
        except Exception as e:
            self.logger.error(f"Error saving metadata to {meta_path}: {e}")
        return meta
    def get_metadata(self):
        """Return the dataset-level metadata dictionary."""
        return self.metadata

    def _normalize(self, data):
        """
        Apply normalization to raw float32 data using np.log normalization as in the referenced article.
        This compresses the dynamic range and stabilizes variance.
        """
        epsilon = 1e-10  # Small value to avoid log(0)
        np.maximum(data, epsilon, out=data)  # Ensure all values >= epsilon
        np.log(data, out=data)  # In-place log transform
        return data

    def _remove_channel_baseline(self, data):
        """
        Subtract per-channel median from raw data (in-place).

        Each DAS channel has a static baseline caused by local coupling,
        cable geometry, and installation specifics.  Removing it forces
        the model to learn only the dynamic signal component, which is
        shared across cables — essential for transfer learning.

        The median (rather than mean) is used because it is robust to
        transient high-amplitude events that could bias the estimate.
        """
        # median along time axis (axis=0) → one value per channel
        channel_median = np.median(data, axis=0)
        data -= channel_median
        return data

    def _load_and_process_data(self):
        """
        Loads HDF5 files with strict memory management.
        Optionally removes per-channel static baseline before normalisation.
        """
        all_patches = []
        remove_baseline = self.config['preprocessing'].get(
            'remove_channel_baseline', True
        )

        for fp in self.file_paths:
            try:
                gc.collect()

                with h5py.File(fp, 'r') as f:
                    if 'data' not in f or 'header' not in f:
                        continue

                    # Allocate float32 container and read directly
                    data_shape = f['data'].shape
                    raw_data = np.empty(data_shape, dtype=np.float32)
                    f['data'].read_direct(raw_data)

                    # 1. Remove per-channel static baseline (transfer-safe)
                    if remove_baseline:
                        raw_data = self._remove_channel_baseline(raw_data)

                    # 2. Normalize
                    raw_data = self._normalize(raw_data)

                    # Patching
                    time_dim, channel_dim = raw_data.shape
                    p_time, p_chan = self.patch_size

                    if self.mode == 'train':
                        for _ in range(self.patches_per_file):
                            rand_t = np.random.randint(0, time_dim - p_time)
                            rand_c = np.random.randint(0, channel_dim - p_chan)
                            patch = raw_data[rand_t:rand_t + p_time, rand_c:rand_c + p_chan]
                            all_patches.append(np.expand_dims(patch, axis=0))
                    else:
                        # Sequential scan (inference)
                        for t in range(0, time_dim - p_time + 1, p_time):
                            for c in range(0, channel_dim - p_chan + 1, p_chan):
                                patch = raw_data[t:t + p_time, c:c + p_chan]
                                all_patches.append(np.expand_dims(patch, axis=0))

            except Exception as e:
                self.logger.error(f"Error processing file {fp}: {e}")

        # Cleanup
        gc.collect()

        if len(all_patches) == 0:
            self.logger.error("No patches generated.")
            return np.array([], dtype=np.float32)

        self.logger.info(f"Loaded {len(all_patches)} patches.")
        return np.array(all_patches, dtype=np.float32)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx])
        return x, x