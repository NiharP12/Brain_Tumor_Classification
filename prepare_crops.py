# Environment: digital_brain_twin (CPU)
# src/step04_deep_features/prepare_crops.py — Tumour-centric 128³ crop extraction
"""
For each patient and each MRI sequence, extract a 128×128×128 tumour-centric
crop by:
  1. Computing 3D bounding box of Whole Tumour (labels 1+2+4)
  2. Extending by 16 voxels in each direction (clipped at brain boundaries)
  3. Cropping the volume
  4. Resizing to 128×128×128 using trilinear interpolation

Outputs per patient:
  data/crops_128/<patient_id>/<seq>_crop.npy
  data/crops_128/<patient_id>/crop_metadata.json

Usage:
    conda activate digital_brain_twin
    python -m src.step04_deep_features.prepare_crops --config configs/study1_config.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_sequence_path, get_segmentation_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer


def compute_tumour_bbox(seg_data: np.ndarray, wt_labels: list,
                         padding: int = 16) -> tuple:
    """Compute 3D bounding box of whole tumour with padding.

    Args:
        seg_data: Segmentation volume.
        wt_labels: Label values for Whole Tumour.
        padding: Voxels of padding around the bounding box.

    Returns:
        Tuple of (slices, original_bbox) where slices can index the volume.
    """
    wt_mask = np.isin(seg_data, wt_labels)

    if np.sum(wt_mask) == 0:
        raise ValueError("No whole tumour voxels found in segmentation")

    # Find bounding box
    coords = np.array(np.where(wt_mask))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1)

    # Add padding and clip
    shape = seg_data.shape
    padded_mins = np.maximum(mins - padding, 0)
    padded_maxs = np.minimum(maxs + padding + 1, shape)

    slices = tuple(slice(lo, hi) for lo, hi in zip(padded_mins, padded_maxs))
    original_bbox = {
        'min': mins.tolist(),
        'max': maxs.tolist(),
        'padded_min': padded_mins.tolist(),
        'padded_max': padded_maxs.tolist(),
        'tumour_voxels': int(np.sum(wt_mask)),
    }
    return slices, original_bbox


def resize_volume(volume: np.ndarray, target_shape: tuple,
                   order: int = 1) -> np.ndarray:
    """Resize a 3D volume to target shape.

    Args:
        volume: Input 3D volume.
        target_shape: Desired output shape.
        order: Interpolation order (1=trilinear, 0=nearest).

    Returns:
        Resized volume.
    """
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=order, mode='constant', cval=0)


@timer()
def run_crop_preparation(cfg) -> None:
    """Prepare 128³ tumour-centric crops for all patients."""
    import pandas as pd
    logger = setup_logger("prepare_crops", "results/logs/step04_crops.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    cohort['patient_id'] = cohort[cfg.col_patient_id].apply(normalise_patient_id)

    # Filter to only patients who successfully extracted radiomics
    radiomics_path = os.path.join(cfg.resolve_path("radiomics_dir"), "raw_features.csv")
    if os.path.exists(radiomics_path):
        import pandas as pd
        radiomics_df = pd.read_csv(radiomics_path)
        valid_ids = radiomics_df['patient_id'].values
        cohort = cohort[cohort['patient_id'].isin(valid_ids)]

    patient_ids = cohort['patient_id'].values

    target_size = tuple(cfg.crop_target_size)  # (128, 128, 128)
    padding = cfg.crop_padding_voxels
    wt_labels = cfg.roi_labels.WT

    logger.info(f"Preparing {target_size} crops for {len(patient_ids)} patients")

    crop_sizes_before = []
    errors = []

    for pid in tqdm(patient_ids, desc="Preparing crops"):
        try:
            # Load segmentation
            seg_path = os.path.join(cfg.resolve_path("processed_dir"),
                                     pid, "seg_preprocessed.nii.gz")
            if not os.path.exists(seg_path):
                seg_path = get_segmentation_path(cfg, pid)

            if not os.path.exists(seg_path):
                logger.warning(f"No segmentation for {pid}")
                errors.append(pid)
                continue

            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata()

            # Compute bounding box
            try:
                bbox_slices, bbox_info = compute_tumour_bbox(
                    seg_data, wt_labels, padding
                )
            except ValueError as e:
                logger.warning(f"No tumour for {pid}: {e}")
                errors.append(pid)
                continue

            crop_shape_before = tuple(s.stop - s.start for s in bbox_slices)
            crop_sizes_before.append(crop_shape_before)

            # Create output directory
            crop_dir = os.path.join(cfg.resolve_path("crops_dir"), pid)
            os.makedirs(crop_dir, exist_ok=True)

            # Process each sequence
            for seq in cfg.mri_sequences:
                # Find best input volume
                candidates = [
                    os.path.join(cfg.resolve_path("processed_dir"), pid,
                                  f"{seq}_preprocessed.nii.gz"),
                    os.path.join(cfg.resolve_path("processed_dir"), pid,
                                  f"{seq}_normalised.nii.gz"),
                    get_sequence_path(cfg, pid, seq,
                                      use_bias_corrected=cfg.use_existing_bias_corrected),
                    get_sequence_path(cfg, pid, seq, use_bias_corrected=False),
                ]
                vol_path = None
                for c in candidates:
                    if os.path.exists(c):
                        vol_path = c
                        break

                if vol_path is None:
                    logger.warning(f"No {seq} found for {pid}")
                    continue

                vol_nii = nib.load(vol_path)
                vol_data = vol_nii.get_fdata().astype(np.float32)

                # Handle shape mismatch between volume and segmentation
                if vol_data.shape != seg_data.shape:
                    # Adjust bbox slices to volume shape
                    adjusted_slices = tuple(
                        slice(
                            min(s.start, dim - 1),
                            min(s.stop, dim)
                        )
                        for s, dim in zip(bbox_slices, vol_data.shape)
                    )
                    crop = vol_data[adjusted_slices]
                else:
                    crop = vol_data[bbox_slices]

                # Resize to target size
                crop_resized = resize_volume(crop, target_size, order=1)

                # Verify shape
                assert crop_resized.shape == target_size, \
                    f"Expected {target_size}, got {crop_resized.shape}"

                # Save as numpy
                crop_path = os.path.join(crop_dir, f"{seq}_crop.npy")
                np.save(crop_path, crop_resized.astype(np.float32))

            # Save crop metadata
            metadata = {
                'patient_id': str(pid),
                'original_bbox': bbox_info,
                'crop_shape_before_resize': [int(x) for x in crop_shape_before],
                'target_shape': [int(x) for x in target_size],
                'scale_factors': [float(t / s) for t, s in
                                   zip(target_size, crop_shape_before)],
                'volume_shape': [int(x) for x in seg_data.shape],
            }
            meta_path = os.path.join(crop_dir, "crop_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Failed for {pid}: {e}")
            errors.append(pid)

    # Print statistics
    if crop_sizes_before:
        sizes_arr = np.array(crop_sizes_before)
        print(f"\n{'='*60}")
        print(f"Crop Preparation Summary")
        print(f"{'='*60}")
        print(f"  Patients processed: {len(patient_ids) - len(errors)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Original crop sizes (before resize):")
        for dim, name in enumerate(['X', 'Y', 'Z']):
            print(f"    {name}: min={sizes_arr[:, dim].min()}, "
                  f"max={sizes_arr[:, dim].max()}, "
                  f"mean={sizes_arr[:, dim].mean():.1f}")
        print(f"  Target size: {target_size}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Prepare 128³ tumour crops")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    run_crop_preparation(cfg)


if __name__ == "__main__":
    main()
