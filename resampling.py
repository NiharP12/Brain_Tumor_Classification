# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/resampling.py — Isotropic resampling
"""
Resample all volumes to 1mm isotropic and standard matrix (240×240×155).
  - Volumes: trilinear interpolation
  - Segmentation masks: nearest-neighbour interpolation

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.resampling --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, affine_transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def resample_volume(volume: np.ndarray, original_spacing: tuple,
                     target_spacing: tuple = (1.0, 1.0, 1.0),
                     order: int = 1) -> np.ndarray:
    """Resample a 3D volume to target spacing.

    Args:
        volume: Input 3D volume.
        original_spacing: Current voxel spacing (mm).
        target_spacing: Desired voxel spacing (mm).
        order: Interpolation order (1=trilinear, 0=nearest).

    Returns:
        Resampled volume.
    """
    zoom_factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    resampled = zoom(volume, zoom_factors, order=order, mode='constant', cval=0)
    return resampled


def pad_or_crop_to_shape(volume: np.ndarray,
                          target_shape: tuple) -> np.ndarray:
    """Pad or crop a volume to the target shape, centred on the brain.

    Args:
        volume: Input 3D volume.
        target_shape: Desired output shape (e.g., (240, 240, 155)).

    Returns:
        Volume with target shape.
    """
    current_shape = volume.shape
    output = np.zeros(target_shape, dtype=volume.dtype)

    # Compute start/end indices for both source and target
    starts_src = []
    ends_src = []
    starts_tgt = []
    ends_tgt = []

    for cs, ts in zip(current_shape, target_shape):
        if cs >= ts:
            # Need to crop: take centre portion
            start_src = (cs - ts) // 2
            starts_src.append(start_src)
            ends_src.append(start_src + ts)
            starts_tgt.append(0)
            ends_tgt.append(ts)
        else:
            # Need to pad: centre the volume
            start_tgt = (ts - cs) // 2
            starts_src.append(0)
            ends_src.append(cs)
            starts_tgt.append(start_tgt)
            ends_tgt.append(start_tgt + cs)

    output[starts_tgt[0]:ends_tgt[0],
           starts_tgt[1]:ends_tgt[1],
           starts_tgt[2]:ends_tgt[2]] = \
        volume[starts_src[0]:ends_src[0],
               starts_src[1]:ends_src[1],
               starts_src[2]:ends_src[2]]

    return output


@timer()
def run_resampling(cfg) -> None:
    """Resample all volumes to isotropic and standard matrix."""
    import pandas as pd
    logger = setup_logger("resampling", "results/logs/step02_resampling.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]

    target_spacing = tuple(cfg.target_spacing)
    target_matrix = tuple(cfg.target_matrix)
    logger.info(f"Target spacing: {target_spacing}, matrix: {target_matrix}")

    tracker = ProgressTracker(len(patient_ids), logger, "Resampling")

    for pid in patient_ids:
        processed_dir = os.path.join(cfg.resolve_path("processed_dir"), pid)
        os.makedirs(processed_dir, exist_ok=True)

        success = True

        # Resample each sequence
        for seq in cfg.mri_sequences:
            # Find best available input
            candidates = [
                os.path.join(processed_dir, f"{seq}_normalised.nii.gz"),
                os.path.join(processed_dir, f"{seq}_n4.nii.gz"),
                os.path.join(processed_dir, f"{seq}_preprocessed.nii.gz"),
            ]
            # For ADC, also check raw
            from src.utils.config import get_sequence_path
            candidates.append(get_sequence_path(cfg, pid, seq,
                                                 use_bias_corrected=False))

            input_path = None
            for c in candidates:
                if os.path.exists(c):
                    input_path = c
                    break

            if input_path is None:
                logger.warning(f"No input found for {pid}/{seq}")
                success = False
                continue

            vol_nii = nib.load(input_path)
            vol_data = vol_nii.get_fdata().astype(np.float32)
            current_spacing = vol_nii.header.get_zooms()[:3]

            # Resample to isotropic
            resampled = resample_volume(vol_data, current_spacing,
                                         target_spacing, order=1)

            # Pad/crop to standard matrix
            standardised = pad_or_crop_to_shape(resampled, target_matrix)

            # Create new affine for isotropic spacing
            new_affine = np.eye(4)
            for i in range(3):
                new_affine[i, i] = target_spacing[i]

            out_nii = nib.Nifti1Image(standardised, new_affine)
            out_path = os.path.join(processed_dir, f"{seq}_preprocessed.nii.gz")
            nib.save(out_nii, out_path)

        # Resample segmentation (nearest-neighbour)
        from src.utils.config import get_segmentation_path
        seg_path = get_segmentation_path(cfg, pid)
        seg_processed = os.path.join(processed_dir, "seg_registered.nii.gz")

        seg_input = seg_processed if os.path.exists(seg_processed) else seg_path

        if os.path.exists(seg_input):
            seg_nii = nib.load(seg_input)
            seg_data = seg_nii.get_fdata().astype(np.float32)
            seg_spacing = seg_nii.header.get_zooms()[:3]

            seg_resampled = resample_volume(seg_data, seg_spacing,
                                             target_spacing, order=0)
            seg_std = pad_or_crop_to_shape(seg_resampled, target_matrix)
            seg_std = np.round(seg_std).astype(np.uint8)

            new_affine = np.eye(4)
            for i in range(3):
                new_affine[i, i] = target_spacing[i]

            seg_out = nib.Nifti1Image(seg_std, new_affine)
            nib.save(seg_out, os.path.join(processed_dir, "seg_preprocessed.nii.gz"))
        else:
            logger.warning(f"No segmentation for {pid}")

        # Resample brain mask (nearest-neighbour)
        mask_path = os.path.join(processed_dir, "brain_mask.nii.gz")
        if not os.path.exists(mask_path):
            from src.utils.config import get_brain_mask_path
            mask_path = get_brain_mask_path(cfg, pid)

        if os.path.exists(mask_path):
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata().astype(np.float32)
            mask_spacing = mask_nii.header.get_zooms()[:3]
            mask_resampled = resample_volume(mask_data, mask_spacing,
                                              target_spacing, order=0)
            mask_std = pad_or_crop_to_shape(mask_resampled, target_matrix)
            mask_std = (mask_std > 0.5).astype(np.uint8)

            new_affine = np.eye(4)
            for i in range(3):
                new_affine[i, i] = target_spacing[i]

            mask_out = nib.Nifti1Image(mask_std, new_affine)
            nib.save(mask_out, os.path.join(processed_dir, "brain_mask.nii.gz"))

        tracker.update(success, pid)

    tracker.finish()


def main():
    parser = argparse.ArgumentParser(description="Isotropic resampling")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    run_resampling(cfg)


if __name__ == "__main__":
    main()
