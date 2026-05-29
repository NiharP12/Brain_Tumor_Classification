# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/intensity_normalisation.py — Intensity normalisation
"""
Normalise MRI intensities per sequence type:
  - T1, T1c, T2, FLAIR: Z-score within brain mask
  - ADC: NO normalisation (retain original quantitative values)
    Verify range 0.2–3.0 × 10⁻³ mm²/s

Outputs per patient:
  data/processed/<patient_id>/<seq>_normalised.nii.gz

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.intensity_normalisation --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_sequence_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def zscore_normalise(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume within the brain mask.

    Args:
        volume: 3D MRI volume.
        mask: Binary brain mask.

    Returns:
        Normalised volume (zero mean, unit variance within mask).
    """
    brain_voxels = volume[mask > 0]
    mean_val = np.mean(brain_voxels)
    std_val = np.std(brain_voxels)

    if std_val < 1e-8:
        return np.zeros_like(volume)

    normalised = np.zeros_like(volume, dtype=np.float32)
    normalised[mask > 0] = (volume[mask > 0] - mean_val) / std_val
    return normalised


def verify_adc_range(volume: np.ndarray, mask: np.ndarray,
                      min_val: float = 0.0002,
                      max_val: float = 0.003) -> dict:
    """Verify ADC values are within physiological range.

    Args:
        volume: ADC volume.
        mask: Brain mask.
        min_val: Minimum expected ADC (mm²/s).
        max_val: Maximum expected ADC (mm²/s).

    Returns:
        Dict with verification results.
    """
    brain_voxels = volume[mask > 0]
    brain_voxels = brain_voxels[brain_voxels > 0]  # Exclude zero background

    result = {
        'median': float(np.median(brain_voxels)) if len(brain_voxels) > 0 else 0,
        'mean': float(np.mean(brain_voxels)) if len(brain_voxels) > 0 else 0,
        'min': float(np.min(brain_voxels)) if len(brain_voxels) > 0 else 0,
        'max': float(np.max(brain_voxels)) if len(brain_voxels) > 0 else 0,
        'in_range': True,
        'note': ''
    }

    # ADC values can be in different units depending on scanner
    # Common: mm²/s (×10⁻³ range) or µm²/ms (×10⁰ range)
    if result['median'] > 1.0:
        # Likely in µm²/ms or scaled units — this is common
        result['note'] = "ADC values appear to be in scaled units (>1.0)"
    elif result['median'] < min_val or result['median'] > max_val:
        result['in_range'] = False
        result['note'] = f"ADC median {result['median']:.6f} outside expected range"

    return result


@timer()
def run_intensity_normalisation(cfg) -> None:
    """Run intensity normalisation for all patients."""
    import pandas as pd
    logger = setup_logger("intensity_norm", "results/logs/step02_intensity_norm.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]

    logger.info(f"Normalising {len(patient_ids)} patients")

    tracker = ProgressTracker(len(patient_ids), logger, "Intensity normalisation")
    adc_reports = []

    for pid in patient_ids:
        processed_dir = os.path.join(cfg.resolve_path("processed_dir"), pid)
        os.makedirs(processed_dir, exist_ok=True)

        # Load brain mask
        mask_path = os.path.join(processed_dir, "brain_mask.nii.gz")
        if not os.path.exists(mask_path):
            # Try to use existing mask from raw data
            from src.utils.config import get_brain_mask_path
            mask_path = get_brain_mask_path(cfg, pid)

        if not os.path.exists(mask_path):
            logger.error(f"No brain mask for {pid}")
            tracker.update(False, pid, "No brain mask")
            continue

        mask_nii = nib.load(mask_path)
        mask = (mask_nii.get_fdata() > 0).astype(np.uint8)

        success = True

        # ── Structural sequences: Z-score ──────────────────
        for seq in cfg.structural_sequences:
            # Try preprocessed first, then raw
            input_path = os.path.join(processed_dir, f"{seq}_n4.nii.gz")
            if not os.path.exists(input_path):
                input_path = get_sequence_path(cfg, pid, seq,
                                                use_bias_corrected=cfg.use_existing_bias_corrected)
            if not os.path.exists(input_path):
                logger.warning(f"Missing {seq} for {pid}")
                success = False
                continue

            vol_nii = nib.load(input_path)
            vol_data = vol_nii.get_fdata().astype(np.float32)

            # Handle shape mismatch with mask
            if vol_data.shape != mask.shape:
                logger.warning(f"Shape mismatch for {pid}/{seq}: "
                               f"vol={vol_data.shape}, mask={mask.shape}")
                # Use the volume as-is, create a simple threshold mask
                temp_mask = (vol_data > np.percentile(vol_data[vol_data > 0], 2)).astype(np.uint8)
                normalised = zscore_normalise(vol_data, temp_mask)
            else:
                normalised = zscore_normalise(vol_data, mask)

            out_path = os.path.join(processed_dir, f"{seq}_normalised.nii.gz")
            out_nii = nib.Nifti1Image(normalised, vol_nii.affine, vol_nii.header)
            nib.save(out_nii, out_path)

        # ── ADC: NO normalisation, just verify ─────────────
        adc_path = get_sequence_path(cfg, pid, 'ADC', use_bias_corrected=False)
        if os.path.exists(adc_path):
            adc_nii = nib.load(adc_path)
            adc_data = adc_nii.get_fdata().astype(np.float32)

            # Verify ADC range
            if adc_data.shape == mask.shape:
                adc_check = verify_adc_range(adc_data, mask)
            else:
                temp_mask = (adc_data > 0).astype(np.uint8)
                adc_check = verify_adc_range(adc_data, temp_mask)

            adc_check['patient_id'] = pid
            adc_reports.append(adc_check)

            if not adc_check['in_range']:
                logger.warning(f"ADC range warning for {pid}: {adc_check['note']}")

            # Save ADC as-is (just copy/rename for consistency)
            adc_out = os.path.join(processed_dir, "ADC_normalised.nii.gz")
            nib.save(adc_nii, adc_out)

        tracker.update(success, pid)

    tracker.finish()

    # Save ADC QC report
    if adc_reports:
        adc_df = pd.DataFrame(adc_reports)
        adc_qc_path = os.path.join(cfg.resolve_path("results_dir"), "qc_adc_values.csv")
        adc_df.to_csv(adc_qc_path, index=False)
        logger.info(f"ADC QC report saved to: {adc_qc_path}")


def main():
    parser = argparse.ArgumentParser(description="Intensity normalisation")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    run_intensity_normalisation(cfg)


if __name__ == "__main__":
    main()
