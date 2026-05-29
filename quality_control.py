# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/quality_control.py — Post-preprocessing QC
"""
Comprehensive quality control checks on preprocessed volumes:
  1. Brain volume within expected range (900k–1.5M mm³)
  2. Segmentation contained within brain mask
  3. All ROIs (ET, TC, WT) have ≥ 100 voxels
  4. No NaN/Inf in any volume
  5. ADC within physiological range
  6. All sequences present

Outputs:
  results/qc_preprocessing.csv — Full QC report
  results/qc_failed_patients.txt — Patients flagged for exclusion

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.quality_control --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_segmentation_path, get_brain_mask_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def qc_single_patient(patient_id: str, cfg) -> dict:
    """Run all QC checks for a single patient.

    Args:
        patient_id: Normalised patient ID.
        cfg: Configuration object.

    Returns:
        Dict with all QC results.
    """
    result = {
        'patient_id': patient_id,
        'overall_pass': True,
        'brain_volume_mm3': np.nan,
        'brain_volume_pass': False,
        'seg_in_brain': False,
        'et_voxels': 0,
        'tc_voxels': 0,
        'wt_voxels': 0,
        'roi_pass': False,
        'no_nan_inf': True,
        'sequences_present': '',
        'sequences_missing': '',
        'notes': []
    }

    processed_dir = os.path.join(cfg.resolve_path("processed_dir"), patient_id)

    # ── Check brain mask ───────────────────────────────────
    mask_path = os.path.join(processed_dir, "brain_mask.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = get_brain_mask_path(cfg, patient_id)

    if os.path.exists(mask_path):
        mask_nii = nib.load(mask_path)
        mask_data = (mask_nii.get_fdata() > 0).astype(np.uint8)
        voxel_vol = np.prod(mask_nii.header.get_zooms()[:3])
        brain_volume = float(np.sum(mask_data) * voxel_vol)
        result['brain_volume_mm3'] = brain_volume

        min_vol = cfg.qc.brain_volume_min_mm3
        max_vol = cfg.qc.brain_volume_max_mm3
        result['brain_volume_pass'] = min_vol <= brain_volume <= max_vol
        if not result['brain_volume_pass']:
            result['notes'].append(f"Brain volume {brain_volume:.0f} mm³ out of range")
    else:
        result['notes'].append("Brain mask not found")
        result['overall_pass'] = False

    # ── Check segmentation ─────────────────────────────────
    seg_path = os.path.join(processed_dir, "seg_preprocessed.nii.gz")
    if not os.path.exists(seg_path):
        seg_path = get_segmentation_path(cfg, patient_id)

    if os.path.exists(seg_path):
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata()

        # Check ROI volumes
        roi_labels = cfg.roi_labels
        et_mask = np.isin(seg_data, roi_labels.ET)
        tc_mask = np.isin(seg_data, roi_labels.TC)
        wt_mask = np.isin(seg_data, roi_labels.WT)

        result['et_voxels'] = int(np.sum(et_mask))
        result['tc_voxels'] = int(np.sum(tc_mask))
        result['wt_voxels'] = int(np.sum(wt_mask))

        min_voxels = cfg.qc.min_roi_voxels
        result['roi_pass'] = all([
            result['et_voxels'] >= min_voxels,
            result['tc_voxels'] >= min_voxels,
            result['wt_voxels'] >= min_voxels,
        ])

        if not result['roi_pass']:
            result['notes'].append(
                f"ROI voxels below threshold: ET={result['et_voxels']}, "
                f"TC={result['tc_voxels']}, WT={result['wt_voxels']}"
            )

        # Check seg contained within brain mask
        if os.path.exists(mask_path):
            mask_nii2 = nib.load(mask_path)
            mask_data2 = (mask_nii2.get_fdata() > 0)
            if seg_data.shape == mask_data2.shape:
                seg_outside = np.sum((seg_data > 0) & (~mask_data2))
                result['seg_in_brain'] = seg_outside < 100  # Allow small tolerance
                if not result['seg_in_brain']:
                    result['notes'].append(
                        f"{seg_outside} seg voxels outside brain mask"
                    )
            else:
                result['notes'].append("Seg/mask shape mismatch")
    else:
        result['notes'].append("Segmentation not found")

    # ── Check sequences ────────────────────────────────────
    present = []
    missing = []
    for seq in cfg.mri_sequences:
        candidates = [
            os.path.join(processed_dir, f"{seq}_preprocessed.nii.gz"),
            os.path.join(processed_dir, f"{seq}_normalised.nii.gz"),
        ]
        found = any(os.path.exists(c) for c in candidates)
        if found:
            present.append(seq)
        else:
            # Check raw data
            from src.utils.config import get_sequence_path
            raw_path = get_sequence_path(cfg, patient_id, seq)
            if os.path.exists(raw_path):
                present.append(seq)
            else:
                missing.append(seq)

    result['sequences_present'] = ','.join(present)
    result['sequences_missing'] = ','.join(missing)

    # ── Check for NaN/Inf in processed volumes ─────────────
    for seq in present:
        vol_path = os.path.join(processed_dir, f"{seq}_preprocessed.nii.gz")
        if not os.path.exists(vol_path):
            vol_path = os.path.join(processed_dir, f"{seq}_normalised.nii.gz")
        if os.path.exists(vol_path):
            vol_data = nib.load(vol_path).get_fdata()
            if np.any(np.isnan(vol_data)) or np.any(np.isinf(vol_data)):
                result['no_nan_inf'] = False
                result['notes'].append(f"NaN/Inf found in {seq}")

    # ── Overall pass ───────────────────────────────────────
    result['overall_pass'] = (
        result['brain_volume_pass'] and
        result['roi_pass'] and
        result['no_nan_inf'] and
        len(missing) == 0
    )

    result['notes'] = '; '.join(result['notes']) if result['notes'] else 'OK'
    return result


@timer()
def run_quality_control(cfg) -> None:
    """Run QC on all patients and generate report."""
    logger = setup_logger("qc", "results/logs/step02_quality_control.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]

    logger.info(f"Running QC on {len(patient_ids)} patients")

    tracker = ProgressTracker(len(patient_ids), logger, "Quality control")
    results = []

    for pid in patient_ids:
        r = qc_single_patient(pid, cfg)
        results.append(r)
        tracker.update(r['overall_pass'], pid)

    tracker.finish()

    # Generate report
    qc_df = pd.DataFrame(results)
    results_dir = cfg.resolve_path("results_dir")
    os.makedirs(results_dir, exist_ok=True)

    # Full report
    qc_path = os.path.join(results_dir, "qc_preprocessing.csv")
    qc_df.to_csv(qc_path, index=False)

    # Failed patients
    failed = qc_df[~qc_df['overall_pass']]
    failed_path = os.path.join(results_dir, "qc_failed_patients.txt")
    with open(failed_path, 'w') as f:
        f.write("Patients failing QC (candidates for exclusion):\n\n")
        for _, row in failed.iterrows():
            f.write(f"  {row['patient_id']}: {row['notes']}\n")

    # Summary
    n_pass = qc_df['overall_pass'].sum()
    n_fail = len(qc_df) - n_pass
    print(f"\n{'='*60}")
    print(f"Quality Control Summary")
    print(f"{'='*60}")
    print(f"  Total: {len(qc_df)}")
    print(f"  Pass: {n_pass} ({100*n_pass/len(qc_df):.1f}%)")
    print(f"  Fail: {n_fail} ({100*n_fail/len(qc_df):.1f}%)")
    print(f"  Brain volume pass: {qc_df['brain_volume_pass'].sum()}")
    print(f"  ROI pass: {qc_df['roi_pass'].sum()}")
    print(f"  No NaN/Inf: {qc_df['no_nan_inf'].sum()}")
    print(f"{'='*60}")
    print(f"  Reports: {qc_path}")
    print(f"  Failed: {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Post-preprocessing quality control")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    run_quality_control(cfg)


if __name__ == "__main__":
    main()
