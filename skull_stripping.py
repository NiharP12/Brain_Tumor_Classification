# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/skull_stripping.py — Brain mask extraction
"""
Skull stripping / brain mask generation.

ADAPTED: UCSF-PDGM v5 already provides brain segmentation masks
(brain_segmentation.nii.gz). This module either uses the existing masks
or runs HD-BET/SynthStrip from scratch as a fallback.

Outputs per patient:
  data/processed/<patient_id>/brain_mask.nii.gz

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.skull_stripping --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_brain_mask_path, get_sequence_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def process_single_patient(patient_id: str, cfg, logger_name: str = "skull_strip") -> dict:
    """Extract or generate brain mask for a single patient.

    If use_existing_brain_masks is True, copies the existing brain_segmentation.nii.gz.
    Otherwise, runs HD-BET on the T1c volume.

    Args:
        patient_id: Patient ID (normalised, e.g. 'UCSF-PDGM-0004').
        cfg: Configuration object.
        logger_name: Logger name.

    Returns:
        Dict with status, patient_id, brain_volume_mm3, method.
    """
    import logging
    logger = logging.getLogger(logger_name)
    result = {'patient_id': patient_id, 'status': 'success',
              'brain_volume_mm3': 0.0, 'method': 'existing', 'error': ''}

    try:
        output_dir = os.path.join(cfg.resolve_path("processed_dir"), patient_id)
        os.makedirs(output_dir, exist_ok=True)
        mask_output_path = os.path.join(output_dir, "brain_mask.nii.gz")

        if cfg.use_existing_brain_masks:
            # ── Use existing brain segmentation mask ────────
            existing_mask_path = get_brain_mask_path(cfg, patient_id)
            if not os.path.exists(existing_mask_path):
                result['status'] = 'failed'
                result['error'] = f"Existing brain mask not found: {existing_mask_path}"
                return result

            # Load and binarise (ensure it's 0/1)
            mask_nii = nib.load(existing_mask_path)
            mask_data = mask_nii.get_fdata()
            binary_mask = (mask_data > 0).astype(np.uint8)

            # Save binary mask
            mask_out = nib.Nifti1Image(binary_mask, mask_nii.affine, mask_nii.header)
            nib.save(mask_out, mask_output_path)
            result['method'] = 'existing'

        else:
            # ── Run HD-BET skull stripping ──────────────────
            t1c_path = get_sequence_path(cfg, patient_id, 'T1c',
                                          use_bias_corrected=True)
            if not os.path.exists(t1c_path):
                result['status'] = 'failed'
                result['error'] = f"T1c not found: {t1c_path}"
                return result

            try:
                from HD_BET.run import run_hd_bet
                run_hd_bet(t1c_path,
                           os.path.join(output_dir, "T1c_brain.nii.gz"),
                           mode='fast', device='cpu', postprocess=True,
                           do_tta=False)
                # Load generated mask
                mask_path_hdbet = os.path.join(output_dir, "T1c_brain_mask.nii.gz")
                if os.path.exists(mask_path_hdbet):
                    os.rename(mask_path_hdbet, mask_output_path)
                    result['method'] = 'HD-BET'
                else:
                    raise FileNotFoundError("HD-BET did not produce mask")

            except Exception as e_hdbet:
                logger.warning(f"HD-BET failed for {patient_id}: {e_hdbet}. "
                               f"Trying SynthStrip...")
                try:
                    # SynthStrip fallback
                    import subprocess
                    subprocess.run([
                        'mri_synthstrip', '-i', t1c_path,
                        '-m', mask_output_path
                    ], check=True, capture_output=True)
                    result['method'] = 'SynthStrip'
                except Exception as e_synth:
                    result['status'] = 'failed'
                    result['error'] = f"Both HD-BET and SynthStrip failed: {e_synth}"
                    return result

        # ── Compute brain volume for QC ────────────────────
        mask_nii = nib.load(mask_output_path)
        mask_data = mask_nii.get_fdata()
        voxel_vol = np.prod(mask_nii.header.get_zooms()[:3])
        brain_volume = float(np.sum(mask_data > 0) * voxel_vol)
        result['brain_volume_mm3'] = brain_volume

        # QC check
        min_vol = cfg.qc.brain_volume_min_mm3
        max_vol = cfg.qc.brain_volume_max_mm3
        if brain_volume < min_vol or brain_volume > max_vol:
            result['status'] = 'qc_warning'
            result['error'] = (f"Brain volume {brain_volume:.0f} mm³ outside "
                               f"expected range [{min_vol}, {max_vol}]")

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


@timer()
def run_skull_stripping(cfg) -> None:
    """Run skull stripping for all patients in the filtered cohort.

    Args:
        cfg: Configuration object.
    """
    import pandas as pd
    logger = setup_logger("skull_strip", "results/logs/step02_skull_strip.log")

    # Load patient list
    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]
    logger.info(f"Processing {len(patient_ids)} patients")

    # Process patients
    n_workers = cfg.get("n_preprocessing_workers", 4)
    results = []

    if n_workers > 1:
        logger.info(f"Using {n_workers} parallel workers")
        func = partial(process_single_patient, cfg=cfg)
        with Pool(n_workers) as pool:
            results = pool.map(func, patient_ids)
    else:
        tracker = ProgressTracker(len(patient_ids), logger, "Skull stripping")
        for pid in patient_ids:
            result = process_single_patient(pid, cfg)
            results.append(result)
            tracker.update(result['status'] == 'success', pid)
        tracker.finish()

    # Generate QC report
    import pandas as pd
    qc_df = pd.DataFrame(results)
    qc_path = os.path.join(cfg.resolve_path("results_dir"), "qc_skull_stripping.csv")
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    qc_df.to_csv(qc_path, index=False)

    # Summary
    success = (qc_df['status'] == 'success').sum()
    warnings = (qc_df['status'] == 'qc_warning').sum()
    failed = (qc_df['status'] == 'failed').sum()
    logger.info(f"Skull stripping complete: {success} success, "
                f"{warnings} QC warnings, {failed} failed")
    print(f"\n✓ QC report saved to: {qc_path}")


def main():
    parser = argparse.ArgumentParser(description="Skull stripping / brain mask extraction")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Override number of parallel workers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    if args.n_workers:
        cfg._data['n_preprocessing_workers'] = args.n_workers
    run_skull_stripping(cfg)


if __name__ == "__main__":
    main()
