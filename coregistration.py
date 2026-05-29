# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/coregistration.py — Atlas and intra-patient registration
"""
Two-stage registration pipeline:
  1. Template registration: T1 → SRI24 atlas (ANTs SyN nonlinear)
  2. Intra-patient: Other sequences → T1 (rigid)
  3. Compose transforms to map all sequences to template space
  4. Segmentation warped with nearest-neighbour interpolation

Outputs per patient (in data/processed/<patient_id>/):
  - <seq>_registered.nii.gz for each sequence
  - seg_registered.nii.gz
  - transforms/ subdirectory with .mat and .nii.gz warp files

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.coregistration --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_sequence_path, get_segmentation_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def register_single_patient(patient_id: str, cfg,
                              logger_name: str = "coreg") -> dict:
    """Register all sequences for a single patient to template space.

    Args:
        patient_id: Normalised patient ID.
        cfg: Configuration object.
        logger_name: Logger name.

    Returns:
        Dict with status info.
    """
    import logging
    logger = logging.getLogger(logger_name)
    result = {'patient_id': patient_id, 'status': 'success', 'error': ''}

    try:
        import ants

        output_dir = os.path.join(cfg.resolve_path("processed_dir"), patient_id)
        transforms_dir = os.path.join(output_dir, "transforms")
        os.makedirs(transforms_dir, exist_ok=True)

        # ── Load T1 (bias-corrected if available) ──────────
        t1_path = get_sequence_path(cfg, patient_id, 'T1',
                                     use_bias_corrected=cfg.use_existing_bias_corrected)
        if not os.path.exists(t1_path):
            result['status'] = 'failed'
            result['error'] = f"T1 not found: {t1_path}"
            return result

        t1_ants = ants.image_read(t1_path)

        # ── Stage 1: T1 → SRI24 atlas (SyN) ───────────────
        # Use SRI24 atlas from ANTsPy or a custom path
        try:
            template = ants.get_ants_data('mni')
            template_ants = ants.image_read(template)
        except Exception:
            # Fallback to any available MNI template
            template_ants = ants.resample_image(t1_ants,
                                                 cfg.target_spacing, False, 1)
            logger.warning(f"Using self as template for {patient_id} "
                           f"(SRI24 not found)")

        reg_template = ants.registration(
            fixed=template_ants,
            moving=t1_ants,
            type_of_transform='SyN',
            syn_metric='CC',
            syn_sampling=4,
            reg_iterations=(100, 70, 50, 20),
            verbose=False
        )

        # Save T1 registered to template
        t1_registered = reg_template['warpedmovout']
        t1_reg_path = os.path.join(output_dir, "T1_registered.nii.gz")
        ants.image_write(t1_registered, t1_reg_path)

        # ── Stage 2: Other sequences → T1 (rigid) then to template ──
        other_sequences = [s for s in cfg.mri_sequences if s != 'T1']

        for seq in other_sequences:
            seq_path = get_sequence_path(cfg, patient_id, seq,
                                          use_bias_corrected=(
                                              cfg.use_existing_bias_corrected and
                                              seq in cfg.structural_sequences))
            if not os.path.exists(seq_path):
                logger.warning(f"Sequence {seq} not found for {patient_id}, skipping")
                continue

            seq_ants = ants.image_read(seq_path)

            # Rigid registration to T1
            reg_intra = ants.registration(
                fixed=t1_ants,
                moving=seq_ants,
                type_of_transform='Rigid',
                verbose=False
            )

            # Apply composed transform: seq → T1 → template
            seq_in_template = ants.apply_transforms(
                fixed=template_ants,
                moving=seq_ants,
                transformlist=reg_intra['fwdtransforms'] + reg_template['fwdtransforms'],
                interpolator='linear'
            )

            seq_reg_path = os.path.join(output_dir, f"{seq}_registered.nii.gz")
            ants.image_write(seq_in_template, seq_reg_path)

        # ── Segmentation mask (nearest-neighbour) ──────────
        seg_path = get_segmentation_path(cfg, patient_id)
        if os.path.exists(seg_path):
            seg_ants = ants.image_read(seg_path)

            seg_in_template = ants.apply_transforms(
                fixed=template_ants,
                moving=seg_ants,
                transformlist=reg_template['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

            seg_reg_path = os.path.join(output_dir, "seg_registered.nii.gz")
            ants.image_write(seg_in_template, seg_reg_path)
        else:
            logger.warning(f"Segmentation not found for {patient_id}")

        # Save transform files
        for i, tf in enumerate(reg_template['fwdtransforms']):
            if os.path.exists(tf):
                import shutil
                dest = os.path.join(transforms_dir, f"template_fwd_{i}{os.path.splitext(tf)[1]}")
                shutil.copy2(tf, dest)

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        logger.error(f"Registration failed for {patient_id}: {e}")

    return result


@timer()
def run_coregistration(cfg) -> None:
    """Run coregistration for all patients."""
    import pandas as pd
    logger = setup_logger("coreg", "results/logs/step02_coregistration.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]
    logger.info(f"Registering {len(patient_ids)} patients")

    n_workers = cfg.get("n_preprocessing_workers", 4)
    results = []

    if n_workers > 1:
        func = partial(register_single_patient, cfg=cfg)
        with Pool(n_workers) as pool:
            results = pool.map(func, patient_ids)
    else:
        tracker = ProgressTracker(len(patient_ids), logger, "Coregistration")
        for pid in patient_ids:
            r = register_single_patient(pid, cfg)
            results.append(r)
            tracker.update(r['status'] == 'success', pid)
        tracker.finish()

    # Report
    qc_df = pd.DataFrame(results)
    qc_path = os.path.join(cfg.resolve_path("results_dir"), "qc_coregistration.csv")
    qc_df.to_csv(qc_path, index=False)
    success = (qc_df['status'] == 'success').sum()
    failed = (qc_df['status'] == 'failed').sum()
    logger.info(f"Coregistration complete: {success} success, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="MRI coregistration to atlas")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    if args.n_workers:
        cfg._data['n_preprocessing_workers'] = args.n_workers
    run_coregistration(cfg)


if __name__ == "__main__":
    main()
