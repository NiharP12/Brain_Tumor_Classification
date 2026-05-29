# Environment: digital_brain_twin (CPU)
# src/step02_mri_preprocessing/bias_field_correction.py — N4 bias field correction
"""
Apply N4 bias field correction using SimpleITK.

ADAPTED: UCSF-PDGM v5 already provides bias-corrected images (_bias.nii.gz).
When use_existing_bias_corrected=True, this simply copies the existing files.

Rules:
  - Apply N4 to: T1, T1c, T2, FLAIR (structural sequences)
  - DO NOT apply to: ADC (quantitative — would distort values)

Outputs per patient:
  data/processed/<patient_id>/<seq>_n4.nii.gz

Usage:
    conda activate digital_brain_twin
    python -m src.step02_mri_preprocessing.bias_field_correction --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import shutil
import numpy as np
import SimpleITK as sitk
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_sequence_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer, ProgressTracker


def apply_n4_correction(input_path: str, mask_path: str,
                         output_path: str) -> bool:
    """Apply N4 bias field correction to a single volume.

    Args:
        input_path: Path to input NIfTI volume.
        mask_path: Path to brain mask for correction within brain only.
        output_path: Path to save corrected volume.

    Returns:
        True if successful.
    """
    try:
        image = sitk.ReadImage(input_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # Ensure mask matches image dimensions
        if mask.GetSize() != image.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask = resampler.Execute(mask)

        # N4 bias field correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
        corrector.SetConvergenceThreshold(0.001)

        corrected = corrector.Execute(image, mask)
        sitk.WriteImage(corrected, output_path)
        return True

    except Exception as e:
        print(f"  N4 correction failed: {e}")
        return False


@timer()
def run_bias_correction(cfg) -> None:
    """Run bias field correction for all patients.

    If use_existing_bias_corrected is True, copies existing _bias.nii.gz files
    instead of running N4 from scratch.
    """
    import pandas as pd
    logger = setup_logger("bias_correct", "results/logs/step02_bias_correction.log")

    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    from src.step01_clinical_data.filter_cohort import normalise_patient_id
    patient_ids = [normalise_patient_id(pid) for pid in cohort[cfg.col_patient_id]]

    structural_seqs = cfg.structural_sequences  # ['T1', 'T1c', 'T2', 'FLAIR']
    quant_seqs = cfg.quantitative_sequences     # ['ADC']
    logger.info(f"Processing {len(patient_ids)} patients")
    logger.info(f"Structural (N4/copy): {structural_seqs}")
    logger.info(f"Quantitative (skip): {quant_seqs}")

    tracker = ProgressTracker(len(patient_ids), logger, "Bias correction")

    for pid in patient_ids:
        output_dir = os.path.join(cfg.resolve_path("processed_dir"), pid)
        os.makedirs(output_dir, exist_ok=True)

        success = True
        for seq in structural_seqs:
            output_path = os.path.join(output_dir, f"{seq}_n4.nii.gz")

            if cfg.use_existing_bias_corrected:
                # Use existing bias-corrected file
                bias_path = get_sequence_path(cfg, pid, seq,
                                               use_bias_corrected=True)
                if os.path.exists(bias_path):
                    shutil.copy2(bias_path, output_path)
                else:
                    # Fallback: try without bias suffix
                    raw_path = get_sequence_path(cfg, pid, seq,
                                                  use_bias_corrected=False)
                    if os.path.exists(raw_path):
                        shutil.copy2(raw_path, output_path)
                        logger.warning(f"No _bias file for {pid}/{seq}, "
                                       f"using raw")
                    else:
                        logger.error(f"No file found for {pid}/{seq}")
                        success = False
            else:
                # Run N4 from scratch
                raw_path = get_sequence_path(cfg, pid, seq,
                                              use_bias_corrected=False)
                mask_path = os.path.join(output_dir, "brain_mask.nii.gz")
                if os.path.exists(raw_path) and os.path.exists(mask_path):
                    ok = apply_n4_correction(raw_path, mask_path, output_path)
                    if not ok:
                        success = False
                else:
                    logger.warning(f"Missing input for N4: {pid}/{seq}")
                    success = False

        # Copy ADC without correction
        for seq in quant_seqs:
            adc_src = get_sequence_path(cfg, pid, seq, use_bias_corrected=False)
            adc_dst = os.path.join(output_dir, f"{seq}_preprocessed.nii.gz")
            if os.path.exists(adc_src):
                shutil.copy2(adc_src, adc_dst)
            else:
                logger.warning(f"ADC not found for {pid}")

        tracker.update(success, pid)

    tracker.finish()


def main():
    parser = argparse.ArgumentParser(description="N4 bias field correction")
    parser.add_argument("--config", type=str, default="configs/study1_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    run_bias_correction(cfg)


if __name__ == "__main__":
    main()
