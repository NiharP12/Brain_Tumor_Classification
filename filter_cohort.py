# Environment: digital_brain_twin (CPU)
# src/step01_clinical_data/filter_cohort.py — Cohort selection with CONSORT flow
"""
Apply sequential inclusion/exclusion criteria to UCSF-PDGM metadata.
Produces a filtered cohort of IDH-wildtype glioma patients with:
  - Valid MGMT status
  - Available OS data
  - Tumour segmentation mask
  - Core MRI sequences (T1, T1c, T2, FLAIR)
  - Optional ADC check

Outputs:
  data/clinical/cohort_filtered.csv

Usage:
    conda activate digital_brain_twin
    python -m src.step01_clinical_data.filter_cohort --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config, get_patient_folder, get_sequence_path, get_segmentation_path
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer


def check_file_exists(cfg, patient_id: str, file_type: str) -> bool:
    """Check if a specific file exists for a patient.

    Args:
        cfg: Configuration object.
        patient_id: Patient ID from metadata (e.g. 'UCSF-PDGM-004').
        file_type: 'segmentation', or a sequence name like 'T1', 'T1c', etc.

    Returns:
        True if file exists.
    """
    # Handle patient ID format: metadata uses 'UCSF-PDGM-004' but folders
    # use 'UCSF-PDGM-0004'. Need to normalise.
    patient_id_norm = normalise_patient_id(patient_id)

    if file_type == 'segmentation':
        path = get_segmentation_path(cfg, patient_id_norm)
    else:
        # Use bias-corrected for structural if available
        use_bias = (cfg.use_existing_bias_corrected and
                    file_type in cfg.structural_sequences)
        path = get_sequence_path(cfg, patient_id_norm, file_type,
                                 use_bias_corrected=use_bias)
    return os.path.exists(path)


def normalise_patient_id(patient_id: str) -> str:
    """Normalise patient ID to 4-digit format.

    UCSF-PDGM metadata uses 'UCSF-PDGM-004' but folders use
    'UCSF-PDGM-0004'. This function ensures consistent 4-digit IDs.

    Args:
        patient_id: Raw patient ID from metadata.

    Returns:
        Normalised patient ID with 4-digit number.
    """
    # Handle follow-up IDs like 'UCSF-PDGM-0429_FU003d'
    parts = patient_id.split('_FU')
    base = parts[0]

    # Extract the numeric part
    prefix = "UCSF-PDGM-"
    if base.startswith(prefix):
        num_str = base[len(prefix):]
        try:
            num = int(num_str)
            return f"{prefix}{num:04d}"
        except ValueError:
            return base
    return base


def is_follow_up(patient_id: str) -> bool:
    """Check if a patient ID corresponds to a follow-up scan.

    Args:
        patient_id: Patient ID from metadata.

    Returns:
        True if this is a follow-up scan entry.
    """
    return '_FU' in patient_id or 'FU' in patient_id.split('-')[-1]


@timer()
def filter_cohort(cfg) -> pd.DataFrame:
    """Apply sequential filtering criteria to the UCSF-PDGM cohort.

    Args:
        cfg: Configuration object.

    Returns:
        Filtered DataFrame.
    """
    logger = setup_logger("filter_cohort", "results/logs/step01_filter.log")

    # ── Load metadata ──────────────────────────────────────
    metadata_path = cfg.resolve_path("metadata_file")
    logger.info(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    logger.info(f"Total patients in metadata: {len(df)}")

    # CONSORT flow tracking
    consort = []
    consort.append(("Initial dataset (UCSF-PDGM v5)", len(df)))

    # ── Step 0: Exclude follow-up scans ────────────────────
    if cfg.exclude_followup:
        mask_baseline = ~df[cfg.col_patient_id].apply(is_follow_up)
        df = df[mask_baseline].copy()
        logger.info(f"After excluding follow-up scans: {len(df)}")
        consort.append(("Exclude follow-up scans", len(df)))

    # ── Step 1: IDH-wildtype only ──────────────────────────
    mask_idh = df[cfg.col_idh].str.lower() == cfg.idh_filter.lower()
    df_idh = df[mask_idh].copy()
    excluded_idh = len(df) - len(df_idh)
    logger.info(f"IDH-wildtype filter: {len(df_idh)} remaining "
                f"({excluded_idh} excluded)")
    consort.append(("IDH-wildtype only", len(df_idh)))

    # ── Step 2: Valid MGMT status ──────────────────────────
    valid_mgmt = [v.lower() for v in cfg.mgmt_valid_values]
    mask_mgmt = df_idh[cfg.col_mgmt].str.lower().isin(valid_mgmt)
    df_mgmt = df_idh[mask_mgmt].copy()
    excluded_mgmt = len(df_idh) - len(df_mgmt)
    logger.info(f"MGMT status filter (positive/negative): {len(df_mgmt)} "
                f"remaining ({excluded_mgmt} excluded)")
    consort.append(("Valid MGMT status", len(df_mgmt)))

    # ── Step 3: Non-missing OS ─────────────────────────────
    df_mgmt[cfg.col_os] = pd.to_numeric(df_mgmt[cfg.col_os], errors='coerce')
    mask_os = df_mgmt[cfg.col_os].notna()
    df_os = df_mgmt[mask_os].copy()
    excluded_os = len(df_mgmt) - len(df_os)
    logger.info(f"Non-missing OS: {len(df_os)} remaining "
                f"({excluded_os} excluded)")
    consort.append(("Available OS data", len(df_os)))

    # ── Step 4: Segmentation mask exists ───────────────────
    logger.info("Checking segmentation mask availability...")
    mask_seg = df_os[cfg.col_patient_id].apply(
        lambda pid: check_file_exists(cfg, pid, 'segmentation')
    )
    df_seg = df_os[mask_seg].copy()
    excluded_seg = len(df_os) - len(df_seg)
    logger.info(f"Segmentation available: {len(df_seg)} remaining "
                f"({excluded_seg} excluded)")
    consort.append(("Tumour segmentation available", len(df_seg)))

    # ── Step 5: Core MRI sequences exist ───────────────────
    logger.info("Checking core MRI sequence availability (T1, T1c, T2, FLAIR)...")
    core_sequences = ["T1", "T1c", "T2", "FLAIR"]

    def has_core_sequences(pid):
        for seq in core_sequences:
            if not check_file_exists(cfg, pid, seq):
                return False
        return True

    mask_core = df_seg[cfg.col_patient_id].apply(has_core_sequences)
    df_core = df_seg[mask_core].copy()
    excluded_core = len(df_seg) - len(df_core)
    logger.info(f"Core sequences available: {len(df_core)} remaining "
                f"({excluded_core} excluded)")
    consort.append(("Core MRI sequences (T1, T1c, T2, FLAIR)", len(df_core)))

    # ── Step 6: ADC availability ───────────────────────────
    logger.info("Checking ADC availability...")
    mask_adc = df_core[cfg.col_patient_id].apply(
        lambda pid: check_file_exists(cfg, pid, 'ADC')
    )
    df_adc = df_core[mask_adc].copy()
    excluded_adc = len(df_core) - len(df_adc)
    logger.info(f"ADC available: {len(df_adc)} remaining "
                f"({excluded_adc} excluded)")
    consort.append(("ADC map available", len(df_adc)))

    # ── Normalise patient IDs ──────────────────────────────
    df_final = df_adc.copy()
    df_final['patient_id_norm'] = df_final[cfg.col_patient_id].apply(
        normalise_patient_id
    )

    # ── Print CONSORT flow ─────────────────────────────────
    print("\n" + "=" * 60)
    print("CONSORT-STYLE FLOW DIAGRAM")
    print("=" * 60)
    for step, count in consort:
        excluded = consort[0][1] - count if step != consort[0][0] else 0
        print(f"  {step}: n = {count}")
    print("=" * 60)

    # ── Print final summary ────────────────────────────────
    print(f"\n✓ Final cohort size: {len(df_final)} patients")
    print(f"  IDH-wildtype: {len(df_final)}")
    mgmt_pos = (df_final[cfg.col_mgmt].str.lower() == 'positive').sum()
    mgmt_neg = (df_final[cfg.col_mgmt].str.lower() == 'negative').sum()
    print(f"  MGMT+: {mgmt_pos}, MGMT-: {mgmt_neg}")
    print(f"  Median OS: {df_final[cfg.col_os].median():.0f} days")
    dead = df_final[cfg.col_vital_status].sum()
    print(f"  Events (deaths): {dead} ({100*dead/len(df_final):.1f}%)")

    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Filter UCSF-PDGM cohort for Study 1"
    )
    parser.add_argument("--config", type=str,
                        default="configs/study1_config.yaml",
                        help="Path to configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)

    df = filter_cohort(cfg)

    # Save filtered cohort
    output_dir = cfg.resolve_path("clinical_dir")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cohort_filtered.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved filtered cohort to: {output_path}")


if __name__ == "__main__":
    main()
