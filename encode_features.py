# Environment: digital_brain_twin (CPU)
# src/step01_clinical_data/encode_features.py — Clinical feature encoding
"""
Encode clinical and molecular variables into numeric feature vectors.

Produces:
  - data/clinical/clinical_features.npy  (N × 6)
  - data/clinical/survival_data.csv      (patient_id, OS_days, vital_status)
  - data/clinical/age_scaler.pkl         (StandardScaler for age)

Feature vector (6 dimensions):
  [Age_norm, Sex_enc, MGMT_enc, EOR_biopsy, EOR_STR, EOR_GTR]

Usage:
    conda activate digital_brain_twin
    python -m src.step01_clinical_data.encode_features --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer
from src.step01_clinical_data.filter_cohort import normalise_patient_id


@timer()
def encode_features(cfg) -> None:
    """Encode clinical variables from filtered cohort.

    Processing:
      1. Sex: Female=1, Male=0
      2. MGMT: positive=1, negative=0
      3. EOR: one-hot (biopsy, STR, GTR) — NOT ordinal
      4. Age: Z-score normalisation with saved scaler

    Args:
        cfg: Configuration object.
    """
    logger = setup_logger("encode_features", "results/logs/step01_encode.log")

    # ── Load filtered cohort ───────────────────────────────
    clinical_dir = cfg.resolve_path("clinical_dir")
    cohort_path = os.path.join(clinical_dir, "cohort_filtered.csv")
    logger.info(f"Loading filtered cohort from: {cohort_path}")
    df = pd.read_csv(cohort_path)
    logger.info(f"Cohort size: {len(df)} patients")

    # ── Normalise patient IDs ──────────────────────────────
    df['patient_id'] = df[cfg.col_patient_id].apply(normalise_patient_id)

    # ── Encode Sex ─────────────────────────────────────────
    df['Sex_enc'] = (df[cfg.col_sex].str.upper() == 'F').astype(float)
    logger.info(f"Sex encoding — Female(1): {df['Sex_enc'].sum():.0f}, "
                f"Male(0): {(1 - df['Sex_enc']).sum():.0f}")

    # ── Encode MGMT ────────────────────────────────────────
    df['MGMT_enc'] = (df[cfg.col_mgmt].str.lower() == 'positive').astype(float)
    logger.info(f"MGMT encoding — Positive(1): {df['MGMT_enc'].sum():.0f}, "
                f"Negative(0): {(1 - df['MGMT_enc']).sum():.0f}")

    # ── Encode EOR (one-hot) ───────────────────────────────
    eor_values = df[cfg.col_eor].str.lower().str.strip()
    df['EOR_biopsy'] = (eor_values == 'biopsy').astype(float)
    df['EOR_STR'] = (eor_values == 'str').astype(float)
    df['EOR_GTR'] = (eor_values == 'gtr').astype(float)
    logger.info(f"EOR distribution — GTR: {df['EOR_GTR'].sum():.0f}, "
                f"STR: {df['EOR_STR'].sum():.0f}, "
                f"Biopsy: {df['EOR_biopsy'].sum():.0f}")

    # Check for unmatched EOR values
    eor_sum = df['EOR_biopsy'] + df['EOR_STR'] + df['EOR_GTR']
    unmatched = (eor_sum == 0).sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} patients have unrecognised EOR values: "
                       f"{df[eor_sum == 0][cfg.col_eor].unique()}")
        # Default unmatched to biopsy (most conservative)
        df.loc[eor_sum == 0, 'EOR_biopsy'] = 1.0

    # ── Normalise Age (Z-score) ────────────────────────────
    age_values = df[cfg.col_age].values.astype(float).reshape(-1, 1)
    age_scaler = StandardScaler()
    df['Age_norm'] = age_scaler.fit_transform(age_values).flatten()
    logger.info(f"Age normalisation — Mean: {age_scaler.mean_[0]:.2f}, "
                f"Std: {age_scaler.scale_[0]:.2f}")

    # Save scaler for external validation
    scaler_path = os.path.join(clinical_dir, "age_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(age_scaler, f)
    logger.info(f"Saved age scaler to: {scaler_path}")

    # ── Build clinical feature matrix ──────────────────────
    feature_cols = ['Age_norm', 'Sex_enc', 'MGMT_enc',
                    'EOR_biopsy', 'EOR_STR', 'EOR_GTR']
    clinical_features = df[feature_cols].values.astype(np.float32)
    logger.info(f"Clinical feature matrix shape: {clinical_features.shape}")

    # Verify dimensions
    assert clinical_features.shape[1] == cfg.clinical_feature_dim, \
        f"Expected {cfg.clinical_feature_dim} features, got {clinical_features.shape[1]}"

    # Save clinical features
    features_path = os.path.join(clinical_dir, "clinical_features.npy")
    np.save(features_path, clinical_features)
    logger.info(f"Saved clinical features to: {features_path}")

    # ── Build survival data ────────────────────────────────
    survival_df = pd.DataFrame({
        'patient_id': df['patient_id'].values,
        'OS_days': df[cfg.col_os].values.astype(float),
        'vital_status': df[cfg.col_vital_status].values.astype(int),
    })

    survival_path = os.path.join(clinical_dir, "survival_data.csv")
    survival_df.to_csv(survival_path, index=False)
    logger.info(f"Saved survival data to: {survival_path}")

    # ── Save patient ID mapping ────────────────────────────
    # Save mapping from original to normalised IDs and feature indices
    id_mapping = pd.DataFrame({
        'original_id': df[cfg.col_patient_id].values,
        'patient_id': df['patient_id'].values,
        'feature_index': range(len(df)),
    })
    id_mapping.to_csv(os.path.join(clinical_dir, "patient_id_mapping.csv"),
                      index=False)

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Clinical Feature Encoding — Summary")
    print("=" * 60)
    print(f"  Patients: {len(df)}")
    print(f"  Feature dimensions: {clinical_features.shape[1]}")
    print(f"  Features: {feature_cols}")
    print(f"\n  Feature statistics:")
    for i, col in enumerate(feature_cols):
        vals = clinical_features[:, i]
        print(f"    {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")
    print(f"\n  Survival:")
    print(f"    Median OS: {survival_df['OS_days'].median():.0f} days")
    print(f"    Events: {survival_df['vital_status'].sum()} / {len(survival_df)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Encode clinical features for Study 1"
    )
    parser.add_argument("--config", type=str,
                        default="configs/study1_config.yaml",
                        help="Path to configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)
    encode_features(cfg)


if __name__ == "__main__":
    main()
