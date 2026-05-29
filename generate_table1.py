# Environment: digital_brain_twin (CPU)
# src/step01_clinical_data/generate_table1.py — Comprehensive Table 1
"""
Generate a descriptive Table 1 for the filtered cohort, including:
  - Demographics (age, sex)
  - Molecular markers (MGMT)
  - Treatment (extent of resection)
  - Survival statistics
  - MGMT+ vs MGMT- subgroup comparison

Outputs:
  results/tables/table1.csv
  results/tables/table1.txt

Usage:
    conda activate digital_brain_twin
    python -m src.step01_clinical_data.generate_table1 --config configs/study1_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import load_config
from src.utils.seed_utils import set_all_seeds
from src.utils.logging_utils import setup_logger, timer


def compute_km_survival_rate(times: np.ndarray, events: np.ndarray,
                              timepoint: float) -> tuple:
    """Compute KM survival probability at a specific timepoint.

    Args:
        times: Survival times.
        events: Event indicators (1=event).
        timepoint: Time point of interest (days).

    Returns:
        Tuple of (survival_prob, lower_ci, upper_ci).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    timeline = kmf.survival_function_at_times(timepoint)
    ci = kmf.confidence_interval_survival_function_
    try:
        lower = ci.iloc[:, 0].loc[ci.index <= timepoint].iloc[-1]
        upper = ci.iloc[:, 1].loc[ci.index <= timepoint].iloc[-1]
    except (IndexError, KeyError):
        lower, upper = np.nan, np.nan
    return timeline.values[0], lower, upper


def compute_median_os_ci(times: np.ndarray, events: np.ndarray) -> tuple:
    """Compute median OS with 95% CI using Kaplan-Meier.

    Args:
        times: Survival times.
        events: Event indicators.

    Returns:
        Tuple of (median, lower_ci, upper_ci).
    """
    from lifelines.utils import median_survival_times
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    median = kmf.median_survival_time_
    
    try:
        med_ci = median_survival_times(kmf.confidence_interval_)
        lower = med_ci.iloc[0, 0]
        upper = med_ci.iloc[0, 1]
    except Exception:
        lower, upper = np.nan, np.nan
        
    return median, lower, upper


@timer()
def generate_table1(cfg) -> pd.DataFrame:
    """Generate comprehensive Table 1 for the cohort.

    Args:
        cfg: Configuration object.

    Returns:
        Table 1 as a DataFrame.
    """
    logger = setup_logger("table1", "results/logs/step01_table1.log")

    # ── Load data ──────────────────────────────────────────
    clinical_dir = cfg.resolve_path("clinical_dir")
    df = pd.read_csv(os.path.join(clinical_dir, "cohort_filtered.csv"))
    logger.info(f"Loaded cohort: {len(df)} patients")

    # Prepare variables
    df['Age'] = df[cfg.col_age].astype(float)
    df['OS_days'] = pd.to_numeric(df[cfg.col_os], errors='coerce')
    df['Event'] = df[cfg.col_vital_status].astype(int)
    df['MGMT'] = df[cfg.col_mgmt].str.lower()
    df['Sex'] = df[cfg.col_sex].str.upper()
    df['EOR'] = df[cfg.col_eor].str.upper().str.strip()

    # Subgroups
    mgmt_pos = df[df['MGMT'] == 'positive']
    mgmt_neg = df[df['MGMT'] == 'negative']

    rows = []

    # ── N ──────────────────────────────────────────────────
    rows.append(('N', str(len(df)), str(len(mgmt_pos)), str(len(mgmt_neg)), ''))

    # ── Age ────────────────────────────────────────────────
    for label, subset in [('Overall', df), ('MGMT+', mgmt_pos), ('MGMT-', mgmt_neg)]:
        pass  # Computed inline

    age_all = df['Age']
    age_pos = mgmt_pos['Age']
    age_neg = mgmt_neg['Age']

    rows.append(('Age (years)', '', '', '', ''))
    rows.append(('  Mean ± SD',
                 f"{age_all.mean():.1f} ± {age_all.std():.1f}",
                 f"{age_pos.mean():.1f} ± {age_pos.std():.1f}",
                 f"{age_neg.mean():.1f} ± {age_neg.std():.1f}",
                 ''))
    rows.append(('  Median [IQR]',
                 f"{age_all.median():.1f} [{age_all.quantile(0.25):.1f}-{age_all.quantile(0.75):.1f}]",
                 f"{age_pos.median():.1f} [{age_pos.quantile(0.25):.1f}-{age_pos.quantile(0.75):.1f}]",
                 f"{age_neg.median():.1f} [{age_neg.quantile(0.25):.1f}-{age_neg.quantile(0.75):.1f}]",
                 ''))
    rows.append(('  Range',
                 f"{age_all.min():.0f}-{age_all.max():.0f}",
                 f"{age_pos.min():.0f}-{age_pos.max():.0f}",
                 f"{age_neg.min():.0f}-{age_neg.max():.0f}",
                 ''))

    # Mann-Whitney U for age
    stat_age, p_age = stats.mannwhitneyu(age_pos, age_neg, alternative='two-sided')
    rows[-1] = (*rows[-1][:4], f"p={p_age:.3f}")

    # ── Sex ────────────────────────────────────────────────
    rows.append(('Sex', '', '', '', ''))
    for sex_val in ['M', 'F']:
        n_all = (df['Sex'] == sex_val).sum()
        n_pos = (mgmt_pos['Sex'] == sex_val).sum()
        n_neg = (mgmt_neg['Sex'] == sex_val).sum()
        label = 'Male' if sex_val == 'M' else 'Female'
        rows.append((f'  {label}',
                     f"{n_all} ({100*n_all/len(df):.1f}%)",
                     f"{n_pos} ({100*n_pos/len(mgmt_pos):.1f}%)",
                     f"{n_neg} ({100*n_neg/len(mgmt_neg):.1f}%)",
                     ''))

    # Chi-square for sex
    contingency_sex = pd.crosstab(df['Sex'], df['MGMT'])
    chi2_sex, p_sex, _, _ = stats.chi2_contingency(contingency_sex)
    rows[-1] = (*rows[-1][:4], f"p={p_sex:.3f}")

    # ── MGMT ───────────────────────────────────────────────
    rows.append(('MGMT Methylation', '', '', '', ''))
    rows.append(('  Positive',
                 f"{len(mgmt_pos)} ({100*len(mgmt_pos)/len(df):.1f}%)",
                 '-', '-', ''))
    rows.append(('  Negative',
                 f"{len(mgmt_neg)} ({100*len(mgmt_neg)/len(df):.1f}%)",
                 '-', '-', ''))

    # ── EOR ────────────────────────────────────────────────
    rows.append(('Extent of Resection', '', '', '', ''))
    for eor_val in ['GTR', 'STR', 'BIOPSY']:
        n_all = (df['EOR'] == eor_val).sum()
        n_pos = (mgmt_pos['EOR'] == eor_val).sum()
        n_neg = (mgmt_neg['EOR'] == eor_val).sum()
        pct_all = 100 * n_all / len(df) if len(df) > 0 else 0
        pct_pos = 100 * n_pos / len(mgmt_pos) if len(mgmt_pos) > 0 else 0
        pct_neg = 100 * n_neg / len(mgmt_neg) if len(mgmt_neg) > 0 else 0
        rows.append((f'  {eor_val}',
                     f"{n_all} ({pct_all:.1f}%)",
                     f"{n_pos} ({pct_pos:.1f}%)",
                     f"{n_neg} ({pct_neg:.1f}%)",
                     ''))

    # Chi-square for EOR
    contingency_eor = pd.crosstab(df['EOR'], df['MGMT'])
    if contingency_eor.shape[0] > 1 and contingency_eor.shape[1] > 1:
        chi2_eor, p_eor, _, _ = stats.chi2_contingency(contingency_eor)
        rows[-1] = (*rows[-1][:4], f"p={p_eor:.3f}")

    # ── Survival ───────────────────────────────────────────
    rows.append(('Survival', '', '', '', ''))

    times_all = df['OS_days'].values
    events_all = df['Event'].values
    times_pos = mgmt_pos['OS_days'].values
    events_pos = mgmt_pos['Event'].values
    times_neg = mgmt_neg['OS_days'].values
    events_neg = mgmt_neg['Event'].values

    # Median OS with 95% CI
    med_all, ci_lo_all, ci_hi_all = compute_median_os_ci(times_all, events_all)
    med_pos, ci_lo_pos, ci_hi_pos = compute_median_os_ci(times_pos, events_pos)
    med_neg, ci_lo_neg, ci_hi_neg = compute_median_os_ci(times_neg, events_neg)
    rows.append(('  Median OS [95% CI] (days)',
                 f"{med_all:.0f} [{ci_lo_all:.0f}-{ci_hi_all:.0f}]",
                 f"{med_pos:.0f} [{ci_lo_pos:.0f}-{ci_hi_pos:.0f}]",
                 f"{med_neg:.0f} [{ci_lo_neg:.0f}-{ci_hi_neg:.0f}]",
                 ''))

    # Survival rates at 6, 12, 24 months
    for months, days in [(6, 183), (12, 365), (24, 730)]:
        s_all, _, _ = compute_km_survival_rate(times_all, events_all, days)
        s_pos, _, _ = compute_km_survival_rate(times_pos, events_pos, days)
        s_neg, _, _ = compute_km_survival_rate(times_neg, events_neg, days)
        rows.append((f'  {months}-month survival rate',
                     f"{s_all:.1%}",
                     f"{s_pos:.1%}",
                     f"{s_neg:.1%}",
                     ''))

    # Censoring proportion
    cens_all = (1 - events_all).sum()
    cens_pos = (1 - events_pos).sum()
    cens_neg = (1 - events_neg).sum()
    rows.append(('  Censored',
                 f"{cens_all} ({100*cens_all/len(df):.1f}%)",
                 f"{cens_pos} ({100*cens_pos/len(mgmt_pos):.1f}%)",
                 f"{cens_neg} ({100*cens_neg/len(mgmt_neg):.1f}%)",
                 ''))

    # Log-rank test
    from lifelines.statistics import logrank_test
    lr = logrank_test(times_pos, times_neg, events_pos, events_neg)
    rows[-1] = (*rows[-1][:4], f"p={lr.p_value:.4f}")

    # ── Build DataFrame ────────────────────────────────────
    table1 = pd.DataFrame(rows, columns=['Variable', 'Overall', 'MGMT+',
                                          'MGMT-', 'p-value'])

    return table1


def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 1 for Study 1"
    )
    parser.add_argument("--config", type=str,
                        default="configs/study1_config.yaml",
                        help="Path to configuration YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg.global_seed)

    table1 = generate_table1(cfg)

    # Save as CSV
    tables_dir = cfg.resolve_path("tables_dir")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, "table1.csv")
    table1.to_csv(csv_path, index=False)

    # Save as formatted text
    txt_path = os.path.join(tables_dir, "table1.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("TABLE 1: Baseline Characteristics of IDH-Wildtype Glioma Cohort\n")
        f.write("=" * 100 + "\n\n")
        f.write(table1.to_string(index=False))
        f.write("\n\n" + "=" * 100 + "\n")

    # Print to console
    print("\n")
    print("=" * 100)
    print("TABLE 1: Baseline Characteristics")
    print("=" * 100)
    print(table1.to_string(index=False))
    print("=" * 100)
    print(f"\n✓ Saved to: {csv_path}")
    print(f"✓ Saved to: {txt_path}")


if __name__ == "__main__":
    main()
