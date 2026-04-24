"""
Experiment D: Bonferroni Correction Reanalysis (Reviewer #5)

Offline statistical reanalysis of all pairwise comparisons in the paper.
Applies Bonferroni and Holm-Bonferroni corrections within each comparison family.

Usage:
    python experiments/federated/run_bonferroni_reanalysis.py
    python experiments/federated/run_bonferroni_reanalysis.py --include-new
"""

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ranksums, kruskal
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Bonferroni Correction Reanalysis')
    parser.add_argument('--include-new', action='store_true',
                        help='Include K-ablation and DP-defense results if available')
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def load_latest_csv(pattern: str) -> pd.DataFrame:
    """Load the most recent CSV matching the pattern."""
    base_dir = 'results/federated/tables'
    files = sorted(glob.glob(os.path.join(base_dir, pattern)))
    if not files:
        return None
    return pd.read_csv(files[-1])


def pairwise_tests(df: pd.DataFrame, group_col: str, value_col: str,
                   family_name: str) -> list:
    """Run pairwise Wilcoxon rank-sum tests for all group pairs.

    Args:
        df: DataFrame with data
        group_col: Column name for group labels
        value_col: Column name for numeric values to compare
        family_name: Name for this comparison family

    Returns:
        List of result dicts
    """
    groups = sorted(df[group_col].unique())
    results = []

    for g1, g2 in combinations(groups, 2):
        vals1 = df[df[group_col] == g1][value_col].dropna().values
        vals2 = df[df[group_col] == g2][value_col].dropna().values

        if len(vals1) < 3 or len(vals2) < 3:
            continue

        stat, p_raw = ranksums(vals1, vals2)

        results.append({
            'family': family_name,
            'group_a': str(g1),
            'group_b': str(g2),
            'n_a': len(vals1),
            'n_b': len(vals2),
            'mean_a': float(np.mean(vals1)),
            'mean_b': float(np.mean(vals2)),
            'mean_diff': float(np.mean(vals1) - np.mean(vals2)),
            'statistic': float(stat),
            'p_raw': float(p_raw),
        })

    return results


def apply_bonferroni(results: list) -> list:
    """Apply Bonferroni and Holm-Bonferroni corrections within each family."""
    # Group by family
    families = {}
    for r in results:
        families.setdefault(r['family'], []).append(r)

    corrected = []
    for family_name, family_results in families.items():
        m = len(family_results)  # Number of comparisons in this family

        # Bonferroni
        for r in family_results:
            r['n_comparisons'] = m
            r['p_bonferroni'] = min(r['p_raw'] * m, 1.0)
            r['sig_raw'] = r['p_raw'] < 0.05
            r['sig_bonferroni'] = r['p_bonferroni'] < 0.05

        # Holm-Bonferroni (step-down)
        sorted_by_p = sorted(family_results, key=lambda x: x['p_raw'])
        for i, r in enumerate(sorted_by_p):
            k = m - i  # Remaining comparisons
            r['p_holm'] = min(r['p_raw'] * k, 1.0)

        # Enforce monotonicity for Holm
        for i in range(1, len(sorted_by_p)):
            sorted_by_p[i]['p_holm'] = max(
                sorted_by_p[i]['p_holm'],
                sorted_by_p[i - 1]['p_holm']
            )

        for r in sorted_by_p:
            r['sig_holm'] = r['p_holm'] < 0.05

        corrected.extend(family_results)

    return corrected


def main():
    args = parse_args()

    print("=" * 70)
    print("Experiment D: Bonferroni Correction Reanalysis")
    print("=" * 70)

    all_comparisons = []

    # ================================================================
    # Family 1: Method comparisons on global_l2_auc (per dataset)
    # ================================================================
    main_df = load_latest_csv('main_results_*.csv')
    if main_df is not None:
        print(f"\nLoaded main results: {len(main_df)} rows")

        for dataset in main_df['dataset'].unique():
            sub = main_df[main_df['dataset'] == dataset]
            results = pairwise_tests(
                sub, 'method', 'global_l2_auc',
                f'methods_global_l2_{dataset}'
            )
            all_comparisons.extend(results)

        # Aggregate across datasets
        results = pairwise_tests(
            main_df, 'method', 'global_l2_auc',
            'methods_global_l2_all'
        )
        all_comparisons.extend(results)

        # Family 2: Level comparisons (melt to long format)
        for method in main_df['method'].unique():
            sub = main_df[main_df['method'] == method]
            level_data = []
            for _, row in sub.iterrows():
                level_data.append({'level': 'Global', 'l2_auc': row['global_l2_auc']})
                level_data.append({'level': 'Local', 'l2_auc': row['local_l2_auc']})
                level_data.append({'level': 'Cross', 'l2_auc': row['mean_cross_l2_auc']})
            level_df = pd.DataFrame(level_data)
            results = pairwise_tests(
                level_df, 'level', 'l2_auc',
                f'levels_{method}'
            )
            all_comparisons.extend(results)
    else:
        print("WARNING: No main results found!")

    # ================================================================
    # Family 3: Threat model comparisons
    # ================================================================
    tm_df = load_latest_csv('rq4_threat_model_*.csv')
    if tm_df is not None:
        print(f"Loaded threat model results: {len(tm_df)} rows")
        results = pairwise_tests(
            tm_df, 'threat_model', 'l2_auc',
            'threat_models'
        )
        all_comparisons.extend(results)
    else:
        print("WARNING: No threat model results found!")

    # ================================================================
    # Family 4: Granularity comparisons
    # ================================================================
    gran_df = load_latest_csv('rq5_granularity_*.csv')
    if gran_df is not None:
        print(f"Loaded granularity results: {len(gran_df)} rows")
        for metric in ['global_l2_auc', 'local_l2_auc', 'mean_cross_l2_auc']:
            results = pairwise_tests(
                gran_df, 'granularity', metric,
                f'granularity_{metric}'
            )
            all_comparisons.extend(results)
    else:
        print("WARNING: No granularity results found!")

    # ================================================================
    # Family 5: K-ablation (if available)
    # ================================================================
    if args.include_new:
        k_df = load_latest_csv('k_ablation_*.csv')
        if k_df is not None:
            print(f"Loaded K-ablation results: {len(k_df)} rows")
            gnndelete = k_df[k_df['method'] == 'FedGNNDelete']
            results = pairwise_tests(
                gnndelete, 'num_clients', 'global_l2_auc',
                'k_ablation_global_l2'
            )
            all_comparisons.extend(results)

        # DP defense
        dp_df = load_latest_csv('dp_defense_*.csv')
        if dp_df is not None:
            print(f"Loaded DP defense results: {len(dp_df)} rows")
            results = pairwise_tests(
                dp_df, 'epsilon', 'global_l2_auc',
                'dp_defense_global_l2'
            )
            all_comparisons.extend(results)

    if not all_comparisons:
        print("\nNo comparisons to analyze!")
        return

    # Apply corrections
    corrected = apply_bonferroni(all_comparisons)

    df = pd.DataFrame(corrected)

    # Save
    output_dir = 'results/federated/tables'
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'bonferroni_reanalysis_{timestamp}.csv')

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Bonferroni Correction Summary")
    print("=" * 70)

    families = df['family'].unique()
    for family in sorted(families):
        sub = df[df['family'] == family]
        n_total = len(sub)
        n_sig_raw = sub['sig_raw'].sum()
        n_sig_bonf = sub['sig_bonferroni'].sum()
        n_sig_holm = sub['sig_holm'].sum()
        print(f"\n{family} ({n_total} comparisons):")
        print(f"  Significant (raw α=0.05):        {n_sig_raw}/{n_total}")
        print(f"  Significant (Bonferroni):         {n_sig_bonf}/{n_total}")
        print(f"  Significant (Holm-Bonferroni):    {n_sig_holm}/{n_total}")

        # Show details for each comparison
        for _, row in sub.iterrows():
            sig_mark = "***" if row['sig_bonferroni'] else ("*" if row['sig_raw'] else "ns")
            print(f"    {row['group_a']:<20} vs {row['group_b']:<20} "
                  f"Δ={row['mean_diff']:+.4f}  "
                  f"p_raw={row['p_raw']:.2e}  "
                  f"p_bonf={row['p_bonferroni']:.2e}  "
                  f"p_holm={row['p_holm']:.2e}  {sig_mark}")

    # Check if any conclusions change after correction
    changed = df[(df['sig_raw'] == True) & (df['sig_bonferroni'] == False)]
    if len(changed) > 0:
        print(f"\n{'!' * 70}")
        print(f"WARNING: {len(changed)} comparisons lose significance after Bonferroni:")
        for _, row in changed.iterrows():
            print(f"  {row['family']}: {row['group_a']} vs {row['group_b']} "
                  f"(p_raw={row['p_raw']:.2e} → p_bonf={row['p_bonferroni']:.2e})")
    else:
        print(f"\nAll significant results remain significant after Bonferroni correction.")


if __name__ == '__main__':
    main()
