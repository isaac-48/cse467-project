import argparse
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, entropy, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

### USAGE: /opt/homebrew/bin/python3.10 comparison.py --real_csv ../Student_data.csv --synthetic_csv ./pategan_synth_out.csv --output_plot ./pategan_comp.png --output_metrics_json ./privacy_utility.json

FEATURES_FOR_PLOTS = [
    'age',
    'study_hours_per_day',
    'attendance_pct',
    'previous_gpa',
    'sleep_hours',
    'final_cgpa',
]


# Correlation heatmap mapping: -1 -> gray, 0 -> blue, +1 -> red.
CORR_CMAP = LinearSegmentedColormap.from_list(
    'corr_red_blue_gray',
    ['#bfbfbf', '#2b6cb0', '#c53030'],
    N=256,
)


def _normalize_columns(df):
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _common_numeric_columns(real_df, synth_df):
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    numeric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(real_df[c]) and pd.api.types.is_numeric_dtype(synth_df[c])
    ]
    return common_cols, numeric_cols


def _build_hist_probs(real_vals, synth_vals, bins=20):
    lo = float(min(np.nanmin(real_vals), np.nanmin(synth_vals)))
    hi = float(max(np.nanmax(real_vals), np.nanmax(synth_vals)))
    if np.isclose(lo, hi):
        hi = lo + 1e-6

    real_counts, edges = np.histogram(real_vals, bins=bins, range=(lo, hi))
    synth_counts, _ = np.histogram(synth_vals, bins=edges)

    real_probs = real_counts.astype(float) + 1e-10
    synth_probs = synth_counts.astype(float) + 1e-10
    real_probs /= real_probs.sum()
    synth_probs /= synth_probs.sum()

    return real_counts, synth_counts, real_probs, synth_probs


def calc_utility_metrics(real_df, synth_df, numeric_cols):
    """Calculate utility metrics across shared numeric columns."""

    per_feature_rows = []
    for col in numeric_cols:
        real_vals = pd.to_numeric(real_df[col], errors='coerce').dropna().to_numpy()
        synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna().to_numpy()

        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue

        real_counts, synth_counts, real_probs, synth_probs = _build_hist_probs(real_vals, synth_vals)

        kl_div = float(entropy(real_probs, synth_probs))
        js_div = float(jensenshannon(real_probs, synth_probs) ** 2)
        w_dist = float(wasserstein_distance(real_vals, synth_vals))

        contingency = np.vstack([real_counts, synth_counts])
        non_empty_bins = contingency.sum(axis=0) > 0
        contingency = contingency[:, non_empty_bins]

        if contingency.shape[1] >= 2:
            chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency)
        else:
            chi2_stat, chi2_p_value = np.nan, np.nan

        per_feature_rows.append({
            'feature': col,
            'kl_divergence': kl_div,
            'jensen_shannon_divergence': js_div,
            'wasserstein_distance': w_dist,
            'chi_square_stat': float(chi2_stat),
            'chi_square_p_value': float(chi2_p_value),
        })

    utility_df = pd.DataFrame(per_feature_rows)

    # Distinguishability test: lower AUC is better utility privacy balance.
    min_size = min(len(real_df), len(synth_df))
    if min_size < 20:
        logreg_auc = np.nan
    else:
        real_sample = real_df[numeric_cols].sample(n=min_size, random_state=42, replace=False)
        synth_sample = synth_df[numeric_cols].sample(n=min_size, random_state=42, replace=False)

        combined = pd.concat([real_sample, synth_sample], axis=0, ignore_index=True)
        combined = combined.apply(pd.to_numeric, errors='coerce')
        combined = combined.fillna(combined.median(numeric_only=True))
        labels = np.concatenate([np.ones(min_size), np.zeros(min_size)])

        X_train, X_test, y_train, y_test = train_test_split(
            combined.values,
            labels,
            test_size=0.3,
            random_state=42,
            stratify=labels,
        )

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        logreg_auc = float(roc_auc_score(y_test, y_score))

    summary = {
        'feature_metric_means': (
            utility_df.drop(columns=['feature']).mean().to_dict() if not utility_df.empty else {}
        ),
        'logistic_regression_auc_real_vs_synth': logreg_auc,
    }

    return utility_df, summary


def calc_privacy_metrics(real_df, synth_df, common_cols, numeric_cols):
    """Calculate privacy metrics: delta presence, k-anonymity, identifiability."""

    # Delta presence proxy based on exact tuple overlap over shared columns.
    real_keys = set(real_df[common_cols].astype(str).agg('|'.join, axis=1).unique())
    synth_keys = set(synth_df[common_cols].astype(str).agg('|'.join, axis=1).unique())
    overlap = len(real_keys.intersection(synth_keys))
    delta_presence = float(overlap / max(len(real_keys), 1))

    # k-anonymity over discretized quasi-identifiers.
    qid_cols = [c for c in ['age', 'gender', 'major', 'attendance_pct'] if c in common_cols]
    if not qid_cols:
        qid_cols = common_cols[: min(4, len(common_cols))]

    synth_qid = synth_df[qid_cols].copy()
    for col in qid_cols:
        if col in numeric_cols:
            synth_qid[col] = pd.qcut(
                pd.to_numeric(synth_qid[col], errors='coerce'),
                q=10,
                duplicates='drop',
            ).astype(str)
        else:
            synth_qid[col] = synth_qid[col].astype(str)

    eq_class_sizes = synth_qid.groupby(qid_cols).size()
    k_min = int(eq_class_sizes.min()) if len(eq_class_sizes) else 0
    k_median = float(eq_class_sizes.median()) if len(eq_class_sizes) else 0.0
    pct_lt5 = float((eq_class_sizes < 5).mean()) if len(eq_class_sizes) else 1.0

    # Identifiability via nearest-neighbor normalized distance to real records.
    identifiability_score = np.nan
    if numeric_cols:
        real_num = real_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        synth_num = synth_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        fill_values = real_num.median(numeric_only=True)
        real_num = real_num.fillna(fill_values)
        synth_num = synth_num.fillna(fill_values)

        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_num)
        synth_scaled = scaler.transform(synth_num)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(real_scaled)
        distances, _ = nn.kneighbors(synth_scaled)
        threshold = 0.5
        identifiability_score = float((distances.flatten() <= threshold).mean())

    privacy_summary = {
        'delta_presence_overlap_ratio': delta_presence,
        'k_anonymity_min_equivalence_class': k_min,
        'k_anonymity_median_equivalence_class': k_median,
        'k_anonymity_pct_classes_lt_5': pct_lt5,
        'identifiability_score_nn_distance_le_0_5': identifiability_score,
        'quasi_identifier_columns_used': qid_cols,
    }

    return privacy_summary


def compare_datasets(real_df, synth_df, output_plot='comparison.png', show=False):
    """Create overlay distribution plots and correlation matrices."""

    real_df = _normalize_columns(real_df)
    synth_df = _normalize_columns(synth_df)

    common_cols, numeric_cols = _common_numeric_columns(real_df, synth_df)
    if not common_cols:
        raise ValueError('No common columns found between real and synthetic datasets.')
    if not numeric_cols:
        raise ValueError('No common numeric columns found for comparison.')

    utility_by_feature, utility_summary = calc_utility_metrics(real_df, synth_df, numeric_cols)
    privacy_summary = calc_privacy_metrics(real_df, synth_df, common_cols, numeric_cols)

    plot_features = [c for c in FEATURES_FOR_PLOTS if c in numeric_cols]
    if len(plot_features) < 6:
        extras = [c for c in numeric_cols if c not in plot_features]
        plot_features.extend(extras[: max(0, 6 - len(plot_features))])
    plot_features = plot_features[:6]

    mpl.rcParams['figure.figsize'] = (16, 9)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 6)

    for i, col in enumerate(plot_features):
        ax = fig.add_subplot(gs[0, i])
        real_vals = pd.to_numeric(real_df[col], errors='coerce').dropna()
        synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna()

        ax.hist(real_vals, bins=20, density=True, alpha=0.55, label='Real')
        ax.hist(synth_vals, bins=20, density=True, alpha=0.55, label='Synthetic')
        ax.set_title(col, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    corr_features = [c for c in FEATURES_FOR_PLOTS if c in numeric_cols]
    if len(corr_features) < 2:
        corr_features = numeric_cols[: min(6, len(numeric_cols))]

    real_corr = real_df[corr_features].corr(numeric_only=True)
    synth_corr = synth_df[corr_features].corr(numeric_only=True)

    ax_real = fig.add_subplot(gs[1, :3])
    im1 = ax_real.imshow(real_corr.values, vmin=-1, vmax=1, cmap=CORR_CMAP)
    ax_real.set_title('Real Data Correlations')
    ax_real.set_xticks(range(len(corr_features)))
    ax_real.set_xticklabels(corr_features, rotation=90)
    ax_real.set_yticks(range(len(corr_features)))
    ax_real.set_yticklabels(corr_features)

    for r in range(real_corr.shape[0]):
        for c in range(real_corr.shape[1]):
            ax_real.text(c, r, f'{real_corr.values[r, c]:.2f}', ha='center', va='center', fontsize=7, color='white')

    ax_synth = fig.add_subplot(gs[1, 3:])
    ax_synth.imshow(synth_corr.values, vmin=-1, vmax=1, cmap=CORR_CMAP)
    ax_synth.set_title('Synthetic Data Correlations')
    ax_synth.set_xticks(range(len(corr_features)))
    ax_synth.set_xticklabels(corr_features, rotation=90)
    ax_synth.set_yticks(range(len(corr_features)))
    ax_synth.set_yticklabels(corr_features)

    for r in range(synth_corr.shape[0]):
        for c in range(synth_corr.shape[1]):
            ax_synth.text(c, r, f'{synth_corr.values[r, c]:.2f}', ha='center', va='center', fontsize=7, color='white')

    fig.colorbar(im1, ax=[ax_real, ax_synth], shrink=0.8)
    fig.suptitle('Real vs Synthetic: Distribution and Correlation Comparison', fontsize=14, fontweight='bold')

    if output_plot:
        fig.savefig(output_plot, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        'utility_metrics_by_feature': utility_by_feature,
        'utility_summary': utility_summary,
        'privacy_summary': privacy_summary,
        'plot_features_used': plot_features,
        'correlation_features_used': corr_features,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare real and synthetic tabular datasets.')
    parser.add_argument('--real_csv', required=True, help='Path to original/real dataset CSV.')
    parser.add_argument('--synthetic_csv', required=True, help='Path to synthetic dataset CSV.')
    parser.add_argument('--output_plot', default='comparison.png', help='Path to save comparison plot image.')
    parser.add_argument('--show', action='store_true', help='Display plot window in addition to saving.')
    parser.add_argument('--output_metrics_json', default='', help='Optional path to save metrics JSON.')
    args = parser.parse_args()

    real_df = pd.read_csv(args.real_csv)
    synth_df = pd.read_csv(args.synthetic_csv)

    result = compare_datasets(real_df, synth_df, output_plot=args.output_plot, show=args.show)

    print('\n=== Utility Metrics (Per Feature) ===')
    utility_table = result['utility_metrics_by_feature']
    if utility_table.empty:
        print('No comparable numeric columns found for utility table.')
    else:
        print(utility_table.to_string(index=False))

    print('\n=== Utility Summary ===')
    print(json.dumps(result['utility_summary'], indent=2))

    print('\n=== Privacy Summary ===')
    print(json.dumps(result['privacy_summary'], indent=2))

    if args.output_metrics_json:
        serializable_payload = {
            'utility_metrics_by_feature': utility_table.to_dict(orient='records'),
            'utility_summary': result['utility_summary'],
            'privacy_summary': result['privacy_summary'],
            'plot_features_used': result['plot_features_used'],
            'correlation_features_used': result['correlation_features_used'],
        }
        with open(args.output_metrics_json, 'w', encoding='utf-8') as fp:
            json.dump(serializable_payload, fp, indent=2)
        print(f"\nSaved metrics JSON to {args.output_metrics_json}")

    print(f"Saved plot to {args.output_plot}")


if __name__ == '__main__':
    main()