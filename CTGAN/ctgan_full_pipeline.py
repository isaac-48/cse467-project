# ctgan_full_pipeline.py
# Synthetic Student Data: Generation and Evaluation

import json
import numpy as np
import pandas as pd
from ctgan import CTGAN
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, entropy, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

#correlation heatmap model
CORR_CMAP = LinearSegmentedColormap.from_list(
    'corr_red_blue_gray',
    ['#bfbfbf', '#2b6cb0', '#c53030'],
    N=256,
)

# STEP1: Load and preprocess data
print("\n[1/5] Loading real data ")
data = pd.read_csv("Student_data.csv")
data = data.drop(columns=["Student_ID"])

discrete_columns = ["Gender", "Major"]
data["Gender"] = data["Gender"].astype(str)
data["Major"] = data["Major"].astype(str)

#STEP2: Train CTGAN and generate

print("[2/5] Training CTGAN ")
# training was based heavily off of the sample on CTGAN github and modified for my use
ctgan = CTGAN(epochs=300, verbose=True)
ctgan.fit(data, discrete_columns)
ctgan.save("ctgan_model.pkl")
# To skip retraining comment 3 lines above and uncomment below
# ctgan = CTGAN.load("ctgan_model.pkl")

print("    Sampling data")
synthetic_data = ctgan.sample(len(data))

# process bounds
synthetic_data["Age"] =                  synthetic_data["Age"].round().clip(18, 24).astype(int)
synthetic_data["Social_Hours_Week"] =    synthetic_data["Social_Hours_Week"].round().clip(0, 20).astype(int)
synthetic_data["Attendance_Pct"] =       synthetic_data["Attendance_Pct"].clip(0, 100)
synthetic_data["Study_Hours_Per_Day"] =  synthetic_data["Study_Hours_Per_Day"].clip(0.1, 14.0)
synthetic_data["Previous_GPA"] =         synthetic_data["Previous_GPA"].clip(0, 4)
synthetic_data["Sleep_Hours"] =          synthetic_data["Sleep_Hours"].clip(4, 10)
synthetic_data["Final_CGPA"] =           synthetic_data["Final_CGPA"].clip(0, 4)

synthetic_data.to_csv("synthetic_students.csv", index=False)
print("    Saved to synthetic_students.csv")

# Normalize column names
def normalize_columns(df):
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

real_df  = normalize_columns(data)
synth_df = normalize_columns(synthetic_data)

common_cols  = [c for c in real_df.columns if c in synth_df.columns]
numeric_cols = [
    c for c in common_cols
    if pd.api.types.is_numeric_dtype(real_df[c]) and pd.api.types.is_numeric_dtype(synth_df[c])
]

#STEP3 utility metrics
print("\n[3/5] Utility evaluation")
#utilized "roenws" reference code for normalizing data
def build_hist_probs(real_vals, synth_vals, bins=20):
    lo = float(min(np.nanmin(real_vals), np.nanmin(synth_vals)))
    hi = float(max(np.nanmax(real_vals), np.nanmax(synth_vals)))
    if np.isclose(lo, hi):
        hi = lo + 1e-6
    real_counts, edges = np.histogram(real_vals, bins=bins, range=(lo, hi))
    synth_counts, _    = np.histogram(synth_vals, bins=edges)
    real_probs  = real_counts.astype(float)  + 1e-10
    synth_probs = synth_counts.astype(float) + 1e-10
    real_probs  /= real_probs.sum()
    synth_probs /= synth_probs.sum()
    return real_counts, synth_counts, real_probs, synth_probs

per_feature_rows = []
for col in numeric_cols:
    real_vals  = pd.to_numeric(real_df[col],  errors='coerce').dropna().to_numpy()
    synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna().to_numpy()
    if len(real_vals) == 0 or len(synth_vals) == 0:
        continue

    real_counts, synth_counts, real_probs, synth_probs = build_hist_probs(real_vals, synth_vals)

    kl_div = float(entropy(real_probs, synth_probs))
    js_div = float(jensenshannon(real_probs, synth_probs) ** 2)
    w_dist = float(wasserstein_distance(real_vals, synth_vals))

    contingency    = np.vstack([real_counts, synth_counts])
    non_empty_bins = contingency.sum(axis=0) > 0
    contingency    = contingency[:, non_empty_bins]

    if contingency.shape[1] >= 2:
        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
    else:
        chi2_stat, chi2_p = np.nan, np.nan

    per_feature_rows.append({
        'feature':                   col,
        'kl_divergence':             kl_div,
        'jensen_shannon_divergence': js_div,
        'wasserstein_distance':      w_dist,
        'chi_square_stat':           float(chi2_stat),
        'chi_square_p_value':        float(chi2_p),
    })

utility_df = pd.DataFrame(per_feature_rows)

#logistic regression real vs synthetic
min_size     = min(len(real_df), len(synth_df))
real_sample  = real_df[numeric_cols].sample(n=min_size, random_state=42, replace=False)
synth_sample = synth_df[numeric_cols].sample(n=min_size, random_state=42, replace=False)
combined     = pd.concat([real_sample, synth_sample], axis=0, ignore_index=True)
combined     = combined.apply(pd.to_numeric, errors='coerce').fillna(combined.median(numeric_only=True))
labels       = np.concatenate([np.ones(min_size), np.zeros(min_size)])

X_tr, X_te, y_tr, y_te = train_test_split(
    combined.values, labels, test_size=0.3, random_state=42, stratify=labels
)
clf_model = Pipeline([('scaler', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=1000, random_state=42))])
clf_model.fit(X_tr, y_tr)
logreg_auc = float(roc_auc_score(y_te, clf_model.predict_proba(X_te)[:, 1]))

utility_summary = {
    'feature_metric_means': utility_df.drop(columns=['feature']).mean().to_dict(),
    'logistic_regression_auc_real_vs_synth': logreg_auc,
}

print("\n=== Utility Metrics (Per Feature) ===")
print(utility_df.to_string(index=False))
print("\n=== Utility Summary ===")
print(json.dumps(utility_summary, indent=2))

#STEP4: Privacy metrics
print("\n[4/5] Privacy evaluation")

#Delta presence exact tuple overlap ratio
real_keys      = set(real_df[common_cols].astype(str).agg('|'.join, axis=1).unique())
synth_keys     = set(synth_df[common_cols].astype(str).agg('|'.join, axis=1).unique())
overlap        = len(real_keys.intersection(synth_keys))
delta_presence = float(overlap / max(len(real_keys), 1))

#k anonymity over quasi identifiers
qid_cols = [c for c in ['age', 'gender', 'major', 'attendance_pct'] if c in common_cols]
if not qid_cols:
    qid_cols = common_cols[:min(4, len(common_cols))]

synth_qid = synth_df[qid_cols].copy()
for col in qid_cols:
    if col in numeric_cols:
        synth_qid[col] = pd.qcut(
            pd.to_numeric(synth_qid[col], errors='coerce'),
            q=10, duplicates='drop'
        ).astype(str)
    else:
        synth_qid[col] = synth_qid[col].astype(str)

eq_class_sizes = synth_qid.groupby(qid_cols).size()
k_min    = int(eq_class_sizes.min())      if len(eq_class_sizes) else 0
k_median = float(eq_class_sizes.median()) if len(eq_class_sizes) else 0.0
pct_lt5  = float((eq_class_sizes < 5).mean()) if len(eq_class_sizes) else 1.0

#Identifiability by nearest neighbor normalized distance
real_num  = real_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
synth_num = synth_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
fill_vals = real_num.median(numeric_only=True)
real_num  = real_num.fillna(fill_vals)
synth_num = synth_num.fillna(fill_vals)

scaler       = StandardScaler()
real_scaled  = scaler.fit_transform(real_num)
synth_scaled = scaler.transform(synth_num)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(real_scaled)
distances, _          = nn.kneighbors(synth_scaled)
identifiability_score = float((distances.flatten() <= 0.5).mean())

privacy_summary = {
    'delta_presence_overlap_ratio':             delta_presence,
    'k_anonymity_min_equivalence_class':        k_min,
    'k_anonymity_median_equivalence_class':     k_median,
    'k_anonymity_pct_classes_lt_5':             pct_lt5,
    'identifiability_score_nn_distance_le_0_5': identifiability_score,
    'quasi_identifier_columns_used':            qid_cols,
}

print("\n    Privacy Summary ")
print(json.dumps(privacy_summary, indent=2))

# Save metrics JSON
metrics_payload = {
    'utility_metrics_by_feature': utility_df.to_dict(orient='records'),
    'utility_summary':  utility_summary,
    'privacy_summary':  privacy_summary,
}
with open("privacy_utility.json", "w") as fp:
    json.dump(metrics_payload, fp, indent=2)
print("\n    Saved privacy_utility.json")


#STEP5: plots
print("\n[5/5] Generating plots")

FEATURES_FOR_PLOTS = ['age', 'study_hours_per_day', 'attendance_pct',
                      'previous_gpa', 'sleep_hours', 'final_cgpa']

plot_features = [c for c in FEATURES_FOR_PLOTS if c in numeric_cols]
extras        = [c for c in numeric_cols if c not in plot_features]
plot_features.extend(extras[:max(0, 6 - len(plot_features))])
plot_features = plot_features[:6]

mpl.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure(constrained_layout=True)
gs  = fig.add_gridspec(2, 6)

for i, col in enumerate(plot_features):
    ax         = fig.add_subplot(gs[0, i])
    real_vals  = pd.to_numeric(real_df[col],  errors='coerce').dropna()
    synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna()
    ax.hist(real_vals,  bins=20, density=True, alpha=0.55, label='Real')
    ax.hist(synth_vals, bins=20, density=True, alpha=0.55, label='Synthetic')
    ax.set_title(col, fontsize=10)
    if i == 0:
        ax.legend(fontsize=8)

corr_features = [c for c in FEATURES_FOR_PLOTS if c in numeric_cols]
if len(corr_features) < 2:
    corr_features = numeric_cols[:min(6, len(numeric_cols))]

real_corr  = real_df[corr_features].corr(numeric_only=True)
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
        ax_real.text(c, r, f'{real_corr.values[r, c]:.2f}',
                     ha='center', va='center', fontsize=7, color='white')

ax_synth = fig.add_subplot(gs[1, 3:])
ax_synth.imshow(synth_corr.values, vmin=-1, vmax=1, cmap=CORR_CMAP)
ax_synth.set_title('Synthetic Data Correlations')
ax_synth.set_xticks(range(len(corr_features)))
ax_synth.set_xticklabels(corr_features, rotation=90)
ax_synth.set_yticks(range(len(corr_features)))
ax_synth.set_yticklabels(corr_features)
for r in range(synth_corr.shape[0]):
    for c in range(synth_corr.shape[1]):
        ax_synth.text(c, r, f'{synth_corr.values[r, c]:.2f}',
                      ha='center', va='center', fontsize=7, color='white')

fig.colorbar(im1, ax=[ax_real, ax_synth], shrink=0.8)
fig.suptitle('Real vs Synthetic: Distribution and Correlation Comparison',
             fontsize=14, fontweight='bold')

fig.savefig("comparison.png", dpi=150)
plt.close(fig)
print("    Saved comparison.png")

print("\nFinished exiting")
