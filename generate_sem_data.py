"""
Generate 304 synthetic survey responses for PLS-SEM analysis.

Research Framework:
- BNPL Usage Intensity (BNPLU) → Impulsive Buying Behavior (IMPULSE)
- Consumer Trust in BNPL (TRUST) → IMPULSE
- Perceived Convenience & Flexibility (CONV) → IMPULSE
- Financial Literacy (FIN. LIT) moderates above paths (negative moderator)
- Materialistic Values (MATERIALISM) moderates above paths (positive moderator)

All construct items use 1-5 Likert scale.
Reverse-coded items: MATERIALISM 4, IMPULSE 5.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=42)

N = 304  # number of respondents

# ---------------------------------------------------------------------------
# Step 1: Generate correlated latent factors
# ---------------------------------------------------------------------------
# 5 exogenous latent factors: bnpl, trust, conv, mat, finlit
# Correlation matrix (between-construct: 0.25-0.45)
corr_exog = np.array([
    # bnpl  trust  conv   mat   finlit
    [1.00,  0.40,  0.38,  0.30, -0.15],   # bnpl
    [0.40,  1.00,  0.42,  0.28, -0.12],   # trust
    [0.38,  0.42,  1.00,  0.32, -0.12],   # conv
    [0.30,  0.28,  0.32,  1.00, -0.18],   # mat
    [-0.15, -0.12, -0.12, -0.18, 1.00],   # finlit
])

# Cholesky decomposition to generate correlated normals
L_chol = np.linalg.cholesky(corr_exog)
Z = RNG.standard_normal((N, 5))
factors = Z @ L_chol.T  # shape (N, 5)

L_bnpl   = factors[:, 0]
L_trust  = factors[:, 1]
L_conv   = factors[:, 2]
L_mat    = factors[:, 3]
L_finlit = factors[:, 4]

# ---------------------------------------------------------------------------
# Step 2: Structural model for IMPULSE (dependent latent variable)
# ---------------------------------------------------------------------------
error_impulse = RNG.standard_normal(N) * 0.38

L_impulse = (
    0.22 * L_bnpl
    + 0.22 * L_trust
    + 0.22 * L_conv
    + 0.15 * L_mat
    - 0.15 * L_finlit
    + 0.13 * (L_mat * L_bnpl)
    + 0.13 * (L_mat * L_trust)
    + 0.13 * (L_mat * L_conv)
    - 0.13 * (L_finlit * L_bnpl)
    - 0.13 * (L_finlit * L_trust)
    - 0.13 * (L_finlit * L_conv)
    + error_impulse
)

# Standardize L_impulse so loadings behave uniformly
L_impulse = (L_impulse - L_impulse.mean()) / L_impulse.std()

# ---------------------------------------------------------------------------
# Step 3: Generate reflective indicators from each latent factor
# ---------------------------------------------------------------------------

def make_indicators(latent: np.ndarray, n_items: int, lambdas: np.ndarray) -> np.ndarray:
    """
    Generate n_items reflective indicators for a latent factor.

    indicator_i = lambda_i * latent + sqrt(1 - lambda_i^2) * unique_error
    """
    indicators = np.zeros((len(latent), n_items))
    for i, lam in enumerate(lambdas):
        unique = RNG.standard_normal(len(latent))
        indicators[:, i] = lam * latent + np.sqrt(1 - lam ** 2) * unique
    return indicators


def to_likert(data: np.ndarray, cuts: list | None = None) -> np.ndarray:
    """
    Convert a continuous column to 1-5 Likert scale using percentile-based binning.

    Target approximate distribution: 7% 1s, 13% 2s, 22% 3s, 33% 4s, 25% 5s
    """
    if cuts is None:
        cuts = [7, 20, 42, 75]  # percentile cut-points → 4 thresholds → 5 bins
    result = np.ones(len(data), dtype=int)
    thresholds = [np.percentile(data, p) for p in cuts]
    for score, thr in enumerate(thresholds, start=2):
        result[data >= thr] = score
    return result


# Lambda values for each construct (vary between 0.87-0.93)
lam_bnpl    = np.array([0.91, 0.89, 0.92, 0.88, 0.90, 0.87])
lam_trust   = np.array([0.90, 0.92, 0.88, 0.91, 0.89, 0.93])
lam_conv    = np.array([0.89, 0.91, 0.87, 0.90, 0.92, 0.88, 0.91, 0.89, 0.90])
lam_mat     = np.array([0.90, 0.88, 0.91, 0.92, 0.89, 0.87, 0.90, 0.93, 0.88])
lam_finlit  = np.array([0.91, 0.89, 0.92])
lam_impulse = np.array([0.90, 0.92, 0.88, 0.91, 0.89, 0.93])

# Generate continuous indicators
raw_bnpl    = make_indicators(L_bnpl,    6, lam_bnpl)
raw_trust   = make_indicators(L_trust,   6, lam_trust)
raw_conv    = make_indicators(L_conv,    9, lam_conv)
raw_mat     = make_indicators(L_mat,     9, lam_mat)
raw_finlit  = make_indicators(L_finlit,  3, lam_finlit)
raw_impulse = make_indicators(L_impulse, 6, lam_impulse)

# ---------------------------------------------------------------------------
# Step 4: Convert to Likert 1-5 scale
# ---------------------------------------------------------------------------
cuts = [7, 20, 42, 75]

bnpl_likert    = np.column_stack([to_likert(raw_bnpl[:, i],    cuts) for i in range(6)])
trust_likert   = np.column_stack([to_likert(raw_trust[:, i],   cuts) for i in range(6)])
conv_likert    = np.column_stack([to_likert(raw_conv[:, i],    cuts) for i in range(9)])
mat_likert     = np.column_stack([to_likert(raw_mat[:, i],     cuts) for i in range(9)])
finlit_likert  = np.column_stack([to_likert(raw_finlit[:, i],  cuts) for i in range(3)])
impulse_likert = np.column_stack([to_likert(raw_impulse[:, i], cuts) for i in range(6)])

# ---------------------------------------------------------------------------
# Step 5: Apply reverse coding
# ---------------------------------------------------------------------------
# MATERIALISM 4 (index 3): "I usually buy only the things I need" — reverse coded
mat_likert[:, 3] = 6 - mat_likert[:, 3]

# IMPULSE 5 (index 4): "I only buy things that I really need" — reverse coded
impulse_likert[:, 4] = 6 - impulse_likert[:, 4]

# ---------------------------------------------------------------------------
# Step 6: Generate realistic demographics
# ---------------------------------------------------------------------------
# Age: 19-30, centred around 22-25
age_probs = np.array([0.04, 0.09, 0.15, 0.18, 0.17, 0.14, 0.09, 0.06, 0.04, 0.03, 0.01, 0.00])
age_probs = age_probs / age_probs.sum()
ages = RNG.choice(np.arange(19, 31), size=N, p=age_probs)

# Gender: roughly balanced
genders = RNG.choice(["Male", "Female"], size=N, p=[0.51, 0.49])

# Education Level
edu_levels = RNG.choice(
    ["Undergraduate", "Postgraduate", "Professional Degree"],
    size=N,
    p=[0.55, 0.30, 0.15],
)

# Occupation
occupations = RNG.choice(
    ["Student", "Salaried Employee", "Self-Employed"],
    size=N,
    p=[0.55, 0.35, 0.10],
)

# Monthly Income
incomes = RNG.choice(
    ["Nil", "10000", "15000", "20000", "25000", "30000", "35000", "40000", "50000"],
    size=N,
    p=[0.30, 0.12, 0.12, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04],
)

# ---------------------------------------------------------------------------
# Step 7: Assemble DataFrame
# ---------------------------------------------------------------------------
columns = [
    "Age", "Gender", "Education Level", "Occupation", "Monthly Income",
    "BNPLU 1", "BNPLU 2", "BNPLU 3", "BNPLU 4", "BNPLU 5", "BNPLU 6",
    "TRUST 1", "TRUST 2", "TRUST 3", "TRUST 4", "TRUST 5", "TRUST 6",
    "CONV. 1", "CONV. 2", "CONV. 3", "CONV. 4", "CONV. 5", "CONV. 6",
    "CONV. 7", "CONV. 8", "CONV. 9",
    "MATERIALISM 1", "MATERIALISM 2", "MATERIALISM 3", "MATERIALISM 4",
    "MATERIALISM 5", "MATERIALISM 6", "MATERIALISM 7", "MATERIALISM 8", "MATERIALISM 9",
    "FIN. LIT 1", "FIN. LIT 2", "FIN. LIT 3",
    "IMPULSE 1", "IMPULSE 2", "IMPULSE 3", "IMPULSE 4", "IMPULSE 5", "IMPULSE 6",
]

df = pd.DataFrame({
    "Age":             ages,
    "Gender":          genders,
    "Education Level": edu_levels,
    "Occupation":      occupations,
    "Monthly Income":  incomes,
})

# Add Likert items
for i in range(6):
    df[f"BNPLU {i+1}"]       = bnpl_likert[:, i]
for i in range(6):
    df[f"TRUST {i+1}"]       = trust_likert[:, i]
for i in range(9):
    df[f"CONV. {i+1}"]       = conv_likert[:, i]
for i in range(9):
    df[f"MATERIALISM {i+1}"] = mat_likert[:, i]
for i in range(3):
    df[f"FIN. LIT {i+1}"]    = finlit_likert[:, i]
for i in range(6):
    df[f"IMPULSE {i+1}"]     = impulse_likert[:, i]

# Reorder columns to match required order
df = df[columns]

# ---------------------------------------------------------------------------
# Step 8: Data quality verification
# ---------------------------------------------------------------------------
print("=== Data Quality Verification ===")
print(f"Rows: {len(df)}  (expected 304)")
print(f"Columns: {len(df.columns)}")

likert_cols = [c for c in df.columns if c not in
               ["Age", "Gender", "Education Level", "Occupation", "Monthly Income"]]

# Check value range
vals = df[likert_cols].values
assert vals.min() >= 1 and vals.max() <= 5, "Values outside 1-5 range!"
print(f"All Likert values in [1,5]: ✓")

# Check no constant columns
for col in likert_cols:
    assert df[col].nunique() > 1, f"Constant column detected: {col}"
print("No constant columns: ✓")

# Intra-construct correlations
def check_construct(name, cols_subset):
    corr_matrix = df[cols_subset].corr()
    off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    min_r = off_diag.min()
    print(f"  {name}: min intra-construct r = {min_r:.3f} {'✓' if min_r > 0.45 else '⚠'}")

print("\nIntra-construct correlations:")
check_construct("BNPLU",  [f"BNPLU {i}" for i in range(1, 7)])
check_construct("TRUST",  [f"TRUST {i}" for i in range(1, 7)])
check_construct("CONV.",  [f"CONV. {i}" for i in range(1, 10)])
# For MATERIALISM exclude item 4 (reverse-coded) for positive check
mat_fwd = [f"MATERIALISM {i}" for i in range(1, 10) if i != 4]
check_construct("MATERIALISM (fwd items)", mat_fwd)
check_construct("FIN. LIT", [f"FIN. LIT {i}" for i in range(1, 4)])
# For IMPULSE exclude item 5 (reverse-coded)
imp_fwd = [f"IMPULSE {i}" for i in range(1, 7) if i != 5]
check_construct("IMPULSE (fwd items)", imp_fwd)

# Reverse-coded checks
print("\nReverse-coded item checks:")
mat_other = [f"MATERIALISM {i}" for i in range(1, 10) if i not in [4]]
r_mat4 = df["MATERIALISM 4"].corr(df[mat_other].mean(axis=1))
print(f"  MATERIALISM 4 vs rest: r = {r_mat4:.3f} {'✓ (negative)' if r_mat4 < 0 else '⚠ (should be negative)'}")

imp_other = [f"IMPULSE {i}" for i in range(1, 7) if i not in [5]]
r_imp5 = df["IMPULSE 5"].corr(df[imp_other].mean(axis=1))
print(f"  IMPULSE 5 vs rest:     r = {r_imp5:.3f} {'✓ (negative)' if r_imp5 < 0 else '⚠ (should be negative)'}")

# Cross-construct check: FIN.LIT negatively correlated with IMPULSE
finlit_mean  = df[[f"FIN. LIT {i}" for i in range(1, 4)]].mean(axis=1)
impulse_mean = df[imp_fwd].mean(axis=1)
r_fl_imp = finlit_mean.corr(impulse_mean)
print(f"\nFIN.LIT vs IMPULSE correlation: {r_fl_imp:.3f} {'✓ (negative)' if r_fl_imp < 0 else '⚠'}")

mat_mean = df[mat_fwd].mean(axis=1)
r_mat_imp = mat_mean.corr(impulse_mean)
print(f"MATERIALISM vs IMPULSE correlation: {r_mat_imp:.3f} {'✓ (positive)' if r_mat_imp > 0 else '⚠'}")

# ---------------------------------------------------------------------------
# Step 9: Save CSV
# ---------------------------------------------------------------------------
output_path = "304_SEM_responses.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}  ({len(df)} rows × {len(df.columns)} columns)")
