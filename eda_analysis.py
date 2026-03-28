"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    df = pd.read_csv(filepath)

    # --- Handle missing values ---
    # commute_minutes: 181 missing (~9%), assumed MCAR → impute with median
    df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())

    # scholarship: 389 missing (~19.5%), likely MNAR (students with no
    # scholarship have no entry) → fill with "None" to treat as a category
    df["scholarship"] = df["scholarship"].fillna("None")

    # --- Write profile report ---
    lines = []
    lines.append("=== DATA PROFILE REPORT ===\n")

    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    lines.append("\n--- Data Types ---")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")

    lines.append("\n\n--- Missing Values (after cleaning) ---")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = count / len(df) * 100
        lines.append(f"  {col}: {count} ({pct:.1f}%)")

    lines.append("\n\n--- Missing Value Handling Decisions ---")
    lines.append(
        "  commute_minutes: 181 missing (~9%). Assumed MCAR (no systematic pattern)."
        " Imputed with median (25 min) to preserve row count without skewing distribution."
    )
    lines.append(
        "  scholarship: 389 missing (~19.5%). Likely MNAR — students without any"
        " scholarship have no entry. Filled with 'None' to preserve the group as a"
        " meaningful category rather than dropping nearly 20% of rows."
    )

    lines.append("\n\n--- Descriptive Statistics (numeric columns) ---")
    lines.append(df.describe().to_string())

    with open("output/data_profile.txt", "w") as f:
        f.write("\n".join(lines))

    print("✓ Data profile saved to output/data_profile.txt")
    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    # 1. Histogram + KDE for GPA
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["gpa"], kde=True, bins=30, color="steelblue", ax=ax)
    ax.set_title("GPA Distribution — Roughly Normal, Centred Around 2.7")
    ax.set_xlabel("GPA")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/dist_gpa.png", dpi=150)
    plt.close()

    # 2. Histogram + KDE for study_hours_weekly
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["study_hours_weekly"], kde=True, bins=30, color="darkorange", ax=ax)
    ax.set_title("Weekly Study Hours — Right-Skewed, Most Students Study 10–20 hrs")
    ax.set_xlabel("Study Hours (Weekly)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/dist_study_hours.png", dpi=150)
    plt.close()

    # 3. Histogram + KDE for attendance_pct
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["attendance_pct"], kde=True, bins=30, color="seagreen", ax=ax)
    ax.set_title("Attendance % — Spread Evenly, No Strong Skew")
    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/dist_attendance.png", dpi=150)
    plt.close()

    # 4. Box plot: GPA by department
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("department")["gpa"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="department", y="gpa", order=order,
                hue="department", palette="Set2", legend=False, ax=ax)
    ax.set_title("GPA by Department — Engineering and CS Median GPA Slightly Higher")
    ax.set_xlabel("Department")
    ax.set_ylabel("GPA")
    plt.tight_layout()
    plt.savefig("output/gpa_by_department.png", dpi=150)
    plt.close()

    # 5. Bar chart: scholarship status counts
    fig, ax = plt.subplots(figsize=(8, 5))
    order_s = df["scholarship"].value_counts().index
    sns.countplot(data=df, x="scholarship", order=order_s,
                  hue="scholarship", palette="pastel", legend=False, ax=ax)
    ax.set_title("Scholarship Status Distribution — ~20% of Students Have No Scholarship")
    ax.set_xlabel("Scholarship Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("output/dist_scholarship.png", dpi=150)
    plt.close()

    print("✓ Distribution plots saved to output/")


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    numeric_cols = ["course_load", "study_hours_weekly", "gpa",
                    "attendance_pct", "commute_minutes"]
    corr = df[numeric_cols].corr()

    # 1. Annotated heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix — Numeric Variables")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png", dpi=150)
    plt.close()

    # Find top 2 correlated pairs (excluding self-correlations)
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .abs()
        .sort_values(ascending=False)
    )
    top_pairs = corr_pairs.head(2).index.tolist()

    # 2 & 3. Scatter plots for top correlated pairs
    for i, (col_a, col_b) in enumerate(top_pairs, start=1):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(df[col_a], df[col_b], alpha=0.3, edgecolors="none", color="royalblue")
        m, b = np.polyfit(df[col_a], df[col_b], 1)
        x_line = np.linspace(df[col_a].min(), df[col_a].max(), 100)
        ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.5)
        r = corr.loc[col_a, col_b]
        ax.set_title(f"{col_a} vs {col_b}  (r = {r:.2f})")
        ax.set_xlabel(col_a)
        ax.set_ylabel(col_b)
        plt.tight_layout()
        plt.savefig(f"output/scatter_{col_a}_vs_{col_b}.png", dpi=150)
        plt.close()

    print("✓ Correlation plots saved to output/")
    return corr, top_pairs


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """
    results = {}

    # --- Hypothesis 1: Internship vs GPA (independent t-test) ---
    gpa_intern = df.loc[df["has_internship"] == "Yes", "gpa"]
    gpa_no_intern = df.loc[df["has_internship"] == "No", "gpa"]
    t_stat, p_val = stats.ttest_ind(gpa_intern, gpa_no_intern)

    # Cohen's d
    pooled_std = np.sqrt(
        (gpa_intern.std() ** 2 + gpa_no_intern.std() ** 2) / 2
    )
    cohens_d = (gpa_intern.mean() - gpa_no_intern.mean()) / pooled_std

    results["internship_ttest"] = {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "cohens_d": round(cohens_d, 4),
        "mean_intern": round(gpa_intern.mean(), 3),
        "mean_no_intern": round(gpa_no_intern.mean(), 3),
    }

    print("\n=== Hypothesis 1: Internship → GPA (Independent t-test) ===")
    print(f"  Mean GPA (internship=Yes): {gpa_intern.mean():.3f}")
    print(f"  Mean GPA (internship=No):  {gpa_no_intern.mean():.3f}")
    print(f"  t-statistic: {t_stat:.4f},  p-value: {p_val:.4f}")
    print(f"  Cohen's d:   {cohens_d:.4f}")
    if p_val < 0.05:
        print("  → Statistically significant (p < 0.05). Students with internships"
              " have a meaningfully different GPA.")
    else:
        print("  → Not statistically significant (p ≥ 0.05).")
    if abs(cohens_d) < 0.2:
        print("  → Effect size is negligible (|d| < 0.2). Practical impact is minimal.")
    elif abs(cohens_d) < 0.5:
        print("  → Effect size is small (0.2 ≤ |d| < 0.5).")
    else:
        print("  → Effect size is medium or larger (|d| ≥ 0.5).")

    # --- Hypothesis 2: Scholarship vs Department (chi-square test) ---
    contingency = pd.crosstab(df["scholarship"], df["department"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    results["scholarship_dept_chi2"] = {
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_chi2, 4),
        "degrees_of_freedom": dof,
    }

    print("\n=== Hypothesis 2: Scholarship ↔ Department (Chi-square test) ===")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  p-value:              {p_chi2:.4f}")
    print(f"  Degrees of freedom:   {dof}")
    if p_chi2 < 0.05:
        print("  → Statistically significant (p < 0.05). Scholarship type is"
              " associated with department.")
    else:
        print("  → Not statistically significant (p ≥ 0.05). No strong association"
              " between scholarship type and department.")

    return results


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    df = load_and_profile("data/student_performance.csv")
    plot_distributions(df)
    corr, top_pairs = plot_correlations(df)
    test_results = run_hypothesis_tests(df)

    # --- Compute a few numbers needed for the report ---
    numeric_cols = ["course_load", "study_hours_weekly", "gpa",
                    "attendance_pct", "commute_minutes"]
    corr_matrix = df[numeric_cols].corr()
    corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    top_pos = corr_pairs.head(1)
    top_neg = corr_pairs.tail(1)

    tt = test_results["internship_ttest"]
    ch = test_results["scholarship_dept_chi2"]

    findings = f"""# FINDINGS REPORT — Student Performance EDA

## 1. Dataset Description

- **Shape:** 2,000 rows × 10 columns
- **Columns:** student_id, department, semester, course_load, study_hours_weekly,
  gpa, attendance_pct, has_internship, commute_minutes, scholarship
- **Data quality issues:**
  - `commute_minutes`: 181 missing values (~9%). Imputed with the median (25 min),
    assumed Missing Completely At Random (MCAR).
  - `scholarship`: 389 missing values (~19.5%). Likely Missing Not At Random (MNAR) —
    students without a scholarship had no entry. Filled with `"None"` to preserve
    them as a meaningful group.

---

## 2. Key Distribution Findings

- **GPA** (`output/dist_gpa.png`): Roughly bell-shaped, centred near 2.7 with a
  slight right tail. Most students fall in the 2.0–3.5 range.
- **Study hours** (`output/dist_study_hours.png`): Right-skewed — the bulk of
  students study 10–20 hours per week, but a small group studies 25+ hrs.
- **Attendance %** (`output/dist_attendance.png`): Spread fairly evenly between
  ~40% and 100% with no strong skew, suggesting diverse attendance habits.
- **GPA by department** (`output/gpa_by_department.png`): Median GPAs are similar
  across departments (≈ 2.6–2.8). Engineering and Computer Science show slightly
  higher medians, but overlapping IQRs indicate the difference is modest.
- **Scholarship** (`output/dist_scholarship.png`): Merit, Athletic, Need-based, and
  Department scholarships are distributed roughly equally. Students with no scholarship
  (~20%) form the largest single group after the split.

---

## 3. Notable Correlations

See `output/correlation_heatmap.png` for the full Pearson matrix.

| Pair | r |
|------|---|
| study_hours_weekly ↔ gpa | {corr_matrix.loc['study_hours_weekly', 'gpa']:.2f} |
| attendance_pct ↔ gpa | {corr_matrix.loc['attendance_pct', 'gpa']:.2f} |
| commute_minutes ↔ gpa | {corr_matrix.loc['commute_minutes', 'gpa']:.2f} |
| course_load ↔ gpa | {corr_matrix.loc['course_load', 'gpa']:.2f} |

**Most correlated pair:** {top_pos.index[0][0]} ↔ {top_pos.index[0][1]}
(r = {top_pos.iloc[0]:.2f}). See `output/scatter_{top_pos.index[0][0]}_vs_{top_pos.index[0][1]}.png`.

**Most negatively correlated pair:** {top_neg.index[0][0]} ↔ {top_neg.index[0][1]}
(r = {top_neg.iloc[0]:.2f}). See `output/scatter_{top_neg.index[0][0]}_vs_{top_neg.index[0][1]}.png`.

**Interpretation:**
- More study hours and higher attendance are both positively linked to GPA —
  consistent with expected academic effort patterns.
- Commute time shows a modest negative association, suggesting long commutes may
  reduce time available for studying.
- Correlations are moderate at best; no single factor dominates GPA.

> ⚠️ **Caveat:** Correlation ≠ causation. A student who studies more may already be
> more motivated, and motivation (unmeasured) could be the true driver of higher GPA.

---

## 4. Hypothesis Test Results

### Hypothesis 1 — Internship and GPA

**Hypothesis:** "Students with internships have a higher GPA than students without."
**Test:** Independent samples t-test (`scipy.stats.ttest_ind`)

| Metric | Value |
|--------|-------|
| Mean GPA (internship = Yes) | {tt['mean_intern']} |
| Mean GPA (internship = No) | {tt['mean_no_intern']} |
| t-statistic | {tt['t_statistic']} |
| p-value | {tt['p_value']} |
| Cohen's d | {tt['cohens_d']} |

**Interpretation:** {'The result is statistically significant (p < 0.05), indicating the GPA difference between groups is unlikely due to chance.' if tt['p_value'] < 0.05 else 'The result is not statistically significant (p ≥ 0.05).'}
Cohen's d = {tt['cohens_d']:.3f} indicates a {'negligible' if abs(tt['cohens_d']) < 0.2 else 'small' if abs(tt['cohens_d']) < 0.5 else 'medium'} practical effect size.
The GPA gap between groups is real but small in magnitude.

---

### Hypothesis 2 — Scholarship Status and Department

**Hypothesis:** "Scholarship status is associated with department."
**Test:** Chi-square test of independence (`scipy.stats.chi2_contingency`)

| Metric | Value |
|--------|-------|
| Chi-square statistic | {ch['chi2_statistic']} |
| p-value | {ch['p_value']} |
| Degrees of freedom | {ch['degrees_of_freedom']} |

**Interpretation:** {'The result is statistically significant (p < 0.05). Scholarship type is not distributed equally across departments.' if ch['p_value'] < 0.05 else 'The result is not statistically significant (p ≥ 0.05). Scholarship type appears to be distributed independently of department.'}

---

## 5. Actionable Recommendations

1. **Expand study-support programmes targeting low-attendance students.**
   Attendance has a positive correlation with GPA. Students attending less than
   70% of classes likely need structured support (e.g., tutoring, check-ins),
   as this group shows lower academic outcomes.

2. **Investigate commute barriers and offer remote/hybrid options.**
   Longer commute times are negatively associated with GPA. The university should
   survey students commuting 45+ minutes and consider hybrid scheduling or shuttle
   services to reduce this friction.

3. **Re-examine internship eligibility criteria.**
   Students with internships show slightly higher GPAs on average. However, the
   effect size is small, suggesting that internship experience alone does not
   guarantee academic improvement. A structured co-op model (tying internships to
   academic credit and mentoring) may reinforce the benefit more reliably.
"""

    with open("output/FINDINGS.md", "w") as f:
        f.write(findings)

    print("\n✓ FINDINGS.md saved to output/FINDINGS.md")
    print("\n=== EDA pipeline complete. All outputs are in output/ ===")


if __name__ == "__main__":
    main()