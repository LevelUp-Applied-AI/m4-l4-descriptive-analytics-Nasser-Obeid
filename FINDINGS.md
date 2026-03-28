# FINDINGS REPORT — Student Performance EDA

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
| study_hours_weekly ↔ gpa | 0.64 |
| attendance_pct ↔ gpa | 0.04 |
| commute_minutes ↔ gpa | 0.01 |
| course_load ↔ gpa | 0.01 |

**Most correlated pair:** study_hours_weekly ↔ gpa
(r = 0.64). See `output/scatter_study_hours_weekly_vs_gpa.png`.

**Most negatively correlated pair:** course_load ↔ commute_minutes
(r = -0.02). See `output/scatter_course_load_vs_commute_minutes.png`.

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
| Mean GPA (internship = Yes) | 2.983 |
| Mean GPA (internship = No) | 2.701 |
| t-statistic | 13.5644 |
| p-value | 0.0 |
| Cohen's d | 0.7061 |

**Interpretation:** The result is statistically significant (p < 0.05), indicating the GPA difference between groups is unlikely due to chance.
Cohen's d = 0.706 indicates a medium practical effect size.
The GPA gap between groups is real but small in magnitude.

---

### Hypothesis 2 — Scholarship Status and Department

**Hypothesis:** "Scholarship status is associated with department."
**Test:** Chi-square test of independence (`scipy.stats.chi2_contingency`)

| Metric | Value |
|--------|-------|
| Chi-square statistic | 17.1358 |
| p-value | 0.3769 |
| Degrees of freedom | 16 |

**Interpretation:** The result is not statistically significant (p ≥ 0.05). Scholarship type appears to be distributed independently of department.

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
