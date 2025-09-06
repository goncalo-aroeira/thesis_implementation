# Random Forest Drug Response Prediction with 10-Fold CV (Concatenated Predictions)

---

## 1. Methodology Recap

- **Model:** Random Forest Regressor
- **Data:** Drug-specific gene expression vs. LN_IC50
- **Cross-Validation:**
  - 10-fold CV with **concatenated test predictions** rather than fold-wise R².
  - For each fold:
    - Train on 9/10 folds.
    - Predict on the held-out fold.
    - Concatenate predictions across all folds to form `y_pred` aligned with `y_true`.
- **Metrics Computed Per Drug:**
  - Global R²
  - Pearson r with p-value
  - RMSE
- **Correlation Analyses:**
  - Pearson r vs. number of samples per drug.
  - Pearson r vs. IC50 variance per drug.

---

## 2. Results Summary

### Overall Predictive Performance

- **Pearson r values across drugs:**
  - Distributed mostly between ~0.3 and ~0.6.
  - Peak around 0.45-0.5, indicating **moderate predictive signal**.
  - Few drugs below 0.2, indicating features have predictive value for most drugs.

### Correlation with Sample Size

Pearson r (sample size vs Pearson r) = 0.160 (p = 2.35e-05)
Spearman ρ (sample size vs Pearson r) = 0.176 (p = 3.07e-06)


**Interpretation:**

- There is a **weak but statistically significant positive correlation** between the number of samples per drug and predictive performance.
- This suggests **increasing samples alone yields small improvements**, emphasizing that feature structure and biological signal quality may have greater impact.

---

## 3. Visualizations Generated

- **Histogram of Pearson r across drugs:** Shows a moderate prediction capability across the dataset.
- **Scatter plot of Pearson r vs. Sample Size:** Confirms the weak positive relationship.
- **Scatter plot of Pearson r vs. IC50 Variance:** Now generated.
- **Scatter plot of Global R² vs. Pearson r:** Now generated for metric consistency validation.
- **Top and bottom 5 drugs by Pearson r identified.**
- **Feature importances extracted for the best predicted drug.**

---

## 4. Practical Interpretation of Pearson r in This Context

- **Pearson r ~0.1-0.3:** Weak but meaningful signal (common in pharmacogenomics).
- **Pearson r ~0.3-0.5:** Moderate predictive strength, indicating your model captures relevant biological signal.
- **Pearson r >0.5:** Strong signal, less common, often linked to known biomarkers.
- **p-values:** Low p-values confirm the significance of the correlation but practical interpretation should focus on r magnitude.

**Relationship with RMSE and R²:**

- RMSE should be compared to the IC50 SD per drug for context.
- R² confirms the proportion of variance explained; often low in biological data even with meaningful Pearson r.

---

## 5. Additional Results Interpretation

### Pearson r vs. IC50 Variance

A weak-to-moderate positive correlation was found:

Pearson r (IC50 variance vs Pearson r) = 0.270 (p = 5.02e-13)
Spearman ρ (IC50 variance vs Pearson r) = 0.269 (p = 6.24e-13)


This indicates that drugs with higher variance in LN_IC50 tend to have better predictability (higher Pearson r), which is expected since low variance limits possible correlation values.

### Global R² vs. Pearson r

The scatter plot shows a near-perfect monotonic relationship, confirming that drugs with higher R² also exhibit higher Pearson r values, validating metric consistency.

### Top and Bottom 5 Drugs by Predictability

- **Top 5 drugs (highest Pearson r ~0.71-0.73, R² ~0.50-0.54):**
  These drugs are highly predictable, indicating strong signal capture by the model.

- **Bottom 5 drugs (Pearson r ~0.08-0.15, R² negative or near zero):**
  These drugs are poorly predictable, likely due to low variance, insufficient signal, or high noise.

### Feature Importances (Best Drug)

For drug `2540` (best predicted drug):

- Top features included `SCF_PC3`, `day4_day1_ratio`, and specific principal components.
- `SCF_PC3` alone contributed ~28% of importance, indicating it may be a key biomarker for this drug’s response prediction.

These results provide a clear direction for focusing on variance, strong feature signals, and prioritizing drugs where models capture meaningful biological signals for further analysis and reporting.

---

## 6. Next Recommended Actions

- Compute **top and bottom 5 drugs by Pearson r** for qualitative investigation.
- Compare **feature importances** across well and poorly predicted drugs.
- Optionally prepare **slide-ready summary figures** for lab presentations.
- Consider advanced models or feature selection for improved performance on hard-to-predict drugs.
