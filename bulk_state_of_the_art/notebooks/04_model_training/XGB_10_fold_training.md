# XGBoost Drug Response Prediction with 10-Fold CV (Concatenated Predictions)

---

## 1. Methodology Recap

* **Model:** XGBoost Regressor
* **Data:** Drug-specific PCA-reduced gene expression vs. LN\_IC50
* **Cross-Validation:**

  * 10-fold CV using **concatenated test predictions**.
  * For each fold:

    * Train on 9/10 folds.
    * Predict on the held-out fold.
    * Concatenate predictions across folds to create `y_pred` aligned with `y_true`.
* **Metrics Computed Per Drug:**

  * Global R²
  * Pearson r with p-value
  * RMSE
* **Correlation Analyses:**

  * Pearson r vs. sample size per drug
  * Pearson r vs. IC50 variance per drug

---

## 2. Results Summary

### Overall Predictive Performance

* **Pearson r values across drugs:**

  * Distributed mainly between \~0.3 and \~0.6.
  * Peak around 0.45-0.5, indicating **moderate predictive signal**.
  * Very few drugs below 0.2, suggesting most drugs retain predictive information.
* **Global R² values:**

  * Concentrated between 0.1 and 0.3, with some up to \~0.5.
  * Confirms XGBoost captures a meaningful portion of variance across many drugs.
* **RMSE values:**

  * Center around 1.0–1.2, indicating moderate error relative to the LN\_IC50 SD.

### Correlation with Sample Size

* Pearson r (samples vs R²): 0.225 (p = 1.81e-09)
* Spearman ρ (samples vs R²): 0.221 (p = 3.94e-09)
* Pearson r (samples vs Pearson r): 0.213 (p = 1.42e-08)
* Spearman ρ (samples vs Pearson r): 0.211 (p = 1.84e-08)

**Interpretation:**

* A **weak but significant positive correlation** between the number of samples and predictability.
* Indicates larger datasets help but are not the primary determinant of predictive success.

### Correlation with IC50 Variance

* Pearson r (IC50 variance vs Pearson r): 0.270 (p = 4.29e-13)
* Spearman ρ (IC50 variance vs Pearson r): 0.271 (p = 3.11e-13)

**Interpretation:**

* Drugs with higher IC50 variance are generally easier to predict.
* Consistent with the expectation that low variance restricts achievable correlation.

---

## 3. Visualizations Generated

* **Histograms:** RMSE, R², and Pearson r distributions across drugs.
* **Scatter Plots:**

  * Pearson r vs. Sample Size
  * Pearson r vs. IC50 Variance
  * Global R² vs. Sample Size
  * Global R² vs. IC50 Variance
  * Global R² vs. Pearson r for consistency validation

These plots validate metric consistency and confirm moderate, systematic predictability across most drugs.

---

## 4. Practical Interpretation of Pearson r in This Context

* **0.1–0.3:** Weak but meaningful signals, common in pharmacogenomics.
* **0.3–0.5:** Moderate predictive power, suggesting capture of biologically relevant signals.
* **>0.5:** Strong predictive signals, often aligned with drugs linked to clear biomarkers.
* Low p-values confirm the significance of these correlations, but focus should remain on r magnitude for practical evaluation.

### RMSE and R² Context:

* RMSE should be compared to drug-specific IC50 variance.
* R² represents the proportion of variance explained; low R² can still coexist with meaningful Pearson r due to biological noise.

---

## 5. Additional Results Interpretation

### Pearson r vs. IC50 Variance

The positive correlation indicates that drugs with greater IC50 variability are more predictable, as low variance limits potential correlation values.

### Global R² vs. Pearson r

Plots demonstrate a strong, nearly monotonic relationship, confirming metric alignment and consistency across drugs.

### Top and Bottom Drugs Analysis

* **Top drugs:** High Pearson r (\~0.6–0.7) and R² (\~0.4–0.5) suggest strong predictability and biological signal capture.
* **Bottom drugs:** Low Pearson r (\~0.1–0.2) and negative or near-zero R² likely reflect low variance or high noise.

### Feature Importance (Next Step)

* Extracting SHAP values or feature importances from XGBoost can reveal which PCs or covariates drive predictability in top drugs.
* This can guide biomarker prioritization or dimensionality reduction refinement.

---

## 6. Next Recommended Actions

* Compute **top and bottom 5 drugs by Pearson r** for qualitative investigation.
* Compare **feature contributions across models** to understand non-linear capture in XGBoost.
* Consider **variance filtering before PCA** to improve signal.
* Prepare **clean slide figures** for lab presentations.
* Begin drafting **systematic model comparison (Elastic Net vs. RF vs. XGBoost)** to finalize the benchmarking phase.
