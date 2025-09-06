# Elastic Net Drug Response Prediction with 10-Fold CV (Concatenated Predictions)

## 1. Methodology Recap

* **Model:** Elastic Net Regression
* **Data:** Drug-specific gene expression (PCA-reduced) vs. LN\_IC50
* **Cross-Validation:**

  * 10-fold CV with **concatenated predictions** across test folds.
  * Metrics computed globally per drug:

    * Global R²
    * Pearson r with p-value
    * RMSE
* **Correlation Analyses:**

  * Pearson r vs. number of samples
  * Pearson r vs. IC50 variance

## 2. Results Summary

### Overall Predictive Performance

* **Pearson r distribution:** Peaks \~0.4-0.5, with most drugs between 0.3-0.6, indicating **moderate linear predictability**.
* **R² distribution:** Concentrated between 0.1-0.3, with some up to \~0.5, reflecting partial variance capture.
* **RMSE distribution:** Peaks around 1.0-1.2, aligning with typical LN\_IC50 SD, showing moderate absolute prediction error.

### Correlation with Sample Size

* Pearson r (samples vs R²): 0.323 (p = 7.4e-26)
* Spearman ρ (samples vs R²): 0.363 (p = 1.38e-32)
* Pearson r (samples vs Pearson r): 0.347 (p = 1.08e-29)
* Spearman ρ (samples vs Pearson r): 0.360 (p = 3.9e-32)

**Interpretation:** There is a **moderate, statistically significant positive relationship** between sample size and predictability, indicating larger datasets help but are not the sole drivers of predictability.

### Correlation with IC50 Variance

* Pearson r (IC50 variance vs Pearson r): 0.175 (p = 2.3e-08)
* Spearman ρ (IC50 variance vs Pearson r): 0.162 (p = 2.42e-07)

**Interpretation:** Weak but significant correlation shows higher IC50 variance slightly aids predictability, but high variance alone does not guarantee accurate models.

## 3. Visualizations Generated

* **Histograms:** RMSE and R² distributions across drugs.
* **Scatter plots:**

  * Pearson r vs. sample size
  * Pearson r vs. IC50 variance
  * Global R² vs. sample size
  * Global R² vs. IC50 variance
  * Global R² vs. Pearson r (monotonic validation)

## 4. Practical Interpretation of Pearson r

* **0.1–0.3:** Weak but meaningful linear signal.
* **0.3–0.5:** Moderate predictive strength, indicating Elastic Net captures biologically relevant linear signals.
* **>0.5:** Strong but rare, often tied to known drug-specific biomarkers.
* Low p-values confirm statistical significance; focus on r’s magnitude for practical interpretation.

## 5. Implications for Your Project

✅ Elastic Net serves as a **strong baseline linear model** for drug response prediction.
✅ Non-linear methods (RF, XGBoost) may capture additional variance from interactions missed by Elastic Net.
✅ Applying variance filtering before PCA is a logical next step for signal focusing.
✅ Drugs with high/low predictability merit deeper biological inspection or data quality checks.

## 6. Next Recommended Actions

* Systematically compare Elastic Net vs. RF vs. XGBoost.
* Identify top/bottom 5 drugs by Pearson r for case studies.
* Extract feature weights from Elastic Net for interpretability.
* Optionally re-run with variance filtering before PCA.
* Prepare clean figures for lab slides and thesis documentation.

---

This detailed interpretation is ready for your lab documentation and thesis notes to maintain consistency with your RF analysis style.
