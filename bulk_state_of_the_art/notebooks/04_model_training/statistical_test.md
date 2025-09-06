# Paired t-Test for Model Performance Comparison

## 1. Goal of the Test

The **paired t-test** is used to determine whether the **performance difference between two models is statistically significant** when both are evaluated on the same set of samples — in this case, **per-drug prediction performance**.

Instead of just comparing average metrics, the paired t-test accounts for how consistently one model outperforms another **across individual drugs**.

---

## 2. When to Use It

Use a **paired t-test** when:
- You're comparing **two models**.
- Each model produces **one value per drug** for a specific metric (e.g., R², RMSE, or Pearson r).
- You want to know if the **mean difference in scores** (across drugs) is significantly different from zero.

---

## 3. Metrics to Compare

This test can be applied to any **continuous performance metric** computed **per drug**, such as:
- **R²**: Coefficient of determination (higher = better)
- **Pearson r**: Correlation between predicted and true IC50 (higher = better)
- **RMSE**: Root Mean Squared Error (lower = better)

---

## 4. Procedure

### Step-by-step:

1. **Align results**:
   - Ensure both models have predictions for the same set of drugs.
   - Each row should correspond to the same drug across models.

2. **Compute the differences**:
   - For each drug, calculate the difference in performance metric:
     ```
     Δ = Metric_Model_A - Metric_Model_B
     ```

3. **Perform the paired t-test**:
   - Use `scipy.stats.ttest_rel`:
     ```python
     from scipy.stats import ttest_rel
     t_stat, p_val = ttest_rel(model_a_scores, model_b_scores)
     ```

4. **Interpret the output**:
   - `t_stat`: The magnitude and direction of the difference.
   - `p_val`: The probability that the observed difference occurred by chance.

---

## 5. Interpreting Results

| Value      | Interpretation                                        |
|------------|--------------------------------------------------------|
| **t-statistic** | Positive: Model A > Model B (on average)             |
|                | Negative: Model B > Model A                          |
| **p-value**     | Low p-value → strong evidence of true difference     |
|                | Common threshold: `p < 0.05` = statistically significant |

If the **p-value is low**, you can **reject the null hypothesis** (that there's no difference) and conclude that one model performs significantly better than the other for the given metric.

---

## 6. Why This Matters

- Visualizations (e.g., boxplots) show general trends, but the t-test quantifies **how confident** we are that one model is better.
- This is especially important in bioinformatics and drug response modeling, where **performance differences can be subtle** but still meaningful.
- It helps **guide model selection** with a statistically grounded approach.

---

## 7. Optional: Extend to Other Metrics

Repeat the same process for:
- Other performance metrics (e.g., MAE, Spearman ρ)
- Different cross-validation settings
- Different feature sets (e.g., raw vs PCA vs embeddings)

Each comparison gives insights into whether model or feature changes are genuinely improving predictive power — not just appearing to do so by chance.
