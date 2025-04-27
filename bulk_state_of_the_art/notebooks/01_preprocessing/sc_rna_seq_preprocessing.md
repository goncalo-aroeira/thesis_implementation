# 🧬 Single-Cell RNA-seq: Quality Control Summary (Draft)

## 🔹 Step 1: Data Loading and Preparation
- Loaded a gene expression matrix in FPKM format from a Parquet file.
- Removed metadata rows and transposed the matrix to have samples (cell lines) as rows and genes as columns.
- Ensured all values were numeric (`float`) and filled any missing entries with zeros.

## 🔹 Step 2: Per-Cell Quality Control Metrics
- Calculated:
  - `total_counts`: sum of all gene expression values per cell line.
  - `n_genes`: number of genes with non-zero expression per cell line.
  - `log_total_counts` and `log_n_genes`: log-transformed versions for better distributional properties.

## 🔹 Step 3: Visual Inspection
- Plotted histograms of total counts and number of detected genes.
- Created a scatterplot of `total_counts` vs. `n_genes` to identify outlier patterns visually.

## 🔹 Step 4: Outlier Detection via MAD Thresholding
- Applied Median Absolute Deviation (MAD)-based thresholding with a tolerance of **5 MADs**.
- Marked samples as outliers if their metrics were outside `[median ± 5 * MAD]` for either log-transformed QC metric.
- Printed thresholds used (median, MAD, lower and upper bounds) for reproducibility.

## 🔹 Step 5: Filtering and Export
- Removed all samples flagged as outliers.
- Also filtered genes to keep only those expressed in at least **20 samples**.
- Saved the cleaned matrix as a new Parquet file for use in downstream analysis.

> 💡 Note: Mitochondrial gene content and ambient RNA correction were skipped because this dataset consists of bulk RNA-seq or preprocessed cell line data, where such artifacts are not applicable.
