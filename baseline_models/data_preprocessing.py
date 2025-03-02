import pandas as pd
import numpy as np

def load_gene_embeddings():
    """Loads the scGPT gene embeddings (precomputed)."""
    return pd.read_csv("pseudo_bulk_gene_embeddings.csv", index_col=0)

def load_ic50_data():
    """Loads and filters GDSC IC50 dataset."""
    gdsc_ic50 = pd.read_csv("/home/goncalo/scgpt/datasets/gdsc/gdsc_final_cleaned.csv")
    
    # Keep only drug-cell line pairs that exist in gene embeddings
    gene_embeddings = load_gene_embeddings()
    common_cells = list(set(gene_embeddings.index) & set(gdsc_ic50["SANGER_MODEL_ID"]))
    
    gdsc_ic50_filtered = gdsc_ic50[gdsc_ic50["SANGER_MODEL_ID"].isin(common_cells)]
    
    return gene_embeddings, gdsc_ic50_filtered

def prepare_data():
    """Prepares the dataset for training."""
    gene_embeddings, gdsc_ic50_filtered = load_ic50_data()
    
    # Expand gene embeddings for each drug-cell pair
    X = np.array([gene_embeddings.loc[cell_line].values for cell_line in gdsc_ic50_filtered["SANGER_MODEL_ID"]])
    y = gdsc_ic50_filtered["LN_IC50"].values
    
    return X, y

if __name__ == "__main__":
    X, y = prepare_data()
    print(f"✅ Data Prepared: X shape {X.shape}, y shape {y.shape}")
