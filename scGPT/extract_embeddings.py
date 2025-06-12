import pandas as pd
import scanpy as sc
import numpy as np
from scgpt.tasks.cell_emb import embed_data


'''
# Run embedding extraction
adata = embed_data(
    adata_or_file="data/pancancer_subset_200.h5ad",  # original file
    model_dir=Path("examples/save/"),
    gene_col="index",  # 👈 use index as gene names
    max_length=1200,
    batch_size=64,
    device="cuda",  # or "cpu"
)



# Save the resulting AnnData with embeddings
adata.write("bulk_data_with_embeddings.h5ad")
'''

# Load bulk CSV
df = pd.read_csv("data/processed_bulk.csv", index_col=0)  # SANGER_MODEL_ID as index

# Create AnnData
adata = sc.AnnData(X=df.values.astype(np.float32))
adata.obs_names = df.index
adata.var_names = df.columns
adata.var["feature_name"] = adata.var_names  # Required by embed_data()

# Save to .h5ad
adata.write("bulk_data.h5ad")

embedded_adata = embed_data(
    "bulk_data.h5ad",
    model_dir="examples/save/",
    gene_col="feature_name",
    max_length= 10000,
    return_new_adata=True
)
embedded_adata.write("bulk_embeddings.h5ad")