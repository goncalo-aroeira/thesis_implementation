import numpy as np
import matplotlib.pyplot as plt
import umap

# Load embeddings
embeddings = np.loadtxt("gene_embeddings.csv", delimiter=",", skiprows=1, usecols=range(1, 513))

# Reduce dimensions with UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
embeddings_2d = reducer.fit_transform(embeddings)

# Plot and Save
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.6)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of Gene Embeddings")

# Save as image file
plt.savefig("gene_embeddings_umap.png", dpi=300)
print("UMAP visualization saved as gene_embeddings_umap.png")
