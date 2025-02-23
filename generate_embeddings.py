import torch
from transformers import AutoModel

# Load pre-trained scGPT model
scgpt_model = AutoModel.from_pretrained("scgpt_pretrained_model_path")

# Function to generate embeddings
def generate_embeddings(expression_matrix):
    """
    Convert gene expression data into scGPT embeddings.
    """
    with torch.no_grad():
        embeddings = scgpt_model(expression_matrix)
    return embeddings

# Example Usage
cell_embeddings = generate_embeddings(expression_matrix)  # expression_matrix is preprocessed input
