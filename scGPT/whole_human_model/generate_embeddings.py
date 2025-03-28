import torch
import json
from scgpt.model.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import numpy as np
import pandas as pd

# ====== Step 1: Load Configuration & Fix Missing Arguments ====== #

# Load arguments from args.json
with open("args.json", "r") as f:
    args = json.load(f)

# Load vocabulary and determine ntoken (total number of tokens)
vocab_path = "vocab.json"
gene_vocab = GeneVocab.from_file(vocab_path)  # Load vocab object
ntoken = len(gene_vocab)  # Total number of genes

# Manually set required missing values
fixed_args = {
    "ntoken": ntoken,
    "d_model": args.get("embsize", 512),  # Default to 512 if missing
    "nhead": args.get("nheads", 8),       # Default to 8 if missing
    "d_hid": args.get("d_hid", 512),
    "nlayers": args.get("nlayers", 12),
    "dropout": args.get("dropout", 0.2),
    "pad_token": args.get("pad_token", "<pad>"),
    "pad_value": args.get("pad_value", -2),
    "vocab": gene_vocab,  # Pass the loaded vocab object
}

# Initialize model with corrected args
model = TransformerModel(**fixed_args)  
checkpoint = torch.load("best_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint, strict=False)  # Allow partial loading if needed
model.eval()  # Set model to evaluation mode

# ====== Step 2: Tokenize All Genes ====== #

# Extract all gene names from vocab.json
all_genes = list(gene_vocab.get_stoi().keys())  # Get all gene names

# Convert gene names to token IDs
tokens = [gene_vocab[gene] if gene in gene_vocab else gene_vocab["<pad>"] for gene in all_genes]
tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

print(f"Total Genes Tokenized: {len(tokens)}")

# ====== Step 3: Generate Embeddings for All Genes ====== #
with torch.no_grad():
    embeddings = model.encoder(tokens_tensor)  # Extract raw embeddings from GeneEncoder

print("Generated Embeddings Shape:", embeddings.shape)  # Expected: (1, ntoken, d_model)

# ====== Step 4: Save Embeddings to File ====== #

# Convert embeddings to a NumPy array
embeddings_array = embeddings.squeeze(0).numpy()

# Save as Parquet
df = pd.DataFrame(embeddings_array, index=all_genes)
df.index.name = "gene"
df.to_parquet("gene_embeddings.parquet", index=True)

print("Embeddings saved to gene_embeddings.parquet!")