from rdkit import Chem
import deepchem as dc
import pandas as pd

def extract_drug_features():
    """Extracts molecular fingerprints for drugs."""
    drug_smiles = pd.read_csv("gdsc_drug_smiles.csv")  # File with "DRUG_ID" and "SMILES"
    
    featurizer = dc.feat.CircularFingerprint(size=2048)
    drug_features = {drug: featurizer.featurize(Chem.MolFromSmiles(smiles))[0] for drug, smiles in zip(drug_smiles["DRUG_ID"], drug_smiles["SMILES"])}
    
    drug_feature_df = pd.DataFrame.from_dict(drug_features, orient="index")
    drug_feature_df.to_csv("drug_features.csv")
    
    print("✅ Drug features extracted and saved.")

if __name__ == "__main__":
    extract_drug_features()
