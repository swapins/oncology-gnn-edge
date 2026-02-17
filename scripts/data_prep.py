import numpy as np

def preprocess_transcriptomics(data):
    """
    Log-transform and standardize gene expression per paper methodology.
    """
    # log-transformed and standardized on a per-gene basis
    log_data = np.log1p(data)
    standardized = (log_data - np.mean(log_data, axis=0)) / np.std(log_data, axis=0)
    return standardized

print("Preprocessing script aligned with TCGA-BRCA/LUAD methodology.")