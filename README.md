# Syntactic Role Representation in mBERT

Analysis of how noun embeddings in mBERT distinguish syntactic roles (subject vs. object) across English, Hindi, and Turkish.

## Pipeline

1. **Data Preparation** (`01_data_prep.ipynb`)
   - Parse UD treebanks for English, Hindi, Turkish
   - Extract noun tokens with subject/object labels
   - Balance dataset across roles and languages

2. **Embedding Extraction** (`02_embeddings.ipynb`)
   - Extract mBERT token embeddings from all 13 layers
   - Mean-pool sub-token representations
   - Save as NumPy arrays (compressed format)

3. **Probing & Analysis** (`03_probing.ipynb`)
   - Logistic regression probes (5-fold CV per layer)
   - Cosine distance analysis between role centroids
   - PCA visualization and statistical significance tests


## Results

See `figures/` for output results
