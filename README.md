# Syntactic Role Representation in Multilingual BERT

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

## Data

Stored in `data/` (UD treebanks):
- English: en_ewt-ud-{train,dev,test}.conllu
- Hindi: hi_hdtb-ud-{train,dev,test}.conllu
- Turkish: tr_imst-ud-{train,dev,test}.conllu

Outputs in `outputs/`:
- `*_embeddings.npz` (13 layers × token count × 768 dims)
- `*_meta.csv` (token metadata)
- `*_balanced.csv` (balanced datasets)

## Results

Peak probe accuracy by language:
- English: Layer 8
- Hindi: Layer 8
- Turkish: Layer 12

See `outputs/` for detailed metrics (accuracy, cosine distance by layer).
