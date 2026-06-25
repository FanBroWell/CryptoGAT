# CryptoGAT
We introduce a set of simple yet effective models named CryptoGAT as a new baseline for comparison. CryptoGAT uses an attention mechanism architecture to process graph-structured data, allowing each node to focus on its neighbors to compute hidden representations and assign different weights. To handle cryptocurrency markets with different characteristics, we further introduce a variant, named FGAT. Now we have a CryptoGAT family:

- **GAT:** A simple yet effective graph-based model via a graph attention mechanism.
- **FGAT:** FeatureMixer Graph Attention Network (FGAT) combines the Indicator mixing mechanism with the graph attention architecture.

# Getting Started

## Environment Requirements

- tqdm==4.64.1
- pandas==2.0.3
- pyyaml==6.0
- numpy==1.22.1
- matplotlib==3.5.1
- torch>=1.9.0

## Dataset and Preprocessing

In order to improve file reading speed, we process the raw data to generate corresponding .pkl or .npy files. Datasets are provided in the dataset folder.

### CRYPTO_1D_ALL

```bash
# Process the data into CRYPTO_1D_ALL
cd src
python process_crypto_ALL.py
```

### CRYPTO_1D_ENHANCED

```bash
# Process the data into CRYPTO_1D_ENHANCED for FGAT
cd src_Feature
python process_crypto_enhanced.py
```

## Running the Code

### Train GAT Model

```bash
cd src
python train_gat_crypto.py
```

### Train FGAT Model

```bash
cd src
python train_gat_enhanced.py
```
## Evaluation metrics

- **MSE:** Mean Squared Error
- **IC:** Information Coefficient
- **ICIR:** Information Coefficient Information Ratio
- **Prec@10:** Precision at Top 10
- **Sharpe:** Sharpe Ratio
