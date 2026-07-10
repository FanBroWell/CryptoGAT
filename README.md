<div align="center">

# CryptoGAT

### Are Time Series Models Effective for Cryptocurrency Forecasting?

[![arXiv](https://img.shields.io/badge/arXiv-2606.27670-b31b1b.svg)](https://arxiv.org/abs/2606.27670)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2606.27670)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E.svg)](https://huggingface.co/datasets/CharlieYPeng/cryptogat-crypto-1d)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)](https://arxiv.org/abs/2606.27670)
[![GitHub](https://img.shields.io/badge/Code-GitHub-blue.svg)](https://github.com/FanBroWell/CryptoGAT)

**Official implementation of CryptoGAT, a lightweight Graph Attention Network for cryptocurrency forecasting.**

</div>

## Overview

We introduce a set of simple yet effective models named CryptoGAT as a new baseline for comparison. CryptoGAT uses an attention mechanism architecture to process graph-structured data, allowing each node to focus on its neighbors to compute hidden representations and assign different weights. To handle cryptocurrency markets with different characteristics, we further introduce a variant, named FGAT. Now we have a CryptoGAT family:

- **GAT:** A simple yet effective graph-based model via a graph attention mechanism.
- **FGAT:** FeatureMixer Graph Attention Network (FGAT) combines the Indicator mixing mechanism with the graph attention architecture.

## Dataset

The CryptoGAT Crypto 1D dataset used in this project is available on the Hugging Face Hub:

**[CharlieYPeng/cryptogat-crypto-1d](https://huggingface.co/datasets/CharlieYPeng/cryptogat-crypto-1d)**

## Paper

Our paper is available on arXiv:

**CryptoGAT: Are Time Series Models Effective for Cryptocurrency Forecasting?**<br>
Yu Peng, Matloob Khushi, Josiah Poon<br>
arXiv:2606.27670, 2026<br>
[Paper](https://arxiv.org/abs/2606.27670) | [PDF](https://arxiv.org/pdf/2606.27670)

## Citation

If you find this repository or paper useful, please cite our work.

### BibTeX

```bibtex
@misc{peng2026cryptogat,
  title = {{CryptoGAT}: Are Time Series Models Effective for Cryptocurrency Forecasting?},
  author = {Peng, Yu and Khushi, Matloob and Poon, Josiah},
  year = {2026},
  eprint = {2606.27670},
  archivePrefix = {arXiv},
  primaryClass = {cs.CE},
  doi = {10.48550/arXiv.2606.27670},
  url = {https://arxiv.org/abs/2606.27670}
}
```

### APA

```text
Peng, Y., Khushi, M., & Poon, J. (2026). CryptoGAT: Are Time Series Models Effective for Cryptocurrency Forecasting? arXiv. https://doi.org/10.48550/arXiv.2606.27670
```

### IEEE

```text
Y. Peng, M. Khushi, and J. Poon, "CryptoGAT: Are Time Series Models Effective for Cryptocurrency Forecasting?," arXiv:2606.27670, 2026. doi: 10.48550/arXiv.2606.27670.
```

### Plain Text

```text
Yu Peng, Matloob Khushi, and Josiah Poon. CryptoGAT: Are Time Series Models Effective for Cryptocurrency Forecasting? arXiv:2606.27670, 2026. https://arxiv.org/abs/2606.27670
```

## Getting Started

### Environment Requirements

- tqdm==4.64.1
- pandas==2.0.3
- pyyaml==6.0
- numpy==1.22.1
- matplotlib==3.5.1
- torch>=1.9.0

### Dataset and Preprocessing

In order to improve file reading speed, we process the raw data to generate corresponding .pkl or .npy files. Datasets are provided in the dataset folder.

#### CRYPTO_1D_ALL

```bash
# Process the data into CRYPTO_1D_ALL
cd src
python process_crypto_ALL.py
```

#### CRYPTO_1D_ENHANCED

```bash
# Process the data into CRYPTO_1D_ENHANCED for FGAT
cd src_Feature
python process_crypto_enhanced.py
```

### Running the Code

#### Train GAT Model

```bash
cd src
python train_gat_crypto.py
```

#### Train FGAT Model

```bash
cd src
python train_gat_enhanced.py
```
## Evaluation Metrics

- **MSE:** Mean Squared Error
- **IC:** Information Coefficient
- **ICIR:** Information Coefficient Information Ratio
- **Prec@10:** Precision at Top 10
- **Sharpe:** Sharpe Ratio
