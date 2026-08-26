# 4i-SRAE-for-multimode-process-monitoring
## About This Repository

This repository contains the official implementation of 4i‑SRAE (Four‑Input Siamese Recurrent Autoencoder), a deep‑learning‑based fault detection and diagnosis model designed for multimode industrial process monitoring. The proposed model adopts a four‑input Siamese recurrent autoencoder architecture together with three customized normalized loss functions, which achieves effective fault monitoring for both benchmark simulation dataset (Tennessee‑Eastman, TE) and real‑world industrial catalytic reforming process dataset.

## Project Structure
4i‑SRAE‑multimode‑process‑monitoring/
├── model.py               # Definition of 4i‑SRAE network and three normalized loss functions
├── preprocessing.py       # Data preprocessing pipeline: random seed fix, RobustScaler, dataset split, sliding‑window construction, four‑input sample generation, latent feature extraction
├── monitoring.py          # Monitoring metrics calculation: Mahalanobis statistic, KDE‑derived control limit, FAR, FDR, fault detection and diagnosis (FDD) evaluation
├── experiment.py          # Shared workflow: model training, control‑limit estimation, test‑set evaluation, results saving for two case studies
├── data_te.py             # Load & organize Tennessee Eastman dataset (training / multimode / unseen‑mode / fault test set)
├── data_industrial.py      # Load & organize industrial catalytic reforming dataset (training / multimode / unseen‑mode / normal / fault test set)
├── run_TE.py              # Configuration & entry script for TE benchmark case study
├── run_industrial.py      # Configuration & entry script for industrial catalytic reforming case study
└── data/
    ├── TE/                # TE simulation dataset: normal, multimode, unseen‑mode and fault‑related data
    └── industrial/        # Industrial DCS dataset and variable metadata. Industrial raw data is confidential and available upon reasonable request to the corresponding author.

## File Descriptions

- **`model.py`**: Implements the proposed four‑input Siamese recurrent autoencoder (4i‑SRAE), including network layers and three customized normalized loss functions for model optimization.
- **`preprocessing.py`**: Provides full preprocessing utilities: reproducible random‑seed setup, `RobustScaler` normalization, train‑test splitting, sliding‑window segmentation, four‑input sample construction, and latent feature extraction from trained model.
- **`monitoring.py`**: Calculates process monitoring statistics, including Mahalanobis distance statistics, KDE‑based adaptive control limits. It also computes standard evaluation metrics: False Alarm Rate (FAR), Fault Detection Rate (FDR), and completes fault detection & diagnosis (FDD) assessment.
- **`experiment.py`**: Encapsulates reusable shared experimental workflows for both case studies, including model training procedure, control‑limit estimation, test‑dataset evaluation, and persistent saving of experimental outputs.
- **`data_te.py`**: Data‑loading module for the Tennessee Eastman (TE) benchmark; organizes training set, multimode test set, unseen‑mode test set and various fault test datasets.
- **`data_industrial.py`**: Data‑loading module for real‑world catalytic reforming industrial process; handles training data, multimode data, unseen‑mode data, normal‑condition and fault test datasets.
- **`run_TE.py`**: Main entry script for the TE benchmark experiment. Set hyper‑parameters and execute the 4i‑SRAE monitoring pipeline for TE dataset.
- **`run_industrial.py`**: Main entry script for industrial catalytic reforming experiment. Configure hyper‑parameters and run 4i‑SRAE on real‑world industrial dataset.
- **`data/TE/`**: Stores TE benchmark datasets for simulation case study, covering normal operating condition, multimode, unseen‑mode and multiple fault scenarios.
- **`data/industrial/`**: Contains industrial DCS collected dataset and variable‑description metadata. **The raw industrial process data is confidential. Access can be obtained upon reasonable request sent to the corresponding author.**
