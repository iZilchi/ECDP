# Error-Corrected Differential Privacy for Federated Learning (EC-DP-FL)

This repository implements a **privacy‑preserving federated learning framework** with **error‑corrected differential privacy** for medical image classification. It supports four datasets:

- **HAM10000** – skin cancer classification (7 classes)
- **Chest X-ray (Pneumonia)** – binary pneumonia detection
- **Uganda TB Chest X-ray** – binary tuberculosis detection
- **BreastMNIST** – breast ultrasound (benign/malignant)

The framework includes:
- Standard Federated Averaging (FedAvg)
- Basic Differential Privacy (DP) with Gaussian mechanism and gradient clipping
- **EC-DP-FL (proposed)** – post‑aggregation error correction using **Extreme Value Clipping (EVC)** and **Adaptive Gradient Smoothing (AGS)**

## Features

- Simulated federated learning with configurable **IID** or **non‑IID** data partitioning (Dirichlet distribution)
- Privacy budget control via **per‑round ε** or **total target ε** with Rényi Differential Privacy (RDP) accounting
- Ablation studies to isolate EVC and AGS contributions
- Privacy‑utility tradeoff analysis across multiple ε values
- Statistical validation with multiple independent runs and paired t‑tests
- Gradient norm analysis to help choose clipping norms

## Requirements

- Python **3.11** or **3.12** (Python 3.14 not recommended due to package compatibility)
- See `requirements.txt` for full list

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ECDP.git
cd ECDP
```

### 2. Create and activate a virtual environment
Windows (PowerShell):
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter PyTorch issues, install the CPU version manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
### 4. Prepare the datasets
All datasets will be stored under ./data/. Use the commands below to download, unzip, and prepare each dataset.

### 4.1 HAM10000 (Skin Cancer)
```bash
# Install Kaggle CLI if not already installed
pip install kaggle
# Set up Kaggle credentials (place kaggle.json in ~/.kaggle/ or set environment variables)
# Then download and unzip the dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./data/skin_cancer --unzip
```
After download, run the setup script to create train/test splits and the image mapping:

```bash
python setup_skin_cancer.py
```
Verify the dataset is ready:

```bash
python check_dataset.py
# Expected output: ✅ DATASET READY!
```

### 4.2 Chest X‑ray (Pneumonia)
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data/chest_xray --unzip
```
The archive will extract to ./data/chest_xray/chest_xray/ (nested). The data loaders automatically find the correct subfolder. Verify with:

```bash
python check_chest_xray_dataset.py
# Expected output: ✅ DATASET READY!
```

### 4.3 Uganda TB Chest X‑ray
```bash
kaggle datasets download -d apolotdenise/chest-xrays-for-tb -p ./data/chest_xray_tb --unzip
```
This places TB‑positive images in ./data/chest_xray_tb/. The normal (non‑TB) images are reused from the pneumonia dataset under ./data/chest_xray/chest_xray/train/NORMAL/. No further extraction is required.

### 4.4 BreastMNIST (Breast Ultrasound)
BreastMNIST is downloaded automatically via the medmnist package. Run the following one‑liner to download and extract the dataset into ./data/breastmnist/extracted/:

```bash
pip install medmnist
python -c "import numpy as np; from medmnist import BreastMNIST; import os; os.makedirs('./data/breastmnist/extracted', exist_ok=True); [np.save(f'./data/breastmnist/extracted/{split}_{key}.npy', np.array([dataset[i][0] if key=='images' else dataset[i][1][0] for i in range(len(dataset))])) for split in ['train','val','test'] for key,dataset in [(key, BreastMNIST(split=split, download=False, root='./data/breastmnist')) for key in ['images','labels']]]; print('Extracted!')"
```
Note: The first run will automatically download the .npz files from MedMNIST (approx 20 MB). The command converts them to separate .npy files for faster loading.

After completing the steps above, all datasets are ready for federated learning experiments.

### Running Experiments
All experiments are controlled by experiments/run_experiments.py. Use the --mode argument to select the type of analysis.

#### Example Commands
### 1. Standard comparison (Standard FL vs Basic DP‑FL vs EC‑DP‑FL)
```bash
python experiments/run_experiments.py --mode comparison --dataset skin --per_round_epsilon 2.0 --rounds 20 --clients 5 --clip_norm 2.4 --c 2.5 --alpha 0.8
```
Plots convergence curves and saves results in results/.

### 2. Privacy‑utility tradeoff
```bash
python experiments/run_experiments.py --mode tradeoff --dataset chest --rounds 20 --clients 3 --clip_norm 6.1
```
Evaluates Basic DP‑FL and EC‑DP‑FL across ε ∈ {0.1, 1.0, 2.0} and saves a tradeoff plot.

### 3. Statistical validation (multiple runs with t‑test)
```bash
python experiments/run_experiments.py --mode validation --dataset tb_uganda --target_epsilon 2.0 --rounds 20 --trials 10 --clients 10 --clip_norm 2.5
```
Runs 10 independent trials and reports mean ± std, win rate, and paired t‑test comparing EC‑DP‑FL vs Basic DP‑FL.

### 4. Ablation study (isolate EVC and AGS)
```bash
python experiments/run_experiments.py --mode ablation --dataset skin --per_round_epsilon 2.0 --rounds 20 --clients 5 --clip_norm 2.4
```
Compares: Standard FL, DP only, DP+EVC, DP+AGS. Results saved to results/ablation_results.txt.

### 5. Gradient norm analysis (choose clip_norm)
```bash
python utils/analyze_gradients.py --dataset skin --clients 10
```
Prints update norm statistics and recommends a clip_norm value (e.g., 75th percentile).

### Project Structure
ECDP/
├── core/                     # FL, DP, and error correction implementations
│   ├── federated_learning.py
│   ├── differential_privacy.py
│   ├── dpfl.py
│   └── error_correction.py
├── experiments/              # Main experiment script
│   └── run_experiments.py
├── models/                   # CNN model definitions
│   ├── medium_cnn.py        # HAM10000
│   ├── chest_cnn.py         # Pneumonia
│   └── tb_cnn.py            # Uganda TB
├── utils/                    # Data loaders, metrics, helpers
│   ├── data_loader.py
│   ├── chest_xray_loader.py
│   ├── tb_uganda_loader.py
│   ├── metrics.py
│   ├── privacy_accountant.py
│   ├── analyze_gradients.py
│   └── ...
├── data/                     # Dataset folders (ignored by Git)
├── results/                  # Output plots and logs (ignored)
├── requirements.txt
├── README.md
├── setup_skin_cancer.py
├── check_dataset.py
├── check_chest_xray_dataset.py
└── prepare_*.py              # Other preparation scripts

### Important Notes
Do not commit the dataset folders or the virtual environment – they are excluded via .gitignore.

All results (plots, metrics) are saved in results/ and are also ignored by Git.

If you encounter ModuleNotFoundError for a missing package, install it manually (pip install <package>) and consider updating requirements.txt.

For reproducibility, a fixed random seed (default 42) is used throughout.

### Citation
If you use this code in your research, please cite the original thesis:

Landicho, A. M. M., Lontok, M. J. M., & Pagcaliuangan, K. M. D. (2026). A Privacy-Preserving Federated Learning Using Error-Corrected Differential Privacy. Bachelor's thesis, Batangas State University.

### License
This project is provided for academic and research purposes.
