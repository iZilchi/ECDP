# Error-Corrected Differential Privacy for Federated Learning (EC-DP-FL)

This repository implements a privacy‑preserving federated learning framework with error‑corrected differential privacy for medical image classification (skin cancer HAM10000 and chest X‑ray datasets).

## Features
- Federated Averaging (FedAvg) baseline
- Differential Privacy (DP) with Gaussian mechanism
- Error‑corrected DP using extreme value clipping and adaptive gradient smoothing
- Comprehensive evaluation on HAM10000 (skin cancer) dataset
- Privacy‑utility tradeoff analysis across multiple ε budgets

## Requirements
- Python 3.11 or 3.12 (Python 3.14 is not recommended due to package compatibility)
- See `requirements.txt` for full list

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ECDP.git
cd ECDP

2. Create and activate a virtual environment
Windows (PowerShell):

powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux/macOS:

bash
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
Note: If you encounter issues with PyTorch, install it manually:

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

4. Download the HAM10000 dataset
The dataset is not included in the repository. You can download it using the Kaggle API:

bash
# Install Kaggle CLI
pip install kaggle

# Set up your Kaggle credentials (place kaggle.json in ~/.kaggle/ or set environment variables)
# Then download and unzip
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./data/skin_cancer --unzip
Alternatively, download manually from Kaggle and place the files in ./data/skin_cancer/.

5. Prepare the dataset for federated learning
Run the setup script to create train/test splits and an image mapping:

bash
python setup_skin_cancer.py
Verify the dataset is ready:

bash
python check_dataset.py
Expected output: ✅ DATASET READY!

6. Run experiments
Comprehensive comparison (Standard FL, DP‑FL, EC‑DP‑FL)
bash
python experiments/run_comprehensive_comparison.py
This will train all three methods for 5 rounds and save plots in the results/ folder.

Privacy‑utility tradeoff analysis
bash
python experiments/run_privacy_utility_tradeoff.py
Evaluates performance across ε ∈ {0.1, 0.2, 0.3, 0.5, 1.0, 2.0}.

Gradient norm analysis (for clipping calibration)
bash
python utils/analyze_gradients.py
Important Notes
Do not commit the dataset or virtual environment – they are excluded via .gitignore.

All results (plots, metrics) are saved in results/ and are also ignored by Git.

If you encounter ModuleNotFoundError for any package, install it manually with pip install <package> and consider adding it to requirements.txt.

Project Structure
text
ECDP/
├── core/                     # DP and FL implementations
├── experiments/              # Experiment scripts
├── models/                   # CNN model definitions
├── utils/                    # Data loading, metrics, helpers
├── data/                     # Dataset folder (ignored)
├── results/                  # Output plots (ignored)
├── .gitignore
├── requirements.txt
├── setup_skin_cancer.py
├── check_dataset.py
└── README.md