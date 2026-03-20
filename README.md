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

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ECDP.git
cd ECDP

### 2. Create and activate a virtual environment
Windows (PowerShell):

powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

Linux/macOS:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

PyTorch Installation
Option 1: CPU‑only (no GPU)
pip install torch torchvision torchaudio

Option 2: With GPU (CUDA)
- Check your CUDA driver version:
nvidia-smi
Look at the top‑right corner for the CUDA Version (e.g., 11.8, 12.4, 12.6).
If nvidia-smi is not found, install the latest NVIDIA driver from nvidia.com.

-Install PyTorch with the corresponding CUDA version. For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
Replace cu124 with your CUDA version (cu118 for 11.8, cu126 for 12.6, etc.).

-Verify the installation:
python -c "import torch; print(torch.cuda.is_available())"

Note: If you install the GPU version but later use --device cuda without a compatible GPU, the code will automatically fall back to CPU with a warning.

4. Download the HAM10000 dataset
The dataset is not included in the repository. You can download it using the Kaggle API:

bash
pip install kaggle

kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p ./data/skin_cancer
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data/chest_xray

Unzip:
cd data/skin_cancer
Expand-Archive -Path skin-cancer-mnist-ham10000.zip -DestinationPath .
Remove-Item skin-cancer-mnist-ham10000.zip   # optional cleanup
cd ../..

cd data/chest_xray
Expand-Archive -Path chest-xray-pneumonia.zip -DestinationPath .
Remove-Item chest-xray-pneumonia.zip   # optional cleanup
cd ../..

For linux:
cd data/skin_cancer
unzip skin-cancer-mnist-ham10000.zip
rm skin-cancer-mnist-ham10000.zip
cd ../..

cd data/chest_xray
unzip chest-xray-pneumonia.zip
rm chest-xray-pneumonia.zip
cd ../..


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
