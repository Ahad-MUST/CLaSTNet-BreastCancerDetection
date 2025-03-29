# CLaSTNet: Breast Cancer Detection

## Introduction
CLaSTNet is a deep learning model designed for breast cancer detection using histopathological images from the **BreaKHis dataset**. The model leverages CNN and Vision Transformers (ViTs) to classify benign and malignant tumors with high accuracy.

## Requirements
Ensure you have the following dependencies installed:

- Python 3.10.11
- PyTorch == X.X.X  
- torchvision == X.X.X  
- pandas == X.X.X  
- numpy == X.X.X  
- Pillow == X.X.X  
- scikit-learn == X.X.X  

To install dependencies, run:
```bash
pip install torch torchvision pandas numpy Pillow scikit-learn
```

## Usage
### 1️ **Dataset Setup**
Download the BreaKHis dataset from the following link:
- **[BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)**

### 2️ **Update Dataset Path**
In `train.py`, replace the dataset path with your actual dataset location:
```python
base_dir = "<YOUR_DATASET_PATH>"
df = pd.read_csv("<YOUR_CSV_FILE_PATH>")
```

### 3️ **Run Training**
To start training, execute:
```bash
python train.py
```

## Training
- The model is trained on histopathological images using CNN-ViT architecture.
- Data augmentation techniques (horizontal/vertical flipping, rotation) are applied.
- The dataset is split into **training (70%)**, **validation (15%)**, and **test (15%)**.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **BreaKHis Dataset**: [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
- **PyTorch Framework** 

