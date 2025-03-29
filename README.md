# CLaSTNet: Breast Cancer Detection

## Introduction
CLaSTNet is a deep learning model designed for breast cancer detection using histopathological images from the **BreaKHis dataset**. The model leverages CNN and Vision Transformers (ViTs) to classify benign and malignant tumors with high accuracy.

## Requirements
Ensure you have the following dependencies installed:

- torch==2.5.1+cu118  
- torchvision==0.20.1+cu118  
- pandas==2.2.3  
- Pillow==10.2.0  
- scikit-learn==1.6.1  
- numpy==1.26.3  
- matplotlib==3.10.0  
- seaborn==0.13.2  
- timm==1.0.13  
- tqdm==4.67.1  
- torchinfo==1.8.0  
- thop==0.1.1.post2209072238  


To install dependencies, run:
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 pandas==2.2.3 numpy==1.26.3 Pillow==10.2.0 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 timm==1.0.13 tqdm==4.67.1 torchinfo==1.8.0 thop==0.1.1.post2209072238
```

## Usage
### 1️ **Dataset Setup**
Download the BreaKHis dataset from the following link:
- **[BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)**

### 2️ **Update Dataset Path**
In `dataloader.py`, replace the dataset path with your actual dataset location:
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
- The model is trained on histopathological images using hybrid CLaST-Net architecture.
- Data augmentation techniques (horizontal/vertical flipping, rotation) are applied.
- The dataset is split into **training (70%)**, **validation (15%)**, and **test (15%)**.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **BreaKHis Dataset**: [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
- **PyTorch Framework** 

