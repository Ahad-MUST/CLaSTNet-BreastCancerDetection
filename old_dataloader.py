
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

# Define base path for images
base_dir = "D:/FYP/Code/BreakHis/BreakHis dataset/BreaKHis_v1/"  # Update with your path

# Load fold information
df = pd.read_csv("D:/FYP/Code/BreakHis/BreakHis dataset/Folds.csv")

# Extract labels from filenames (malignant or benign)
df["label"] = df["filename"].str.extract("(malignant|benign)")

# Add the base directory to filenames to create full paths
df["filename"] = base_dir + df["filename"]

# Remove duplicate file paths from the DataFrame
df = df.drop_duplicates(subset="filename").reset_index(drop=True)

# Define simpler transform (without augmentation)
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per pre-trained models
])

# Define a custom Dataset class for PyTorch
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index, 3]  # Assuming the first column has image paths
        label = self.dataframe.iloc[index, 4]  # Assuming the second column has labels


        # Ensure img_path is a valid string (image path)
        if isinstance(img_path, str) and img_path.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(img_path).convert("RGB")  # Open and convert image to RGB
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img = Image.new('RGB', (224, 224))  # Use a placeholder image in case of error
        else:
            raise ValueError(f"Invalid image path: {img_path}")

        if self.transform:
            img = self.transform(img)
        label = torch.tensor(0 if label == "benign" else 1)  # Convert 'benign' to 0 and 'malignant' to 1

        return img, label

# Perform initial train-test split (90% train+val, 10% test)
train_val_data, test_data = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)

# Further split the training+validation data into 80% training and 10% validation (from the 90% of original data)
train_data, val_data = train_test_split(train_val_data, test_size=0.1111, stratify=train_val_data["label"], random_state=42)

# Check shapes of train, validation, and test data
print(f"Train Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# Create Dataset objects
train_dataset = BreastCancerDataset(train_data, transform=simple_transform)
val_dataset = BreastCancerDataset(val_data, transform=simple_transform)
test_dataset = BreastCancerDataset(test_data, transform=simple_transform)