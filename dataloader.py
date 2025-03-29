import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataset paths
base_dir = "path_to_your_dataset/BreaKHis_v1/"
df = pd.read_csv("path_to_your_dataset/Folds.csv")

# Extract label (benign/malignant) from the filename
df['label'] = df['filename'].str.extract("(malignant|benign)")
df['filename'] = base_dir + df['filename']

# Remove duplicate entries and reset index
df = df.drop_duplicates(subset="filename").reset_index(drop=True)
print('Loading data...')

# Function to load an image from file
def load_image(filepath):
    try:
        img = Image.open(filepath).convert('RGB')  # Convert to RGB format
        return img
    except Exception as e:
        print(f"Error loading image: {filepath}, {e}")
        return None

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Define augmentation transformations
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flip image horizontally
    transforms.RandomVerticalFlip(),  # Flip image vertically
    transforms.RandomRotation(15)  # Rotate image randomly
])

# Function to apply augmentation
def augment_image(image):
    return augmentation(image)

# Load images from file paths
df['image'] = df['filename'].apply(load_image)
print('Applying augmentation...')

# Apply data augmentation
df['augmented'] = df['image'].apply(lambda img: augment_image(img) if img is not None else None)

# Convert images to tensors (handling None values)
df['image'] = df['image'].apply(lambda img: transform(img) if img is not None else None)
df['augmented'] = df['augmented'].apply(lambda img: transform(img) if img is not None else None)

# Remove any rows where image loading failed
df = df.dropna().reset_index(drop=True)

# Create two datasets: original and augmented
original_df = df[['image', 'label']]
augmented_df = df[['augmented', 'label']].rename(columns={'augmented': 'image'})

# Combine both datasets and shuffle
combined_df = pd.concat([original_df, augmented_df])
combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset

# Split dataset into train, validation, and test sets
train_val_data, test_data = train_test_split(combined_df, test_size=0.15, stratify=combined_df['label'])
train_data, val_data = train_test_split(train_val_data, test_size=0.1765, stratify=train_val_data['label'])

# Custom PyTorch dataset class
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image = self.dataframe.iloc[index, 0]  # Get image tensor
        label = self.dataframe.iloc[index, 1]  # Get label
        label = 0 if label == 'benign' else 1  # Convert labels to numeric values
        return image, torch.tensor(label, dtype=torch.long)

# Create dataset objects
train_dataset = BreastCancerDataset(train_data)
val_dataset = BreastCancerDataset(val_data)
test_dataset = BreastCancerDataset(test_data)

# Count and print the number of samples for each class
train_benign = train_data[train_data['label'] == 'benign'].shape[0]
train_malignant = train_data[train_data['label'] == 'malignant'].shape[0]
val_benign = val_data[val_data['label'] == 'benign'].shape[0]
val_malignant = val_data[val_data['label'] == 'malignant'].shape[0]
test_benign = test_data[test_data['label'] == 'benign'].shape[0]
test_malignant = test_data[test_data['label'] == 'malignant'].shape[0]

print("Training Data:")
print(f"Benign: {train_benign}, Malignant: {train_malignant}")
print("\nValidation Data:")
print(f"Benign: {val_benign}, Malignant: {val_malignant}")
print("\nTest Data:")
print(f"Benign: {test_benign}, Malignant: {test_malignant}")
