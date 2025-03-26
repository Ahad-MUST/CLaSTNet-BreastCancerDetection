import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
import seaborn as sns
from torchinfo import summary
from thop import profile, clever_format

# Disable unnecessary warnings and logs
import logging
import warnings
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self, model, test_loader, input_size, device, class_names=["Benign", "Malignant"]):
        """
        Initializes the Evaluator class.
        
        Args:
            model (torch.nn.Module): The trained model.
            test_loader (DataLoader): DataLoader for the test dataset.
            input_size (tuple): Shape of the input tensor.
            device (str): Device to run evaluation (CPU/GPU).
            class_names (list): Names of the classes for classification.
        """
        self.model = model
        self.test_loader = test_loader
        self.input_size = input_size
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_names = class_names
    
    def show_predictions(self, num_samples=25):
        """Displays sample predictions from the test dataset."""
        self.model.eval()
        images, labels = next(iter(self.test_loader))  
        images, labels = images[:num_samples], labels[:num_samples]  
        
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)  

        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()

        fig, axes = plt.subplots(1, num_samples, figsize=(20, 5), dpi=300)
        for i in range(num_samples):
            ax = axes[i]
            img = images[i].transpose((1, 2, 0))  
            img = np.squeeze(img)  

            ax.imshow(img, cmap="gray")  
            ax.set_xlabel(f"Real: {self.class_names[labels[i]]}\nPred: {self.class_names[predicted[i]]}", 
                          fontsize=14, fontweight='bold', color="green" if labels[i] == predicted[i] else "red")
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()
    
    def evaluate(self):
        """Evaluates the model on the test dataset and generates performance metrics."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))

        self.plot_confusion_matrix(all_labels, all_preds)
        self.plot_roc_curve(all_labels, all_probs)

    def plot_confusion_matrix(self, labels, preds, save_path="confusion_matrix.png"):
        """Plots and saves the confusion matrix."""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 6), dpi=300)
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=self.class_names, yticklabels=self.class_names,
                    annot_kws={"size": 14, "weight": "bold"})
        plt.xlabel("Predicted", fontsize=16, fontweight="bold")
        plt.ylabel("Actual", fontsize=16, fontweight="bold")
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, labels, probs, save_path="roc_curve.png"):
        """Plots and saves the ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')

        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_model(self):
        """Analyzes the model architecture and computational complexity."""
        print("\nModel Summary:")
        print(summary(self.model, input_size=self.input_size, device=self.device))

        dummy_input = torch.randn(*self.input_size).to(self.device)
        macs, params = profile(self.model, inputs=(dummy_input,))
        macs, params = clever_format([macs, params], "%.3f")

        print("\nModel Analysis:")
        print(f"FLOPs: {macs}")
        print(f"Parameters: {params}")
