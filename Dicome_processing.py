

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import convert_color_space
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load a pre-trained model for classification (ResNet as an example)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_dicom(filepath):
    """Load a DICOM file and extract image & metadata, handling compression"""
    ds = pydicom.dcmread(filepath, force=True)
    transfer_syntax = ds.file_meta.TransferSyntaxUID if 'TransferSyntaxUID' in ds.file_meta else None
    if transfer_syntax and transfer_syntax.is_compressed:
        print(f"Compressed image detected with syntax: {transfer_syntax}")
        try:
            ds.decompress()
            print("Decompressed DICOM image successfully.")
        except Exception as e:
            print(f"Error decompressing DICOM: {e}")
            return None, None
    image = ds.pixel_array
    if len(image.shape) == 3:
        print(f"Multi-frame image detected with {image.shape[0]} slices. Selecting middle slice.")
        image = image[image.shape[0] // 2]
    
    # Ensure patient name is correctly formatted
    if hasattr(ds, "PatientName") and "^" not in str(ds.PatientName):
        ds.PatientName = f"{ds.PatientName}^"  # Add caret to format correctly
    
    return ds, image

def classify_tissue(image):
    """Use a pre-trained deep learning model to classify tissue type."""
    image = np.stack([image]*3, axis=-1) if len(image.shape) == 2 else image  # Convert grayscale to RGB
    image = Image.fromarray(image.astype(np.uint8))  # Convert NumPy array to PIL Image, ensure dtype is uint8
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    return predicted.item()

def display_dicom(image, metadata):
    """Display the DICOM image with metadata"""
    if image is None:
        print("No image to display.")
        return
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(f"DICOM Image: {metadata.PatientName}")
    plt.axis('off')
    plt.show()

def main():
    """Main function to load and process a DICOM file"""
    filepath = r"C:\Users\ccape\Downloads\dicom_viewer_0002\0002.DCM"
    ds, image = load_dicom(filepath)
    if ds is None or image is None:
        print("Failed to load DICOM file.")
        return
    print("DICOM Metadata:")
    print(f"Patient Name: {ds.PatientName}")
    print(f"Modality: {ds.Modality}")
    print(f"Study Date: {ds.StudyDate}")
    print(f"Image Dimensions: {image.shape}")
    
    display_dicom(image, ds)
    tissue_type = classify_tissue(image)
    print(f"Predicted Tissue Type: {tissue_type}")

if __name__ == "__main__":
    main()

