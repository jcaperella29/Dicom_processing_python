import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_dicom(filepath):
    """Load a DICOM file and extract image & metadata"""
    ds = pydicom.dcmread(filepath, force=True)
    
    # Handle compressed images if needed
    transfer_syntax = ds.file_meta.TransferSyntaxUID if 'TransferSyntaxUID' in ds.file_meta else None
    if transfer_syntax and transfer_syntax.is_compressed:
        try:
            ds.decompress()
            print("Decompressed DICOM image successfully.")
        except Exception as e:
            print(f"Error decompressing DICOM: {e}")
            return None, None

    image = ds.pixel_array

    # If multi-frame, select the middle slice
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]

    return ds, image

def classify_tissue(image):
    """Use ResNet model to classify tissue type"""
    image = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image  # Convert grayscale to RGB
    image = Image.fromarray(image.astype(np.uint8))  # Convert NumPy array to PIL Image
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item()

def display_dicom(image, metadata, prediction):
    """Display DICOM image with metadata and classification result"""
    if image is None:
        print("No image to display.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    # Overlay text annotations
    text_props = dict(facecolor='black', alpha=0.5, edgecolor='white', boxstyle='round,pad=0.3')

    # Patient info
    patient_info = [
        f"Patient: {metadata.PatientName if hasattr(metadata, 'PatientName') else 'Unknown'}",
        f"Modality: {metadata.Modality if hasattr(metadata, 'Modality') else 'N/A'}",
        f"Study Date: {metadata.StudyDate if hasattr(metadata, 'StudyDate') else 'N/A'}"
    ]
    for i, line in enumerate(patient_info):
        ax.text(10, 20 + i * 20, line, fontsize=12, color='white', bbox=text_props)

    # Classification result
    ax.text(10, image.shape[0] - 20, f"Prediction: {prediction}", fontsize=14, color='yellow', bbox=text_props)

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

    # Classify tissue and display annotated image
    tissue_type = classify_tissue(image)
    print(f"Predicted Tissue Type: {tissue_type}")

    display_dicom(image, ds, tissue_type)

if __name__ == "__main__":
    main()
