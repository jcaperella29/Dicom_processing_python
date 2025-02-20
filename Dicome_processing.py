import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO  # YOLOv8 for lesion detection
from PIL import Image

# Load pre-trained ResNet for tissue classification
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet_model.eval()

# Load YOLOv8 model for lesion detection
yolo_model = YOLO("yolov8n.pt")  # Replace with medical YOLO model if available

# Define image transformation pipeline
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
        outputs = resnet_model(image)
        _, predicted = outputs.max(1)

    return predicted.item()

def detect_lesions(image):
    """Run YOLO lesion detection and return bounding boxes"""
    image_rgb = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image  # Convert grayscale to RGB

    results = yolo_model(image_rgb)  # Run YOLO detection

    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = float(box.conf[0])  # Get confidence score
            class_id = int(box.cls[0])  # Get class ID
            boxes.append((x1, y1, x2, y2, confidence, class_id))

    return boxes

def display_dicom(image, metadata, prediction, boxes=None):
    """Display DICOM image with metadata, classification result, and lesion bounding boxes"""
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

    # Draw lesion bounding boxes
    if boxes:
        for (x1, y1, x2, y2, conf, cls) in boxes:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))
            ax.text(x1, y1 - 5, f"Lesion {cls}: {conf:.2f}", color='red', fontsize=12, bbox=text_props)

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

    # Classify tissue and detect lesions
    tissue_type = classify_tissue(image)
    boxes = detect_lesions(image)

    print(f"Predicted Tissue Type: {tissue_type}")
    print(f"Detected Lesions: {len(boxes)}")

    display_dicom(image, ds, tissue_type, boxes)

if __name__ == "__main__":
    main()
