import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50

# Load DeepLesion (Faster R-CNN with ResNet50) for lesion detection
deep_lesion_model = fasterrcnn_resnet50_fpn(pretrained=True)
deep_lesion_model.eval()

# Load ResNet50 classifier (pretrained on medical images)
lesion_classifier = resnet50(pretrained=True)
lesion_classifier.fc = torch.nn.Linear(2048, 4)  # Assuming 4 lesion types (Tumor, Cyst, Hemorrhage, Inflammation)
lesion_classifier.eval()

# Define image transformation pipeline
transform_detect = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

transform_classify = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lesion classes (Modify based on actual dataset)
LESION_CLASSES = ["Tumor", "Cyst", "Hemorrhage", "Inflammation"]

def load_dicom(filepath):
    """Load a DICOM file and extract image & metadata"""
    ds = pydicom.dcmread(filepath, force=True)
    image = ds.pixel_array

    # If multi-frame, select the middle slice
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]

    return ds, image

def detect_lesions(image):
    """Run DeepLesion model (Faster R-CNN) and return bounding boxes"""
    image_rgb = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image  # Convert grayscale to RGB
    image_tensor = transform_detect(image_rgb).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        detections = deep_lesion_model(image_tensor)

    # Extract bounding boxes, labels, and confidence scores
    boxes = []
    for box, score in zip(detections[0]['boxes'], detections[0]['scores']):
        if score > 0.75:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append((x1, y1, x2, y2, score.item()))

    return boxes

def classify_lesion(image, box):
    """Classify a detected lesion using ResNet50"""
    x1, y1, x2, y2 = box
    lesion_crop = image[y1:y2, x1:x2]  # Crop detected lesion region

    if lesion_crop.size == 0:  # Avoid empty crops
        return "Unknown"

    lesion_crop = Image.fromarray(lesion_crop.astype(np.uint8))  # Convert to PIL image
    lesion_tensor = transform_classify(lesion_crop).unsqueeze(0)  # Transform for ResNet

    with torch.no_grad():
        output = lesion_classifier(lesion_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return LESION_CLASSES[predicted_class]  # Return lesion type

def display_dicom(image, metadata, boxes=None):
    """Display DICOM image with lesion detection bounding boxes & classifications"""
    if image is None:
        print("No image to display.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    # Draw lesion bounding boxes and classify lesions
    if boxes:
        for (x1, y1, x2, y2, conf) in boxes:
            lesion_type = classify_lesion(image, (x1, y1, x2, y2))

            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))
            ax.text(x1, y1 - 5, f"{lesion_type} ({conf:.2f})", color='red', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    plt.show()

def main():
    """Main function to load and process a DICOM file"""
    filepath = r"C:\Users\ccape\Downloads\dicom_viewer_0002\0002.DCM"
    ds, image = load_dicom(filepath)
    
    if ds is None or image is None:
        print("Failed to load DICOM file.")
        return

    print(f"Patient Name: {ds.PatientName if hasattr(ds, 'PatientName') else 'Unknown'}")
    print(f"Modality: {ds.Modality if hasattr(ds, 'Modality') else 'N/A'}")
    print(f"Study Date: {ds.StudyDate if hasattr(ds, 'StudyDate') else 'N/A'}")

    # Detect lesions
    boxes = detect_lesions(image)

    print(f"Detected Lesions: {len(boxes)}")
    for b in boxes:
        lesion_type = classify_lesion(image, b)
        print(f"Lesion Type: {lesion_type} | Bounding Box: {b}")

    display_dicom(image, ds, boxes)

if __name__ == "__main__":
    main()
