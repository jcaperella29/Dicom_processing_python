📌 Overview
This project detects and classifies lesions in DICOM medical images using DeepLesion (Faster R-CNN) for detection and ResNet50 for classification.

⚡ Features
✅ Lesion Detection – Finds lesion locations using DeepLesion (Faster R-CNN).
✅ Lesion Classification – Identifies lesion type using ResNet50.
✅ False Positive Filtering – Removes misclassified structures (e.g., blood vessels).
✅ Bounding Box Overlays – Draws lesion location & label on the image.

🛠️ Installation
1️⃣ Install Required Dependencies
bash
Copy
Edit
pip install torch torchvision pydicom numpy matplotlib opencv-python
2️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/YOUR_GITHUB_USERNAME/DICOM-Lesion-Detection.git
cd DICOM-Lesion-Detection
3️⃣ Clone DeepLesion (Required for Detection)
bash
Copy
Edit
git clone https://github.com/rsummers11/CADLab.git
cd CADLab/deep-lesion
This contains the pretrained DeepLesion model for detecting lesions.

4️⃣ Run the Script
Modify the DICOM file path in Dicome_processing.py:

python
Copy
Edit
filepath = r"C:\Users\ccape\Downloads\Radiology_script\sample.dcm"
Then, run:

bash
Copy
Edit
python Dicome_processing.py
📂 Project Structure
graphql
Copy
Edit
DICOM-Lesion-Detection/
│── Dicome_processing.py  # Main script for detection & classification
│── sample.dcm            # Example DICOM file (optional)
│── README.md             # Project documentation
│── CADLab/               # DeepLesion (cloned repo)
│   ├── deep-lesion/      # Pretrained model & utilities
📊 How It Works
1️⃣ Load & Preprocess DICOM
Reads DICOM images and extracts metadata.
Converts grayscale images to RGB (for compatibility).
2️⃣ Detect Lesions
DeepLesion (Faster R-CNN) locates potential lesions.
Filters false positives like thickened blood vessels.
3️⃣ Classify Lesion Type
ResNet50 model classifies detected lesions as:
Tumor
Cyst
Hemorrhage
Inflammation
4️⃣ Display Results
Bounding boxes highlight detected lesions on the DICOM image.
Labels & confidence scores annotate each lesion type.
📌 Example Output
Detected Lesions:
✅ Tumor (Score: 0.92) – Bounding Box: (120, 80, 200, 160)
✅ Cyst (Score: 0.85) – Bounding Box: (300, 240, 380, 320)

📷 Annotated Image Output (Lesion Bounding Boxes & Labels)

🚀 Next Steps
🔍 Explainability (Grad-CAM)
✅ Generate heatmaps showing which lesion areas influenced classification.

🤖 Multimodal AI (Lesion Detection + Clinical Reports)
✅ Use NLP models to summarize findings automatically.

📄 DICOM Export with Annotations
✅ Save annotated images back into DICOM format for PACS/Radiology.

📢 Contributing
Want to improve this project? Contributions are welcome! 🚀

Fork the repo
Create a feature branch
Commit your changes
Open a pull request
🏆 Acknowledgments
Built using:

DeepLesion (NIH) – Universal lesion detection dataset.
TorchVision Faster R-CNN – Object detection model.
RadImageNet – ResNet50 trained on medical images.

