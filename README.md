# DICOM Tissue Classification

This project processes DICOM medical images, extracts metadata, and applies a pre-trained deep learning model to classify tissue types. It also displays the DICOM image for visual inspection.

## Features
- **DICOM Loading & Processing**: Reads and decompresses DICOM files if necessary.
- **Image Display**: Displays the medical image using Matplotlib.
- **Tissue Classification**: Uses a pre-trained ResNet18 model to classify tissue types from medical images.

## Installation
To install the necessary dependencies, run:
```bash
pip install pydicom numpy matplotlib torch torchvision pillow
```

## Usage
Run the script with:
```bash
python Dicome_processing.py
```

## Script Overview

### Load and Process DICOM Files
The script reads a DICOM file, extracts metadata, and ensures compatibility with the deep learning model.

### Display Image
The DICOM image is displayed using Matplotlib to allow for visual inspection.

### Classify Tissue Type
A pre-trained **ResNet18** model is used to classify the tissue type from the DICOM image. The classification result is printed in the console.

## Code Structure
```python
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
```

- **`load_dicom(filepath)`**: Reads and preprocesses the DICOM file.
- **`classify_tissue(image)`**: Uses deep learning to classify the tissue type.
- **`display_dicom(image, metadata)`**: Displays the medical image.
- **`main()`**: The main function that orchestrates loading, displaying, and classification.

## Example Output
```
DICOM Metadata:
Patient Name: Rubo DEMO^
Modality: XA
Study Date: 19941013
Image Dimensions: (512, 512)
Predicted Tissue Type: 419
```

## Future Improvements
- Implement support for more advanced models.
- Expand dataset for better classification.
- Add more detailed visualization and segmentation.

## License
This project is open-source and available under the MIT License.

---

### Author
Developed by [Your Name].

