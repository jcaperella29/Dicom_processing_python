# Dicom_processing_python



This repository contains a Python script for loading, processing, and displaying **DICOM (Digital Imaging and Communications in Medicine) files**. The script is designed for use in **radiology and medical physics applications**, allowing users to extract metadata, handle compressed images, and visualize DICOM images.

## Features
- **Load DICOM Files**: Reads `.dcm` files using `pydicom`.
- **Metadata Extraction**: Displays patient details, modality, study date, and image dimensions.
- **Handles Multi-Frame (3D) Images**: Selects the middle slice for visualization.
- **Decompresses Compressed Images**: Supports JPEG-based DICOM compression (requires `gdcm` or `pylibjpeg`).
- **Image Visualization**: Uses `matplotlib` to display grayscale medical images.

## Installation
To run the script, install the required dependencies:

```bash
pip install pydicom matplotlib numpy gdcm pylibjpeg pylibjpeg-libjpeg pillow
```

If `gdcm` fails to install via `pip`, try:
```bash
conda install -c conda-forge gdcm
```

## Usage
1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/dicom-processing.git
cd dicom-processing
```

2. **Run the Script**
```bash
python Dicome_processing.py
```

3. **Select a DICOM File** (or modify `filepath` in the script).

## Example Output
```
DICOM Metadata:
Patient Name: Sample Patient
Modality: CT
Study Date: 20230201
Image Dimensions: (512, 512)
```
The selected **DICOM slice** will then be displayed in a window.

## Troubleshooting
- If you get a **decompression error**, ensure `gdcm` or `pylibjpeg` is installed.
- If your image is **multi-frame (3D)**, the script will automatically pick the middle slice.
- If `pydicom` raises an **invalid shape error**, ensure you are selecting a valid 2D slice.

## Contributing
Pull requests are welcome! If you have improvements or additional features, feel free to contribute.

## License
MIT License. Free to use and modify.

