import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import convert_color_space

def load_dicom(filepath):
    """Load a DICOM file and extract image & metadata, handling compression"""
    ds = pydicom.dcmread(filepath, force=True)

    # Check if the DICOM image is compressed
    transfer_syntax = ds.file_meta.TransferSyntaxUID if 'TransferSyntaxUID' in ds.file_meta else None
    if transfer_syntax and transfer_syntax.is_compressed:
        print(f"Compressed image detected with syntax: {transfer_syntax}")
        try:
            ds.decompress()
            print("Decompressed DICOM image successfully.")
        except Exception as e:
            print(f"Error decompressing DICOM: {e}")
            return None, None

    image = ds.pixel_array  # Extract image data

    # Handle multi-frame (3D) images: Pick the middle slice
    if len(image.shape) == 3:
        print(f"Multi-frame image detected with {image.shape[0]} slices. Selecting middle slice.")
        image = image[image.shape[0] // 2]  # Select the middle slice

    # Convert color images if needed
    if len(image.shape) == 3 and image.shape[-1] in [3, 4]:  # RGB or RGBA images
        image = convert_color_space(image, ds.PhotometricInterpretation, 'RGB')

    return ds, image

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
    """Main function to load and display a DICOM file"""
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

if __name__ == "__main__":
    main()
