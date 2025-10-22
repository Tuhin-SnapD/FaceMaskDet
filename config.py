"""
Configuration file for Face Mask Detection application.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
FACE_DETECTOR_PROTOTXT = os.path.join(BASE_DIR, "models", "face_detector", "deploy.prototxt")
FACE_DETECTOR_WEIGHTS = os.path.join(BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

# Mask detector models
MASK_DETECTOR_MFN = os.path.join(BASE_DIR, "models", "mask_detector", "mask_detector_MFN.h5")
MASK_DETECTOR_RMFD = os.path.join(BASE_DIR, "models", "mask_detector", "mask_detector_RMFD.h5")

# Asset paths
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_NO_MASK = os.path.join(ASSETS_DIR, "no_mask_US_female.mp3")
AUDIO_MASK_INCORRECT = os.path.join(ASSETS_DIR, "mask_incorrect_US_female.mp3")

# Example images
EXAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "example_images")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, FIGURES_DIR, CHECKPOINTS_DIR]:
    os.makedirs(directory, exist_ok=True)
