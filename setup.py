"""
Setup script for Face Mask Detection application.
"""
import os
import subprocess
import sys

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "data",
        "output", 
        "logs",
        "models/face_detector",
        "models/mask_detector"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def download_models():
    """Download required model files."""
    print("Note: Model files need to be downloaded separately.")
    print("Please ensure you have the following files:")
    print("- models/face_detector/deploy.prototxt")
    print("- models/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    print("- models/mask_detector/mask_detector_MFN.h5")
    print("- models/mask_detector/mask_detector_RMFD.h5")
    return True

def main():
    """Main setup function."""
    print("Setting up Face Mask Detection application...")
    print("=" * 50)
    
    # Create directories
    if not create_directories():
        print("✗ Failed to create directories")
        return False
    
    # Install requirements
    if not install_requirements():
        print("✗ Failed to install requirements")
        return False
    
    # Download models
    download_models()
    
    print("=" * 50)
    print("✓ Setup completed successfully!")
    print("\nTo run the application:")
    print("  python main.py")
    print("\nTo run detection scripts:")
    print("  python src/detect_mask_image.py -i example_images/pic1.jpg")
    print("  python src/detect_mask_video.py")
    
    return True

if __name__ == "__main__":
    main()
