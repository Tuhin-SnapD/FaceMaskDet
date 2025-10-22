from tensorflow.keras.models import load_model
from detect_mask_image import detect_mask
import argparse
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    # Instantiate an argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str,
                        default='MFN', choices=['MFN', 'RMFD'],
                        help="face mask detector model")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="minimum probability to filter weak face detections")
    args = parser.parse_args()

    # Change the working directory from src to root if needed
    current_full_dir = os.getcwd()
    print("Current working directory: " + current_full_dir)
    if current_full_dir.split("/")[-1] == "src":
        root = current_full_dir[:-4]
        os.chdir(root)
        print("Changed working directory to: " + root)

    # Validate arguments
    if args.model != "MFN" and args.model != "RMFD":
        raise ValueError("Please provide a valid model choice: `MFN` or `RMFD`.")
    if args.confidence > 1 or args.confidence < 0:
        raise ValueError("Please provide a valid confidence value between 0 and 1 (inclusive).")

    # Initialize model save path
    if args.model == "MFN":
        mask_detector_model_path = config.MASK_DETECTOR_MFN
    else:
        mask_detector_model_path = config.MASK_DETECTOR_RMFD
    confidence_threshold = args.confidence
    print("Mask detector save path: " + mask_detector_model_path)
    print("Face detector thresholding confidence: " + str(confidence_threshold))

    # Load the face detector model from disk
    print("[INFO] loading face detector model...")
    face_detector = cv2.dnn.readNet(config.FACE_DETECTOR_PROTOTXT, config.FACE_DETECTOR_WEIGHTS)

    # Load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    mask_detector = load_model(mask_detector_model_path)

    # Initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    capture = cv2.VideoCapture(0)

    # Loop over the frames from the video stream
    while capture.isOpened():
        # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        flags, frame = capture.read()

        # Detect faces in the frame and determine if they are wearing a face mask or not
        detect_mask(frame, face_detector, mask_detector, confidence_threshold)

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

