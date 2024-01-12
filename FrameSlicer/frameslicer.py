import cv2
import os

def extract_frames(video_path, interval, output_folder):
    """
    Extracts frames from a video at a specified interval.

    :param video_path: Path to the video file.
    :param interval: Interval in seconds to extract frames.
    :param output_folder: Folder to save the extracted frames.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames.")

# Example usage
extract_frames(r"C:\Users\phili\OneDrive\Year4\SE4450\MVIT2\MVIT-2.0\FrameSlicer\Videos\2023-09-26_070005_VID002.mp4", 12, r"C:\Users\phili\OneDrive\Year4\SE4450\MVIT2\MVIT-2.0\FrameSlicer\OutputTest4")
