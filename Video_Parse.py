import cv2
import os
import datetime

def extract_frames(video_path, output_folder, frame_rate):
    # Opens video file
    cap = cv2.VideoCapture(video_path)

    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame interval based on the frame rate
    interval = int(fps / frame_rate)

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Loop through the frames
    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame as an image with a timestamped filename
        output_path = os.path.join(output_folder, f"frame_{timestamp}_{i}.jpg")
        cv2.imwrite(output_path, frame)

    # Release the video capture object
    cap.release()

    print(f"Frames extracted and saved to {output_folder}")

if __name__ == "__main__":
    # Path to the video file
    video_path = r"C:\Users\kavya\OneDrive\Documents\MVIT_Datasets\2020-10-21_100525_VID013.mp4"

    # Specify the output folder to save images
    output_folder = "output_frames"

    # Specify the frame rate (frames per second)
    frame_rate = 20  

    # Call the function
    extract_frames(video_path, output_folder, frame_rate)

