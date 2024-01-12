import cv2

# Change interval and reduce framerate for areas with good quality images

video_path = "./2021-03-18_083909_VID010.mp4" # Path to video being broken up
output_folder = "./Images/03-18-VID10/" # directory where frames are saved
start_frame = 1  # Change the starting frame based on each video
end_frame = 6000  # Set the ending frame, usually 12000+ unless you want to stop gathering frames early

videoInput = cv2.VideoCapture(video_path)

# Set the starting frame position
videoInput.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

count = start_frame
frame_interval = 500  # Save every nth frame, usually 500

while videoInput.isOpened():
    success, image = videoInput.read()

    if not success or count > end_frame:
        break

    if count % frame_interval == 0:
        cv2.imwrite(output_folder + "%d.jpg" % count, image)

    count += 1

videoInput.release()
