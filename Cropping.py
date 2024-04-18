import cv2
import os

def crop_image(input_path, output_path, threshold=60):
    # Read the image
    image = cv2.imread(input_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to the grayscale image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (assuming it's the circular image)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image using the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]
    
    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)

# Path to the directory containing the images
input_dir = "./My_Selected_Images/"

# Path to the directory where cropped images will be saved
output_dir = "./Cropped_Selected_Images/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the input and output paths for the image
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Crop the image and save it
        crop_image(input_path, output_path)

print("Cropping complete.")
