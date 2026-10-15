import cv2
import glob
import os

# 1. Define folder path and output settings
image_folder = 'car_detected_frames'
video_name = 'car_detection_output.mp4'
fps = 5  # Adjust the frame rate (frames per second) as needed

# 2. Gather and sort all matching jpg images chronologically
images = sorted(glob.glob(os.path.join(image_folder, "detected_second_*.jpg")))

if not images:
    print(f"No images found in '{image_folder}' matching the pattern.")
    exit()

# 3. Read the first frame to dynamically get the width and height
first_image_path = images[0]
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 4. Initialize the VideoWriter object
# Use 'mp4v' for MP4 format or 'XVID' for AVI format
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

print(f"Processing {len(images)} frames into {video_name}...")

# 5. Loop through images and write each to the video stack
for image_path in images:
    img = cv2.imread(image_path)
    video.write(img)

# 6. Release the video file structure from memory
video.release()
cv2.destroyAllWindows()

print("Video creation complete!")
