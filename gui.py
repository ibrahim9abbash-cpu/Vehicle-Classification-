import os
import cv2
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Vehicle Detection Pipeline")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # Environment setup
        self.input_folder = "extracted_frames"
        self.output_folder = "car_detected_frames"
        self.model = YOLO("yolov8l.pt")
        self.VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck

        # GUI layout elements
        self.label = tk.Label(root, text="YOLOv8 Vehicle Detection Pipeline", font=("Arial", 16, "bold"))
        self.label.pack(pady=20)

        self.upload_btn = tk.Button(root, text="Upload Video & Process", command=self.start_processing, 
                                    bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.upload_btn.pack(pady=10)

        # Progress elements
        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 10, "italic"))
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=15)

    def prepare_directories(self):
        """Creates clean temporary directories for processing."""
        for folder in [self.input_folder, self.output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            else:
                # Clear out any old remaining files
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    try:
                        if os.path.isfile(file_path): os.unlink(file_path)
                    except Exception as e: print(e)

    def start_processing(self):
        # 1. Ask user to upload video
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if not video_path:
            return

        self.upload_btn.config(state="disabled")
        self.prepare_directories()

        try:
            # 2. Extract frames from video
            self.status_label.config(text="Status: Extracting frames...")
            self.root.update()
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None: 
                fps = 30 # Fallback default
                
            frame_count = 0
            while True:
                ret, frame = cap.get(cv2.CAP_PROP_FPS), cap.read()
                if not frame[0]: # break if no frame read
                    break
                cv2.imwrite(os.path.join(self.input_folder, f"frame_{frame_count:05d}.jpg"), frame[1])
                frame_count += 1
            cap.release()

            # 3. Process frames with YOLO
            image_extensions = (".jpg", ".jpeg", ".png")
            frame_files = sorted([f for f in os.listdir(self.input_folder) if f.lower().endswith(image_extensions)])
            
            if not frame_files:
                raise Exception("Failed to extract frames from the video.")

            total_frames = len(frame_files)
            self.progress["maximum"] = total_frames
            
            detected_count = 0
            for idx, filename in enumerate(frame_files):
                self.status_label.config(text=f"Status: Running YOLO Detection ({idx+1}/{total_frames})...")
                self.progress["value"] = idx + 1
                self.root.update()

                file_path = os.path.join(self.input_folder, filename)
                frame = cv2.imread(file_path)
                if frame is None: continue

                results = self.model(frame, conf=0.15, classes=self.VEHICLE_CLASSES, verbose=False)

                if len(results[0].boxes) > 0:
                    # Save with identical sequential names to preserve chronological sorting order
                    output_path = os.path.join(self.output_folder, f"detected_{filename}")
                    annotated_frame = results[0].plot()
                    cv2.imwrite(output_path, annotated_frame)
                    detected_count += 1

            # 4. Merge frames into output video
            if detected_count == 0:
                raise Exception("No vehicles were detected anywhere in this video sequence.")

            self.status_label.config(text="Status: Compiling output video...")
            self.root.update()

            # Find and sort all generated frames
            output_images = sorted(glob.glob(os.path.join(self.output_folder, "detected_*.jpg")))
            
            first_frame = cv2.imread(output_images[0])
            height, width, _ = first_frame.shape

            output_video_name = "car_detection_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

            for img_path in output_images:
                img = cv2.imread(img_path)
                video_writer.write(img)
            
            video_writer.release()

            # Final success prompt
            self.status_label.config(text="Status: Process Complete!")
            messagebox.showinfo("Success", f"Processing complete!\nVideo saved as: {output_video_name}\n({detected_count} vehicle frames matched)")

        except Exception as e:
            self.status_label.config(text="Status: Error encountered")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            self.progress["value"] = 0
            self.upload_btn.config(state="normal")

if __name__ == "__main__":
    window = tk.Tk()
    app = VehicleDetectionApp(window)
    window.mainloop()
