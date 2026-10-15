# Vehicle-Classification-
<img width="617" height="360" alt="image" src="https://github.com/user-attachments/assets/caea53c4-64f0-416a-ab9b-cced498542be" />



# YOLOv8 Vehicle Detection Pipeline 🚗🚛

A complete computer vision desktop application built with **Python**, **OpenCV**, and **Ultralytics YOLOv8**. This pipeline provides an intuitive Graphical User Interface (GUI) to break down input videos into individual frames, isolate vehicle classes using a heavy-weight model layer, and recompile the detected sequences seamlessly back into a finalized video.

## 🌟 Features
* **Intuitive GUI:** Built completely on Tkinter with dynamic real-time progress bars and application state logging.
* **Smart Filtering:** Isolates specific COCO targets (Cars, Trucks, Buses, and Motorcycles) while suppressing irrelevant default objects.
* **Dynamic Resolution & Frame Rates:** Ingests any aspect ratio automatically and retains the exact frame-per-second (FPS) value of your uploaded asset.
* **Auto-Cleanup Routine:** Automatically flushes localized frame cache environments between operations to prevent data overlap.

---

## 🛠️ Project Structure
```text
YOLO/
│
├── car_detected_frames/     # Temporary workspace for annotated vehicle frames
├── extracted_frames/        # Temporary workspace for raw source frames
│
├── yolov8l.pt              # The local YOLOv8 Large weights file
├── app.py                  # Core Tkinter GUI pipeline code
└── README.md               # Documentation guide
```

---

## 🚀 Setup and Installation

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your operating system.

### 2. Clone this Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 3. Install Dependencies
Install the required tracking and visualization frameworks via pip:
```bash
pip install ultralytics opencv-python
```
*(Note: `tkinter`, `os`, and `glob` are bundled directly inside the Python standard library).*

---

## 💻 How to Run

Execute the pipeline framework using your terminal:
```bash
python app.py
```

### Application Execution Steps:
1. Click the **Upload Video & Process** button.
2. Select your target traffic video file (`.mp4`, `.avi`, `.mov`, etc.).
3. The software will automatically isolate individual sequence fields, pass them chronologically to the **YOLOv8l** neural layers, filter out empty fields, and generate a structural `car_detection_output.mp4` sequence inside your root working tree.

---

## ⚙️ Model Detection Settings
The pipeline targets specific vehicle categories via their specific COCO identifier arrays:
* **Classes Tracked:** Cars (`2`), Motorcycles (`3`), Buses (`5`), and Trucks (`7`).
* **Confidence Floor:** Bound tightly at `conf=0.15` to preserve strict vehicle extraction over unique high-angle and top-down overhead perspective frames.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
