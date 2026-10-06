# AI-Based Industrial Defect Detection & Quality Inspection

An AI-based computer vision system for detecting and identifying surface defects in steel images using YOLO object detection and a Flask web application.

The system allows users to upload a steel surface image, automatically detect defect regions, display the annotated image with bounding boxes, show detection confidence, and maintain inspection history using SQLite.

---

## Project Overview

Quality inspection in steel manufacturing is an important process for identifying surface defects at an early stage. Manual inspection can be time-consuming and may lead to inconsistent results.

This project uses a YOLO-based object detection model to automatically identify and localize different types of steel surface defects.

A Flask web application provides a simple interface where users can:

- Upload a steel surface image
- Run automated defect detection
- View detected defect regions
- View defect type and confidence
- View the annotated detection image
- Store and review previous inspections

---

## Defect Classes

The model detects six types of steel surface defects:

| Class | Description |
|---|---|
| Crazing | Fine cracks or irregular surface patterns |
| Inclusion | Foreign material or inclusion-related defects |
| Patches | Patch-like surface defects |
| Pitted Surface | Small pits or cavities on the surface |
| Rolled-in Scale | Scale embedded into the steel surface |
| Scratches | Linear scratches or surface marks |

---

## Key Features

- YOLO-based object detection
- Six-class steel defect detection
- Bounding-box localization
- Confidence score for each detection
- Automatic annotated image generation
- Flask-based web interface
- Image preview before inspection
- SQLite-based inspection history
- Detection summary with defect count
- Separate result and history pages
- Simple and user-friendly interface

---

## Technology Stack

### Programming Language

- Python

### Machine Learning / Computer Vision

- YOLO
- Ultralytics
- PyTorch
- OpenCV
- NumPy

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Database

- SQLite

### Development Environment

- PyCharm

---

## Dataset

The project uses the **NEU-DET steel surface defect dataset**.

The dataset contains six categories of common steel surface defects:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The original dataset annotations are converted into YOLO format before training.

### Dataset Preparation

The project includes:

```text
convert_annotations.py
```

This script converts the original Pascal VOC XML annotations into YOLO-compatible label format.

The prepared dataset follows this structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

The YOLO labels contain the class ID and normalized bounding-box coordinates required for model training.

---

## Model Training

The YOLO model is trained using the prepared NEU-DET dataset.

Training is performed using:

```text
train.py
```

The trained model is evaluated using:

```text
validate.py
```

Prediction testing can be performed using:

```text
predict.py
```

---

## Project Structure

```text
Defect_detection/
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
│
├── app.py
├── train.py
├── validate.py
├── predict.py
├── convert_annotations.py
├── test_all_classes.py
├── data.yaml
├── .gitignore
└── yolo26n.pt
```

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/lakshitapatel286/Defect_detection.git
cd Defect_detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install ultralytics flask opencv-python numpy pillow torch torchvision pyyaml lxml matplotlib pandas
```

### 5. Prepare the Dataset

```bash
python convert_annotations.py
```

### 6. Train the YOLO Model

```bash
python train.py
```

### 7. Validate the Model

```bash
python validate.py
```

### 8. Run Prediction

```bash
python predict.py
```

### 9. Start the Flask Application

```bash
python app.py
```

### 10. Open the Application

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

### 11. Use the Application

Upload a steel surface image and click **Detect Defects** to view the detected defects, bounding boxes, confidence scores, and inspection history.
