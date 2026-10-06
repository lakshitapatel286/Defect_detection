from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Train the model
results = model.train(
    data="data.yaml",
    epochs=20,
    imgsz=416,
    batch=8,
    patience=15,
    project="runs",
    name="defect_detection_final"
)

print("Training completed!")