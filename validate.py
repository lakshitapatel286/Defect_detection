from ultralytics import YOLO

# Trained YOLO model
MODEL_PATH = r"runs\detect\runs\defect_detection_final-2\weights\best.pt"

# Load model
model = YOLO(MODEL_PATH)

# Validate model
metrics = model.val(
    data="data.yaml",
    imgsz=416
)

print("\n==============================")
print("VALIDATION COMPLETED")
print("==============================")

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")