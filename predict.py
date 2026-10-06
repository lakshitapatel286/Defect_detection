from ultralytics import YOLO
import os

# ============================================
# MODEL PATH
# ============================================

MODEL_PATH = r"runs\detect\runs\defect_detection-4\weights\best.pt"

# ============================================
# IMAGE TO TEST
# ============================================

IMAGE_PATH = r"C:\Users\hario\New folder\archive (2)\NEU-DET\train\images\scratches\scratches_6.jpg"

# ============================================
# LOAD MODEL
# ============================================

model = YOLO(MODEL_PATH)

# ============================================
# RUN PREDICTION
# ============================================

results = model.predict(
    source=IMAGE_PATH,
    save=True,
    conf=0.25
)

# ============================================
# DISPLAY RESULTS
# ============================================

for result in results:

    print("\n==============================")
    print("PREDICTION RESULTS")
    print("==============================")

    if result.boxes is None or len(result.boxes) == 0:
        print("No defect detected.")
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"Defect: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

print("\nPrediction completed.")