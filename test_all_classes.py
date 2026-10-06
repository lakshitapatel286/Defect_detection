from ultralytics import YOLO
import os

# ==============================
# MODEL
# ==============================

MODEL_PATH = r"runs\detect\runs\defect_detection-4\weights\best.pt"

model = YOLO(MODEL_PATH)

# ==============================
# TEST IMAGES
# ==============================

BASE_PATH = r"C:\Users\hario\New folder\archive (2)\NEU-DET\validation\images"

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ==============================
# TEST ONE IMAGE FROM EACH CLASS
# ==============================

for class_name in classes:

    folder = os.path.join(BASE_PATH, class_name)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".jpg")
    ]

    if not images:
        print(f"\n{class_name}: NO IMAGE FOUND")
        continue

    image_path = os.path.join(folder, images[0])

    print("\n==============================")
    print(f"Testing: {class_name}")
    print(f"Image: {images[0]}")
    print("==============================")

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("NO DETECTION")
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        predicted_class = result.names[class_id]

        print(
            f"Predicted: {predicted_class} "
            f"| Confidence: {confidence:.2f}"
        )

print("\n==============================")
print("ALL 6 CLASSES TESTED")
print("==============================")