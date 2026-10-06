from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import os
import sqlite3
from datetime import datetime


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_FOLDER = os.path.join(
    BASE_DIR,
    "templates"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

DATABASE = os.path.join(
    BASE_DIR,
    "inspection.db"
)


# =========================================================
# TRAINED YOLO MODEL
# =========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "runs",
    "defect_detection_final-2",
    "weights",
    "best.pt"
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# =========================================================
# DATABASE
# =========================================================

def init_database():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            defect TEXT NOT NULL,
            regions INTEGER NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()

    conn.close()


init_database()


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("")
print("Loading YOLO model...")

if not os.path.exists(MODEL_PATH):

    print("ERROR: Trained model not found.")
    print("Expected model path:")
    print(MODEL_PATH)

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")
print("Model:", MODEL_PATH)
print("")


# =========================================================
# DEFECT DISPLAY NAMES
# =========================================================

DISPLAY_NAMES = {

    "crazing": "Crazing",

    "inclusion": "Inclusion",

    "patches": "Patches",

    "pitted_surface": "Pitted Surface",

    "rolled-in_scale": "Rolled-in Scale",

    "scratches": "Scratches"
}


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # Check uploaded image
    # -----------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No image selected."


    # -----------------------------------------------------
    # Fixed YOLO settings
    # -----------------------------------------------------
    #
    # 25% confidence threshold
    # 50% IoU threshold
    #
    # These are internal model settings.
    # They are not displayed as detection confidence.
    # -----------------------------------------------------

    confidence_threshold = 0.25

    iou_threshold = 0.50


    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    image_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(
        image_path
    )


    # -----------------------------------------------------
    # Run YOLO prediction
    # -----------------------------------------------------

    print("")
    print("Running prediction...")
    print("Image:", file.filename)


    results = model.predict(

        source=image_path,

        save=True,

        project=RESULT_FOLDER,

        name="prediction",

        exist_ok=True,

        conf=confidence_threshold,

        iou=iou_threshold
    )


    result = results[0]


    # -----------------------------------------------------
    # Process detections
    # -----------------------------------------------------

    grouped_detections = {}


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            detection_confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]


            # -------------------------------------------------
            # Create class entry
            # -------------------------------------------------

            if class_name not in grouped_detections:

                grouped_detections[class_name] = {

                    "name": DISPLAY_NAMES.get(
                        class_name,
                        class_name
                    ),

                    "count": 0,

                    "highest_confidence": 0.0
                }


            # -------------------------------------------------
            # Increase detection count
            # -------------------------------------------------

            grouped_detections[
                class_name
            ]["count"] += 1


            # -------------------------------------------------
            # Store highest confidence
            # -------------------------------------------------

            if (
                detection_confidence
                >
                grouped_detections[
                    class_name
                ]["highest_confidence"]
            ):

                grouped_detections[
                    class_name
                ]["highest_confidence"] = (
                    detection_confidence
                )


    # -----------------------------------------------------
    # Convert dictionary to list
    # -----------------------------------------------------

    detections = list(
        grouped_detections.values()
    )


    # -----------------------------------------------------
    # Convert confidence to percentage
    # -----------------------------------------------------

    for detection in detections:

        detection[
            "highest_confidence"
        ] = round(

            detection[
                "highest_confidence"
            ] * 100,

            2
        )


    # -----------------------------------------------------
    # Total detections
    # -----------------------------------------------------

    total_detections = sum(

        detection["count"]

        for detection in detections
    )


    # -----------------------------------------------------
    # Save inspection history
    # -----------------------------------------------------

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()


    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # -----------------------------------------------------
    # Save detected defects
    # -----------------------------------------------------

    if detections:

        for detection in detections:

            cursor.execute(
                """
                INSERT INTO inspections
                (
                    filename,
                    defect,
                    regions,
                    confidence,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,

                (
                    file.filename,

                    detection["name"],

                    detection["count"],

                    detection[
                        "highest_confidence"
                    ],

                    timestamp
                )
            )


    # -----------------------------------------------------
    # Save no-defect inspection
    # -----------------------------------------------------

    else:

        cursor.execute(
            """
            INSERT INTO inspections
            (
                filename,
                defect,
                regions,
                confidence,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?)
            """,

            (
                file.filename,

                "No Defect",

                0,

                0,

                timestamp
            )
        )


    conn.commit()

    conn.close()


    print("Prediction completed.")

    print(
        "Total detections:",
        total_detections
    )

    print("")


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        filename=file.filename,

        detections=detections,

        total_detections=total_detections
    )


# =========================================================
# RESULT IMAGE
# =========================================================

@app.route(
    "/results/<filename>"
)
def result_file(filename):

    result_folder = os.path.join(
        RESULT_FOLDER,
        "prediction"
    )


    return send_from_directory(

        result_folder,

        filename
    )


# =========================================================
# INSPECTION HISTORY
# =========================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    cursor.execute(
        """
        SELECT *
        FROM inspections
        ORDER BY id DESC
        """
    )


    inspections = cursor.fetchall()


    conn.close()


    return render_template(

        "history.html",

        inspections=inspections
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("")
    print("========================================")
    print("     AI DEFECT DETECTION SYSTEM")
    print("========================================")
    print("")
    print("Starting Flask server...")
    print("")
    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print("")


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False
    )