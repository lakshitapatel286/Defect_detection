import os
import shutil
import xml.etree.ElementTree as ET

# ============================================================
# PATHS
# ============================================================

SOURCE_DATASET = r"C:\Users\hario\New folder\archive (2)\NEU-DET"

PROJECT_DATASET = r"C:\Users\hario\PycharmProjects\Defect_detection\dataset"


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for folder in [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
]:
    os.makedirs(
        os.path.join(PROJECT_DATASET, folder),
        exist_ok=True
    )


# ============================================================
# FIND ALL IMAGES
# ============================================================

print("\nBuilding image index...")

image_index = {}

for split in ["train", "validation"]:

    images_root = os.path.join(
        SOURCE_DATASET,
        split,
        "images"
    )

    for root, dirs, files in os.walk(images_root):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(root, file)

                # Store using filename without extension
                image_key = os.path.splitext(file)[0]

                image_index[image_key] = image_path


print(f"Found {len(image_index)} images in total.")


# ============================================================
# CONVERT DATASET
# ============================================================

def convert_split(annotation_split, output_split):

    annotation_folder = os.path.join(
        SOURCE_DATASET,
        annotation_split,
        "annotations"
    )

    output_images = os.path.join(
        PROJECT_DATASET,
        "images",
        output_split
    )

    output_labels = os.path.join(
        PROJECT_DATASET,
        "labels",
        output_split
    )

    print("\n========================================")
    print(f"Processing: {annotation_split}")
    print("========================================")

    xml_files = [
        file
        for file in os.listdir(annotation_folder)
        if file.lower().endswith(".xml")
    ]

    print(f"Found {len(xml_files)} XML files.")

    converted = 0
    missing = 0

    for xml_file in xml_files:

        xml_path = os.path.join(
            annotation_folder,
            xml_file
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ----------------------------------------------------
        # Get image filename from XML
        # ----------------------------------------------------

        filename_element = root.find("filename")

        if filename_element is None:
            print(f"WARNING: No filename in {xml_file}")
            missing += 1
            continue

        xml_filename = filename_element.text.strip()

        # Remove extension
        image_key = os.path.splitext(xml_filename)[0]

        # ----------------------------------------------------
        # Find actual image
        # ----------------------------------------------------

        image_source = image_index.get(image_key)

        if image_source is None:

            print(
                f"WARNING: Image not found for {xml_file}"
            )

            missing += 1
            continue

        # ----------------------------------------------------
        # Use ACTUAL source filename
        # ----------------------------------------------------

        actual_filename = os.path.basename(image_source)

        actual_stem = os.path.splitext(actual_filename)[0]

        # ----------------------------------------------------
        # Image size
        # ----------------------------------------------------

        size = root.find("size")

        image_width = int(
            size.find("width").text
        )

        image_height = int(
            size.find("height").text
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = os.path.join(
            output_images,
            actual_filename
        )

        shutil.copy2(
            image_source,
            destination_image
        )

        # ----------------------------------------------------
        # YOLO label
        # ----------------------------------------------------

        label_filename = actual_stem + ".txt"

        label_path = os.path.join(
            output_labels,
            label_filename
        )

        with open(label_path, "w") as label_file:

            for obj in root.findall("object"):

                class_element = obj.find("name")

                if class_element is None:
                    continue

                class_name = class_element.text.strip()

                if class_name not in CLASS_MAP:

                    print(
                        f"WARNING: Unknown class "
                        f"{class_name} in {xml_file}"
                    )

                    continue

                class_id = CLASS_MAP[class_name]

                box = obj.find("bndbox")

                xmin = float(
                    box.find("xmin").text
                )

                ymin = float(
                    box.find("ymin").text
                )

                xmax = float(
                    box.find("xmax").text
                )

                ymax = float(
                    box.find("ymax").text
                )

                # ------------------------------------------------
                # Pascal VOC → YOLO
                # ------------------------------------------------

                x_center = (
                    (xmin + xmax) / 2
                ) / image_width

                y_center = (
                    (ymin + ymax) / 2
                ) / image_height

                width = (
                    xmax - xmin
                ) / image_width

                height = (
                    ymax - ymin
                ) / image_height

                label_file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

        converted += 1

    print(f"Converted: {converted}")
    print(f"Missing: {missing}")


# ============================================================
# TRAIN
# ============================================================

convert_split(
    "train",
    "train"
)


# ============================================================
# VALIDATION
# ============================================================

convert_split(
    "validation",
    "val"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("YOLO DATASET CONVERSION COMPLETED")
print("========================================")