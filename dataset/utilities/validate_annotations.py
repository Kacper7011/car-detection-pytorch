import os
from PIL import Image
import xml.etree.ElementTree as ET

# katalog utilities
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# katalog DATASET
DATASET_DIR = os.path.join(BASE_DIR, "..")

IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

os.makedirs(LBL_DIR, exist_ok=True)

created = 0

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    xml_name = img_name.rsplit(".", 1)[0] + ".xml"
    xml_path = os.path.join(LBL_DIR, xml_name)

    if os.path.exists(xml_path):
        continue  # XML już istnieje

    img_path = os.path.join(IMG_DIR, img_name)
    with Image.open(img_path) as img:
        width, height = img.size
        depth = len(img.getbands())

    # --- Pascal VOC ---
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = img_name
    ET.SubElement(annotation, "path").text = img_path

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    created += 1

print(f"Utworzono {created} pustych plików XML.")
