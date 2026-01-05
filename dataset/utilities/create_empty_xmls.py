import os
import xml.etree.ElementTree as ET
from PIL import Image

# katalog utilities
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# katalog DATASET
DATASET_DIR = os.path.join(BASE_DIR, "..")

IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

errors = 0

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.lower().endswith(".png"):
        continue

    xml_name = img_name.replace(".png", ".xml")
    xml_path = os.path.join(LBL_DIR, xml_name)

    if not os.path.exists(xml_path):
        print(f"[BRAK XML] {img_name}")
        errors += 1
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    with Image.open(img_path) as img:
        w, h = img.size

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            print(f"[BRAK BND BOX] {img_name}")
            errors += 1
            continue

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        if (
            xmin < 0 or ymin < 0 or
            xmax > w or ymax > h or
            xmin >= xmax or ymin >= ymax
        ):
            print(f"[BŁĘDNY BOX] {img_name}")
            errors += 1

if errors == 0:
    print("Walidacja zakończona: brak błędów ✅")
else:
    print(f"Wykryto {errors} problemów ❌")
