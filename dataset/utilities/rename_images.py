import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "..", "images")

PREFIX = "car"
EXT = ".png"
START_INDEX = 1

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(EXT)
])

digits = len(str(len(files)))

for i, filename in enumerate(files, start=START_INDEX):
    new_name = f"{PREFIX}_{str(i).zfill(digits)}{EXT}"
    src = os.path.join(IMAGE_DIR, filename)
    dst = os.path.join(IMAGE_DIR, new_name)

    if src != dst:
        os.rename(src, dst)

print(f"Zmieniono nazwy {len(files)} plików.")
