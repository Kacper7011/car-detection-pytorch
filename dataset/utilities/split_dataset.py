import os
import random

IMG_DIR = "dataset/images"
OUT_DIR = "dataset/splits"

os.makedirs(OUT_DIR, exist_ok=True)

images = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"Znaleziono {len(images)} obrazów")

random.seed(42)
random.shuffle(images)

n = len(images)
train = images[:int(0.7 * n)]
val   = images[int(0.7 * n):int(0.85 * n)]
test  = images[int(0.85 * n):]

def save(name, data):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        for x in data:
            f.write(x + "\n")

save("train.txt", train)
save("val.txt", val)
save("test.txt", test)

print("Podział zakończony:")
print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
