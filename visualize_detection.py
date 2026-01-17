import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset_pytorch import CarDataset

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD = 0.7

# ======================
# DATASET
# ======================
dataset = CarDataset(
    images_dir="dataset/images",
    labels_dir="dataset/labels",
    split_file="dataset/splits/test.txt"
)

# ======================
# MODELS
# ======================
models = {}

# 1. Faster R-CNN ResNet50 (pretrained)
models["1"] = (
    "Faster R-CNN ResNet50 (pretrained)",
    torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
)

# 2. RetinaNet ResNet50
models["2"] = (
    "RetinaNet ResNet50",
    torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
)

# 3. SSD300 VGG16
models["3"] = (
    "SSD300 VGG16",
    torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
)

# 4. FCOS ResNet50
models["4"] = (
    "FCOS ResNet50",
    torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
)

# 5. Faster R-CNN MobileNet
models["5"] = (
    "Faster R-CNN MobileNet",
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
)

# 6. Faster R-CNN ResNet50 (fine-tuned)
model_ft = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
model_ft.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, 2
)
model_ft.load_state_dict(
    torch.load("fasterrcnn_car_finetuned.pth", map_location=DEVICE)
)

models["6"] = (
    "Faster R-CNN ResNet50 (fine-tuned)",
    model_ft
)

# Move models to device and eval mode
for key in models:
    models[key][1].to(DEVICE)
    models[key][1].eval()

# Default model
current_model_key = "6"
current_model_name, current_model = models[current_model_key]

# ======================
# VIEWER SETUP
# ======================
index = 0

fig = plt.figure(figsize=(14, 8))
ax_info = fig.add_axes([0.03, 0.15, 0.25, 0.7])
ax_info.axis("off")
ax_img = fig.add_axes([0.32, 0.1, 0.65, 0.8])


# ======================
# DRAW FUNCTIONS
# ======================
def draw_instructions():
    ax_info.clear()
    ax_info.axis("off")

    instructions = (
        "STEROWANIE:\n\n"
        "← / →   : poprzedni / następny obraz\n\n"
        "1       : Faster R-CNN ResNet50 (pretrained)\n"
        "2       : RetinaNet ResNet50\n"
        "3       : SSD300 VGG16\n"
        "4       : FCOS ResNet50\n"
        "5       : Faster R-CNN MobileNet\n"
        "6       : Faster R-CNN ResNet50 (fine-tuned)\n\n"
        "R       : odśwież obraz\n\n"
        "Q / ESC : wyjście"
    )

    ax_info.text(
        0.0, 1.0,
        instructions,
        fontsize=11,
        va="top",
        ha="left",
        wrap=True
    )

def draw_image():
    ax_img.clear()

    image, _ = dataset[index]
    image_gpu = image.to(DEVICE)

    with torch.no_grad():
        output = current_model([image_gpu])[0]

    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()

    ax_img.imshow(image.permute(1, 2, 0))

    for box, score in zip(boxes, scores):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax_img.add_patch(rect)
        ax_img.text(x1, y1 - 5, f"{score:.2f}", color="red", fontsize=9)

    ax_img.set_title(
        f"{current_model_name} | obraz {index + 1}/{len(dataset)}"
    )
    ax_img.axis("off")

    draw_instructions()
    fig.canvas.draw_idle()

# ======================
# KEYBOARD HANDLER
# ======================
def on_key(event):
    global index, current_model_key, current_model, current_model_name

    if event.key == "right":
        index = (index + 1) % len(dataset)
        draw_image()

    elif event.key == "left":
        index = (index - 1) % len(dataset)
        draw_image()

    elif event.key in models:
        current_model_key = event.key
        current_model_name, current_model = models[current_model_key]
        draw_image()

    elif event.key.lower() == "r":
        draw_image()

    elif event.key in ["escape", "q"]:
        plt.close(fig)

# ======================
# RUN
# ======================
fig.canvas.mpl_connect("key_press_event", on_key)
draw_image()
plt.show()
