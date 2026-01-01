import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset_pytorch import CarDataset

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

# Pretrained model
model_pre = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model_pre.to(DEVICE)
model_pre.eval()

# Fine-tuned model
model_ft = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
model_ft.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, 2
)
model_ft.load_state_dict(torch.load("fasterrcnn_car_finetuned.pth", map_location=DEVICE))
model_ft.to(DEVICE)
model_ft.eval()

# ======================
# VISUALIZATION
# ======================

index = 0
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

def draw(ax, image, output, title):
    ax.clear()
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(title)
    ax.axis("off")

    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()

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
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{score:.2f}", color="red", fontsize=9)

def update():
    global index

    image, _ = dataset[index]
    image_gpu = image.to(DEVICE)

    with torch.no_grad():
        out_pre = model_pre([image_gpu])[0]
        out_ft = model_ft([image_gpu])[0]

    draw(axes[0], image, out_pre, f"Pretrained model | Image {index+1}/{len(dataset)}")
    draw(axes[1], image, out_ft, f"Fine-tuned model | Image {index+1}/{len(dataset)}")

    fig.canvas.draw_idle()

def on_key(event):
    global index

    if event.key == "right":
        index = (index + 1) % len(dataset)
        update()
    elif event.key == "left":
        index = (index - 1) % len(dataset)
        update()
    elif event.key in ["escape", "q"]:
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

update()
plt.show()
