import torch
import torchvision
from torch.utils.data import DataLoader
from dataset_pytorch import CarTrainDataset, CarDataset
from utils import collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # background + car
EPOCHS = 7
LR = 0.005

# ======================
# DATASETS
# ======================

# Train dataset – ONLY images with objects
train_dataset = CarTrainDataset(
    images_dir="dataset/images",
    labels_dir="dataset/labels",
    split_file="dataset/splits/train.txt"
)

# Validation dataset – can include empty images
val_dataset = CarDataset(
    images_dir="dataset/images",
    labels_dir="dataset/labels",
    split_file="dataset/splits/val.txt"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

# ======================
# MODEL
# ======================

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.to(DEVICE)

# ======================
# OPTIMIZER
# ======================

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=LR,
    momentum=0.9,
    weight_decay=0.0005
)

# ======================
# TRAINING LOOP
# ======================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, targets in train_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train loss: {avg_loss:.4f}")

# ======================
# SAVE MODEL
# ======================

torch.save(model.state_dict(), "fasterrcnn_car_finetuned.pth")
print("Model zapisany jako fasterrcnn_car_finetuned.pth")
