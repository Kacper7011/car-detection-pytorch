import torch
import torchvision
from torch.utils.data import DataLoader
from dataset_pytorch import CarDataset
from utils import collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_test = CarDataset(
    images_dir="dataset/images",
    labels_dir="dataset/labels",
    split_file="dataset/splits/test.txt",
)

loader = DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

models = {
    "FasterRCNN_ResNet50": torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT"),
    "RetinaNet_ResNet50": torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT"),
    "SSD300_VGG16": torchvision.models.detection.ssd300_vgg16(weights="DEFAULT"),
    "FCOS_ResNet50": torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT"),
    "FasterRCNN_MobileNet": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT"),
}

# --- Fine-tuned model ---
model_ft = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model_ft.load_state_dict(torch.load("fasterrcnn_car_finetuned.pth", map_location=DEVICE))

models["FasterRCNN_ResNet50_FineTuned"] = model_ft

for name, model in models.items():
    model.to(DEVICE)
    model.eval()

    print(f"\n=== {name} ===")

    total_detections = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            scores = outputs[0]["scores"]
            count = (scores > 0.7).sum().item()

            total_detections += count
            total_images += 1

    avg_detections = total_detections / total_images if total_images > 0 else 0

    print(f"Średnia liczba detekcji (score > 0.7): {avg_detections:.2f}")


