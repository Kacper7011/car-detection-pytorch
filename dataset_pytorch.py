import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import xml.etree.ElementTree as ET



class CarDataset(Dataset):
    def __init__(self, images_dir, labels_dir, split_file, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        with open(split_file) as f:
            self.images = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.images_dir, img_name)
        xml_path = os.path.join(
            self.labels_dir,
            img_name.rsplit(".", 1)[0] + ".xml"
        )

        image = F.to_tensor(Image.open(img_path).convert("RGB"))

        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # 1 = car (0 = background)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
