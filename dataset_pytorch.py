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

    def _parse_xml(self, xml_path):
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
            labels.append(1)  # 1 = car

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return boxes, labels

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.images_dir, img_name)
        xml_path = os.path.join(
            self.labels_dir,
            img_name.rsplit(".", 1)[0] + ".xml"
        )

        image = F.to_tensor(Image.open(img_path).convert("RGB"))

        boxes, labels = self._parse_xml(xml_path)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


class CarTrainDataset(CarDataset):
    """
    Dataset ONLY for training Faster R-CNN.
    Filters out images without any objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valid_indices = []
        for i in range(len(self.images)):
            img_name = self.images[i]
            xml_path = os.path.join(
                self.labels_dir,
                img_name.rsplit(".", 1)[0] + ".xml"
            )

            tree = ET.parse(xml_path)
            root = tree.getroot()

            if len(root.findall("object")) > 0:
                self.valid_indices.append(i)

        print(f"[INFO] Train dataset size (with objects): {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.valid_indices[idx])
