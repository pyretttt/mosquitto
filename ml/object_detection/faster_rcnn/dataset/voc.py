import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


CLASSES = [
    'aeroplane', 'background', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

IDX2LABEL = dict(enumerate(CLASSES))
LABEL2IDX = {label: idx for idx, label in IDX2LABEL.items()}


def load_dataset_info(im_dir, ann_dir, label2idx = LABEL2IDX):
    infos = []
    
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, "*.xml"))):
        info = {}
        info["img_id"] = os.path.basename(ann_file).split(".xml")[0]
        info["filename"] = os.path.join(im_dir, f"{info['img_id']}.jpg")
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        info["width"] = width
        info["height"] = height
        info["detections"] = []

        for obj in ann_info.findall("object"):
            det = {}
            det["label"] = label2idx[obj.find("name").text]
            bbox_info = obj.find("bndbox")
            det["bbox"]  = [
                int(float(bbox_info.find("xmin").text)) - 1,
                int(float(bbox_info.find("ymin").text)) - 1,
                int(float(bbox_info.find("xmax").text)) - 1,
                int(float(bbox_info.find("ymax").text)) - 1
            ]
            info["detections"].append(det)
        
        infos.append(info)
    
    print(f"Total images found {len(infos)}")
    
    return infos


def get_annotations_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "Annotations")


def get_images_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "JPEGImages")


class VocDataset(Dataset):
    def __init__(
        self, 
        data_directory: str,
        
    ):
        self.data_directory = data_directory
        self.image_info = load_dataset_info(
            im_dir=get_images_dir(data_directory),
            ann_dir=get_annotations_dir(data_directory)
        )


    def __len__(self):
        return len(self.image_info)


    def __getitem__(self, index):
        im_info = self.image_info[index]
        im = Image.open(im_info["filename"])
        
        return {
            "img_id": im_info["img_id"],
            "filename": im_info["filename"],
            "im": im,
            "orig_width": im_info["width"],
            "orig_height": im_info["height"],
            "targets": {
                "bboxes": torch.as_tensor([det["bbox"] for det in im_info["detections"]]),
                "labels": torch.as_tensor([det["label"] for det in im_info["detections"]])
            }
        }
        
        
        
if __name__ == "__main__":
    import yaml
    with open("object_detection/faster_rcnn/config.yaml") as file:
        config = yaml.safe_load(file)
    
    voc_dataset = VocDataset(config["train_path"])
    print(next(iter(voc_dataset)))