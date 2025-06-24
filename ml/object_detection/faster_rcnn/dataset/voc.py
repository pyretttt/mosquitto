import glob
import os
import random

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

IDX2LABEL = dict(enumerate(CLASSES))
LABEL2IDX = {label: idx for idx, label in IDX2LABEL}


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
        detections = []

        for obj in ann_info.findall("object"):
            det = {}
            label = label2idx[obj.find("name").text]
            bbox_info = obj.find("bndbox")
            bbox = [
                
            ]
        
    
class VocDataset(Dataset):
    ...   
