"""
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

from .coco_dataset import (
    CocoDetection as CocoDetection,
    CocoDetection_share_memory as CocoDetection_share_memory,
    mscoco_category2label as mscoco_category2label,
    mscoco_label2category as mscoco_label2category,
    mscoco_category2name as mscoco_category2name,
)
from .coco_eval import *

from .coco_utils import get_coco_api_from_dataset as get_coco_api_from_dataset
