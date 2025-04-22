# Import libraries
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import os
from pathlib import Path

from ultralytics import YOLO

# INPUT_DIRS
INPUT_DATA_DIR = Path("dataset")

## Drop the Folder if it already exists
DATASETS_DIR = Path("dataset")


# Image & labels directory
TRAIN_IMAGES_DIR = DATASETS_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASETS_DIR / "labels" / "train"
TEST_IMAGES_DIR = DATASETS_DIR / "images" / "test"
VAL_IMAGES_DIR = DATASETS_DIR / "images" / "val"
VAL_LABELS_DIR = DATASETS_DIR / "labels" / "val"


# Load train and test files
train = pd.read_csv(INPUT_DATA_DIR / "Train_df.csv")
val = pd.read_csv(INPUT_DATA_DIR / "Val_df.csv")
test = pd.read_csv(INPUT_DATA_DIR / "Test.csv")
ss = pd.read_csv(INPUT_DATA_DIR / "SampleSubmission.csv")

class_map = {cls: i for i, cls in enumerate(sorted(train["class"].unique().tolist()))}



# Strip any spacing from the class item and make sure that it is a str
train["class"] = train["class"].str.strip()

# Map {'healthy': 2, 'cssvd': 1, anthracnose: 0}
train["class_id"] = train["class"].map(class_map)


train[["class", "class_id"]].value_counts()


# Number of unique images path
train["ImagePath"].nunique()



import logging

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load a yolo pretrained model
# Load a yolo pretrained model
model = YOLO("yolo8n.pt")
# model = YOLO("runs/detect/train4/weights/best.pt")

# Fine tune model to our data
model.train(
    data="data.yaml",  # Path to the dataset configuration
    epochs=100,  # Number of epochs
    imgsz=1024,  # Image size (height, width)
    batch=32,  # Batch size
    patience=10,
    # device=(
    #     list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
    # ),  # Use all available GPUs
    device="cuda",  # Use the first GPU
    multi_scale=True,
    # cos_lr=True,
    box=9,
    dropout=0.1,
    mixup=0.1,
    max_det=30,
    nms=True,
    workers=4,
    flipud=0.3,
    cls=1.0,
    augment=True,
    copy_paste=0.1,
    copy_paste_mode="mixup",
    auto_augment="augmix",
)

from glob import glob

best_model = sorted(glob("runs/detect/train*/weights/best.pt"))[-1]

# Validate the model on the validation set
BEST_PATH = best_model
model = YOLO(BEST_PATH)
results = model.val()
