# Import libraries
import os
os.environ["WANDB_PROJECT"] = "zindi_challenge_cacao"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "none"

import logging
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
from glob import glob

def setup_directories(base_dir):
    """Setup dataset directories."""
    datasets_dir = Path(base_dir)
    
    return {
        'datasets_dir': datasets_dir,
        'train_images_dir': datasets_dir / "images" / "train",
        'train_labels_dir': datasets_dir / "labels" / "train",
        'test_images_dir': datasets_dir / "images" / "test",
        'val_images_dir': datasets_dir / "images" / "val",
        'val_labels_dir': datasets_dir / "labels" / "val"
    }

def load_data(input_data_dir):
    """Load train, validation and test data."""
    input_dir = Path(input_data_dir)
    
    train = pd.read_csv(input_dir / "Train_df.csv")
    val = pd.read_csv(input_dir / "Val_df.csv")
    test = pd.read_csv(input_dir / "Test.csv")
    ss = pd.read_csv(input_dir / "SampleSubmission.csv")
    
    # Process class mapping
    train["class"] = train["class"].str.strip()
    class_map = {cls: i for i, cls in enumerate(sorted(train["class"].unique().tolist()))}
    train["class_id"] = train["class"].map(class_map)
    
    return train, val, test, ss, class_map

def train_model(args):
    """Train YOLO model with provided arguments."""
    # Load model
    model = YOLO(args.model)
    
    # Configure device
    if args.device == 'auto':
        device = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Train the model
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        device=device,
        multi_scale=args.multi_scale,
        dropout=args.dropout,
        mixup=args.mixup,
        max_det=args.max_det,
        nms=args.nms,
        workers=args.workers if args.workers else args.batch_size // 2,
        flipud=args.flipud,
        cls=args.cls_loss,
        augment=args.augment,
        copy_paste=args.copy_paste,
        copy_paste_mode=args.copy_paste_mode,
        auto_augment=args.auto_augment,
        half=args.half,
        resume=args.resume,
        project="zindi_challenge_cacao"
    )
    
    return model

def validate_model(model_path=None):
    """Validate the best model."""
    if model_path is None:
        best_model = sorted(glob("runs/detect/train*/weights/best.pt"))[-1]
    else:
        best_model = model_path
    
    print(f"Validating model: {best_model}")
    model = YOLO(best_model)
    results = model.val()
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a YOLO model for object detection")
    
    # Dataset related
    parser.add_argument('--data_dir', type=str, default="dataset", help='Base directory for dataset')
    parser.add_argument('--data_yaml', type=str, default="data.yaml", help='Path to data YAML configuration')
    
    # Model related
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='Path to model or model name')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--validate_only', action='store_true', help='Only validate the best model without training')
    parser.add_argument('--model_path', type=str, help='Specific model path to validate (used with --validate_only)')
    
    # Training parameters
    parser.add_argument('--img_size', type=int, default=2048, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--workers', type=int, help='Number of worker threads for dataloader')
    
    # Augmentation and training options
    parser.add_argument('--multi_scale', action='store_true', default=True, help='Use multi-scale training')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha for augmentation')
    parser.add_argument('--max_det', type=int, default=150, help='Maximum detections per image')
    parser.add_argument('--nms', action='store_true', default=True, help='Use NMS')
    parser.add_argument('--flipud', type=float, default=0.3, help='Probability of flipping up-down')
    parser.add_argument('--cls_loss', type=float, default=1.0, help='Classification loss weight')
    parser.add_argument('--augment', action='store_true', default=True, help='Use augmentation')
    parser.add_argument('--copy_paste', type=float, default=0.1, help='Copy-paste augmentation')
    parser.add_argument('--copy_paste_mode', type=str, default='mixup', help='Copy-paste mode')
    parser.add_argument('--auto_augment', type=str, default='augmix', help='Auto augmentation mode')
    parser.add_argument('--half', action='store_true', default=True, help='Use half precision (FP16)')
    
    return parser.parse_args()

def main():
    """Main function to run the training process."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Parse arguments
    args = parse_args()
    
    # Setup directories
    dirs = setup_directories(args.data_dir)
    
    # If only validating
    if args.validate_only:
        results = validate_model(args.model_path)
        logging.info(f"Validation results: {results}")
        return
    
    # Load data
    train, val, test, ss, class_map = load_data(args.data_dir)
    logging.info(f"Loaded data with classes: {class_map}")
    
    # Train model
    logging.info(f"Starting training with model: {args.model}")
    model = train_model(args)
    
    # Validate best model
    results = validate_model()
    logging.info(f"Validation results: {results}")

if __name__ == "__main__":
    main()
