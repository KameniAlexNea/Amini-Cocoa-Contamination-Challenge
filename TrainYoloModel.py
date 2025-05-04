# Import libraries
import json
import os

os.environ["WANDB_PROJECT"] = "zindi_challenge_cacao"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "none"

import wandb

wandb.login(key=os.environ["WANDB_API_KEY"], host="https://api.wandb.ai")

import logging
import argparse
from ultralytics import YOLO
import torch
from glob import glob


def train_model(args):
    """Train YOLO model with provided arguments."""
    # Load model
    model = YOLO(args.model)

    # Configure device
    if args.device == "auto":
        device = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    config = {}
    if args.config:
        config: dict = json.load(open(args.config, "r"))
        config.pop("data", None)

    additional_args = {
        "multi_scale": args.multi_scale,
        "dropout": args.dropout,
        "mixup": args.mixup,
        "max_det": args.max_det,
        "nms": args.nms,
        "flipud": args.flipud,
        "cls": args.cls_loss,
        "augment": args.augment,
        "copy_paste": args.copy_paste,
        "copy_paste_mode": args.copy_paste_mode,
        "auto_augment": args.auto_augment,
        "half": args.half,
        **config,
        "data": args.data_yaml,
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "patience": args.patience,
        "device": device,
        "workers": args.workers if args.workers else args.batch_size // 2,
    }
    if args.classes is not None:
        additional_args["classes"] = [args.classes]

    print(f"Training with the following parameters:\n {additional_args}")

    # Train the model
    model.train(**additional_args, resume=args.resume, project="zindi_challenge_cacao")

    return model


def validate_model(model_path=None):
    """Validate the best model."""
    if model_path is None:
        best_model = sorted(glob("zindi_challenge_cacao/train*/weights/best.pt"))[-1]
    else:
        best_model = model_path

    print(f"Validating model: {best_model}")
    model = YOLO(best_model)
    results = model.val()
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for object detection"
    )

    # Dataset related
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="data.yaml",
        help="Path to data YAML configuration",
    )
    parser.add_argument(
        "--config", type=str, default="", help="Path to tuning config file"
    )

    # Model related
    parser.add_argument(
        "--model", type=str, default="yolo11l.pt", help="Path to model or model name"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate the best model without training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Specific model path to validate (used with --validate_only)",
    )

    # Training parameters
    parser.add_argument(
        "--img_size", type=int, default=1024, help="Image size for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of worker threads for dataloader"
    )

    # Augmentation and training options
    parser.add_argument(
        "--multi_scale",
        action="store_true",
        default=True,
        help="Use multi-scale training",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--mixup", type=float, default=0.2, help="Mixup alpha for augmentation"
    )
    parser.add_argument(
        "--max_det", type=int, default=150, help="Maximum detections per image"
    )
    parser.add_argument("--nms", action="store_true", default=True, help="Use NMS")
    parser.add_argument(
        "--flipud", type=float, default=0.3, help="Probability of flipping up-down"
    )
    parser.add_argument(
        "--cls_loss", type=float, default=1.0, help="Classification loss weight"
    )
    parser.add_argument(
        "--augment", action="store_true", default=True, help="Use augmentation"
    )
    parser.add_argument(
        "--copy_paste", type=float, default=0.1, help="Copy-paste augmentation"
    )
    parser.add_argument(
        "--copy_paste_mode", type=str, default="mixup", help="Copy-paste mode"
    )
    parser.add_argument(
        "--auto_augment", type=str, default="augmix", help="Auto augmentation mode"
    )
    parser.add_argument(
        "--half", action="store_true", default=False, help="Use half precision (FP16)"
    )
    parser.add_argument("--classes", type=int, default=None, help="Train on single class")

    return parser.parse_args()


def main():
    """Main function to run the training process."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse arguments
    args = parse_args()

    # If only validating
    if args.validate_only:
        results = validate_model(args.model_path)
        logging.info(f"Validation results: {results}")
        return

    # Train model
    logging.info(f"Starting training with model: {args.model}")
    model = train_model(args)

    # Validate best model
    results = validate_model()
    logging.info(f"Validation results: {results}")


if __name__ == "__main__":
    main()
