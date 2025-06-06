# Import libraries
import json
import yaml
import os

os.environ["WANDB_PROJECT"] = "zindi_challenge_cacao"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "none"

# import wandb
# wandb.login(key=os.environ["WANDB_API_KEY"], host="https://api.wandb.ai")

import logging
import argparse
import yolo_balancing  # noqa: F401
from ultralytics import YOLO, YOLOE
import torch
from glob import glob


def train_model(args, project_name):
    """Train YOLO model with provided arguments for a specific stage."""
    # Load model
    model = (YOLO if "yoloe" not in args.model else YOLOE)(args.model)

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
        config: dict = (
            json.load(open(args.config, "r"))
            if args.config.endswith(".json")
            else yaml.safe_load(open(args.config, "r"))
        )
        config.pop("data", None)
        config.pop("model", None)

    additional_args = {
        "multi_scale": args.multi_scale,
        "dropout": args.dropout,
        "mixup": args.mixup,
        "nms": args.nms,
        "flipud": args.flipud,
        "cls": args.cls_loss,
        "augment": args.augment,
        "copy_paste": args.copy_paste,
        "copy_paste_mode": args.copy_paste_mode,
        "auto_augment": args.auto_augment,
        **config,
        "half": args.half,
        "iou": args.iou,
        "data": args.data_yaml,
        "epochs": args.epochs,
        "imgsz": args.img_size,
        "batch": args.batch_size,
        "patience": args.patience,
        "device": device,
        "max_det": args.max_det,
        "workers": args.workers if args.workers else args.batch_size // 2,
        # "cache": True,
        "deterministic": False, # Set to True for reproducibility
        "time": 4.5, # I have a smaller GPU, so 4.5 is okay
    }

    # Stage-specific arguments
    if args.stage == 1:
        additional_args["single_cls"] = True
        logging.info("Running Stage 1: Single-class backbone training.")
    elif args.stage == 2:
        if args.freeze_layers is not None:
            additional_args["freeze"] = args.freeze_layers
        logging.info(
            f"Running Stage 2: Multi-class head fine-tuning. Freezing first {args.freeze_layers} layers."
        )
        additional_args["single_cls"] = False
        if args.classes is not None:
            additional_args["classes"] = [args.classes]
    else:
        raise ValueError("Invalid stage specified.")

    print(
        f"Training with the following parameters:\n {json.dumps(additional_args, indent=2)}"
    )

    additional_args["project"] = project_name
    additional_args["resume"] = args.resume

    # Train the model
    model.train(**additional_args)

    return model


def validate_model(args, project_name, model_path=None):
    """Validate the best model from a specific project/stage."""
    if model_path is None:
        try:
            # Find the latest training run within the project directory
            latest_run_dir = sorted(glob(f"{project_name}/train*"))[-1]
            best_model = os.path.join(latest_run_dir, "weights/best.pt")
        except IndexError:
            logging.error(f"Could not find training runs in project '{project_name}'.")
            return None
    else:
        best_model = model_path

    if not os.path.exists(best_model):
        logging.error(f"Model path does not exist: {best_model}")
        return None

    print(f"Validating model: {best_model}")
    model = YOLO(best_model)
    results = model.val(data=args.data_yaml)
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a YOLO model in two stages: backbone pretraining and head fine-tuning."
    )

    # Stage control
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Training stage: 1 for single-class backbone, 2 for multi-class head fine-tuning.",
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
        "--model",
        type=str,
        required=True,
        help="Path to model (e.g., yolov8l.pt for stage 1, stage1_best.pt for stage 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint in the stage's project",
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
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=None,
        help="Number of layers to freeze (from start) during stage 2 fine-tuning.",
    )

    # Augmentation and training options
    parser.add_argument(
        "--multi_scale",
        action="store_true",
        default=False,
        help="Use multi-scale training",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--iou", type=float, default=0.6, help="IOU threshold for NMS")
    parser.add_argument(
        "--mixup", type=float, default=0.2, help="Mixup alpha for augmentation"
    )
    parser.add_argument(
        "--max_det", type=int, default=300, help="Maximum detections per image"
    )
    parser.add_argument("--nms", action="store_true", default=False, help="Use NMS")
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
    parser.add_argument(
        "--classes", type=int, default=None, help="Train on single class"
    )

    return parser.parse_args()


def main(args):
    """Main function to run the training process."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Define project name based on stage
    project_base_name = "zindi_challenge_cacao"
    project_name = f"{project_base_name}_stage{args.stage}"
    os.environ["WANDB_PROJECT"] = project_name

    if args.stage == 1 and not args.model.endswith(".pt"):
        logging.info(f"Starting stage 1 training from base model: {args.model}")

    # Train model for the specified stage
    logging.info(f"Starting training stage {args.stage} with model: {args.model}")
    train_model(args, project_name)

    # Validate best model from the current stage
    logging.info(f"Validating best model from stage {args.stage}...")
    results = validate_model(args, project_name)
    if results:
        logging.info(f"Validation results for stage {args.stage}: {results.maps}")
    else:
        logging.error(f"Validation failed for stage {args.stage}.")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    main(args)
