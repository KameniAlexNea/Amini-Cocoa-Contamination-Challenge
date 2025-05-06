import os
tune_name = "zindi_challenge_cacao_tune_last"
os.environ["WANDB_PROJECT"] = tune_name
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "none"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import torch
from ray.train import RunConfig # noqa
from ultralytics import YOLO
import yaml
from ray import tune  # Add Ray Tune import
import random
import numpy as np


def set_seed(seed):
    """Set the seed for random number generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
print(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser(description="Tune YOLO model hyperparameters")
    parser.add_argument(
        "--model", type=str, default="yolo11x.pt", help="Initial model to tune"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/data/home/eak/learning/nganga_ai/AminiCocoa/Amini-Cocoa-Contamination-Challenge/dataset_subset/data_subset.yaml",
        help="Path to data configuration yaml file",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for each trial"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of tuning iterations"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume previous tuning run"
    )
    parser.add_argument(
        "--name", type=str, default="tuning_run", help="Name of the tuning run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for tuning"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Define hyperparameter search space with proper Ray Tune format
    search_space = {
        # Learning rate parameters
        # "lr0": tune.loguniform(1e-5, 5e-2),  # Initial learning rate
        # "lrf": tune.loguniform(0.001, 0.1),  # Final learning rate factor
        # Optimizer parameters
        # "momentum": tune.uniform(0.6, 0.99),  # SGD momentum/Adam beta1
        # "weight_decay": tune.loguniform(1e-8, 0.001),  # Optimizer weight decay
        # Warmup parameters
        # "warmup_epochs": tune.uniform(0.0, 5.0),  # Warmup epochs
        # "warmup_momentum": tune.uniform(0.0, 0.95),  # Warmup momentum
        # Loss function coefficients
        "box": tune.uniform(5.0, 20.0),  # Box loss gain
        "cls": tune.uniform(0.5, 4.0),  # Class loss gain
        "dfl": tune.uniform(0.5, 3.0),  # DFL loss gain
        # HSV augmentation
        "hsv_h": tune.uniform(0.0, 0.1),  # HSV hue augmentation
        "hsv_s": tune.uniform(0.0, 0.9),  # HSV saturation augmentation
        "hsv_v": tune.uniform(0.0, 0.9),  # HSV value augmentation
        # Geometric augmentations
        "degrees": tune.uniform(0.0, 45.0),  # Image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.2),  # Image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # Image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # Image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # Image perspective (+/- fraction)
        # Flip augmentations
        "flipud": tune.uniform(0.0, 0.2),  # Flip up-down probability
        "fliplr": tune.uniform(0.0, 0.2),  # Flip left-right probability
        # Mosaic and mix augmentations
        "mosaic": tune.uniform(0.0, 1.0),  # Mosaic augmentation probability
        "mixup": tune.loguniform(1e-5, 1.0),  # Mixup augmentation probability
        "copy_paste": tune.loguniform(1e-5, 0.2),  # Copy-paste augmentation probability
        # IOU parameters
        "iou": tune.uniform(0.4, 0.9),
        "nms": tune.choice([True]),  # Non-maximum suppression , False
        "agnostic_nms": tune.choice([True]),  # Agnostic NMS , False
        "cos_lr": tune.choice([True]),  # Cosine learning rate scheduler , False
        # augmentation parameters
        "auto_augment": tune.choice(["augmix"]), # "randaugment", "autoaugment", 
        "copy_paste_mode": tune.choice(["mixup"]), # , "flip"
        "amp": tune.choice([True]),
        "half": tune.choice([True]),
        "dropout": tune.uniform(0.0, 0.4),  # Dropout rate
    }

    print(
        f"Starting hyperparameter tuning with {args.iterations} iterations for {args.epochs} epochs each"
    )
    print(f"Data config: {args.data}")
    print(f"Run name: {args.name}")
    print(f"Resume previous run: {args.resume}")

    # Run hyperparameter tuning
    results = model.tune(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        space=search_space,
        use_ray=True,
        name=args.name,
        resume=args.resume,
        plots=True,
        save=False,
        imgsz=1024,
        batch=args.batch_size,
        device="cuda:0", # list(range(torch.cuda.device_count())),
        gpu_per_trial=1,
        workers=4,
        # tune_args=tune_kwargs,
        project=tune_name,
        seed=42,
    )

    # Print best hyperparameters
    print("\nBest hyperparameters found:")
    print(yaml.dump(results))

    # Save best hyperparameters to file
    output_dir = os.path.join("runs", "tune", args.name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "best_hyperparameters.yaml"), "w") as f:
        yaml.dump(results, f)

    print(
        f"Best hyperparameters saved to {os.path.join(output_dir, 'best_hyperparameters.yaml')}"
    )


if __name__ == "__main__":
    main()
