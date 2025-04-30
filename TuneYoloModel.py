import argparse
import torch
from ultralytics import YOLO
import yaml
import os
from ray import tune  # Add Ray Tune import

print(os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description="Tune YOLO model hyperparameters")
    parser.add_argument(
        "--model", type=str, default="yolo11l.pt", help="Initial model to tune"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/data/home/eak/learning/nganga_ai/AminiCocoa/Amini-Cocoa-Contamination-Challenge/data.yaml",
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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Define hyperparameter search space with proper Ray Tune format
    search_space = {
        # Learning rate parameters
        "lr0": tune.loguniform(1e-5, 5e-2),  # Initial learning rate
        "lrf": tune.loguniform(0.001, 0.1),  # Final learning rate factor
        # Optimizer parameters
        "momentum": tune.uniform(0.6, 0.99),  # SGD momentum/Adam beta1
        "weight_decay": tune.loguniform(1e-8, 0.001),  # Optimizer weight decay
        # Warmup parameters
        "warmup_epochs": tune.uniform(0.0, 5.0),  # Warmup epochs
        "warmup_momentum": tune.uniform(0.0, 0.95),  # Warmup momentum
        # Loss function coefficients
        "box": tune.uniform(5.0, 20.0),  # Box loss gain
        "cls": tune.uniform(0.5, 4.0),  # Class loss gain
        "dfl": tune.uniform(0.5, 2.0),  # DFL loss gain
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
        "flipud": tune.uniform(0.0, 0.5),  # Flip up-down probability
        "fliplr": tune.uniform(0.0, 0.5),  # Flip left-right probability
        # Mosaic and mix augmentations
        "mosaic": tune.uniform(0.0, 1.0),  # Mosaic augmentation probability
        "mixup": tune.loguniform(1e-5, 1.0),  # Mixup augmentation probability
        "copy_paste": tune.uniform(0.0, 0.5),  # Copy-paste augmentation probability
        # Batch parameters
        # "batch": tune.choice([8, 16, 32, 64]),  # Batch size
        # Optimizer parameters
        # "optimizer": tune.choice(["SGD", "Adam", "AdamW"]),  # Optimizer options
        # augmentation parameters
        "auto_augment": tune.choice(["randaugment", "autoaugment", "augmix"]),
        "copy_paste_mode": tune.choice(["mixup", "flip"]),
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
        save=True,
        imgsz=1024,
        batch=16,
        device = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else "cpu",
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
