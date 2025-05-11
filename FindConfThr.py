# %%
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
# Import libraries
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import yaml
from ultralytics.utils.patches import imread
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from torchmetrics.detection import MeanAveragePrecision

# Constants
INPUT_DATA_DIR = Path("dataset")
DATASETS_DIR = Path("dataset")
VAL_IMAGES_DIR = DATASETS_DIR / "images" / "val"
VAL_LABELS_DIR = DATASETS_DIR / "labels" / "val"


# %%
def find_best_model(base_path="zindi_challenge_cacao_stage2/train"):
    """Find the latest trained model and its configuration."""
    latest_run_dir = sorted(
        glob(f"{base_path}*"), key=lambda x: int(x.split("train")[-1])
    )[-1]

    best_path = f"{latest_run_dir}/weights/best.pt"
    best_cfg = f"{latest_run_dir}/args.yaml"

    print(f"Using model weights: {best_path}")
    print(f"Using model config: {best_cfg}")

    return best_path, best_cfg


# %%
def load_image_(filepath):
    """Load an image with proper EXIF orientation."""
    image = Image.open(filepath)
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image


def load_image(filepath):
    """Load a single image."""
    return imread(filepath, cv2.IMREAD_COLOR)


def load_images(filepaths):
    """Load multiple images in parallel."""
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, filepaths))
    return images


# %%
def prepare_model_config(cfg_path, batch_size=16):
    """Prepare model configuration for prediction."""
    with open(cfg_path, "r") as f:
        cfg: dict = yaml.safe_load(f)

    # Set prediction parameters
    cfg["device"] = "cuda"
    cfg["batch"] = batch_size
    cfg["conf"] = 0.0  # Use lowest confidence, filter later
    cfg["verbose"] = False
    cfg["nms"] = True
    cfg["max_det"] = cfg.get("max_det", 100)  # Default if not specified
    cfg["deterministic"] = True
    cfg["iou"] = 0.6
    cfg["agnostic_nms"] = False
    cfg["mode"] = "predict"

    # Remove parameters not needed for prediction
    keys_to_remove = [
        k
        for k in cfg.keys()
        if any(
            x in k
            for x in ["show", "save", "freeze", "plot", "drop", "lr", "mom", "wei"]
        )
        or any(
            k == x
            for x in [
                "epoch",
                "optimizer",
                "project",
                "patience",
                "rect",
                "resume",
                "pretrained",
                "exist_ok",
                "time",
                "overlap_mask",
                "source",
            ]
        )
    ]

    for key in keys_to_remove:
        cfg.pop(key, None)

    return cfg


# %%
def load_and_prepare_model(weights_path, config_path, batch_size=16):
    """Load the YOLO model and prepare its configuration."""
    # Load model
    model = YOLO(weights_path).eval()
    model.fuse()

    # Prepare configuration
    cfg = prepare_model_config(config_path, batch_size)

    return model, cfg


# %%
def predict_images(model: YOLO, image_dir: Path, cfg: dict, batch_size=16):
    """Run prediction on images in the specified directory."""
    # Get image files
    image_files = [i for i in image_dir.glob("*.*") if i.suffix != ".npy"]

    # Initialize results container
    all_data = []

    with torch.no_grad():
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i : i + batch_size]
            batch_paths = [str(img_file) for img_file in batch_files]
            batch_images = load_images(batch_paths)

            # Make predictions
            results = model.predict(batch_images, **cfg)

            # Process results
            for img_file, result in zip(batch_files, results):
                boxes = result.boxes.xyxy.tolist() if result.boxes else []
                classes = result.boxes.cls.tolist() if result.boxes else []
                confidences = result.boxes.conf.tolist() if result.boxes else []
                names = result.names

                if boxes:  # If detections found
                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = box
                        detected_class = names[int(cls)]

                        all_data.append(
                            {
                                "Image_ID": str(img_file),
                                "class": detected_class,
                                "confidence": conf,
                                "ymin": y1,
                                "xmin": x1,
                                "ymax": y2,
                                "xmax": x2,
                            }
                        )
                else:  # No detections
                    all_data.append(
                        {
                            "Image_ID": str(img_file),
                            "class": "None",
                            "confidence": None,
                            "ymin": None,
                            "xmin": None,
                            "ymax": None,
                            "xmax": None,
                        }
                    )

    # Convert to DataFrame and clean up image IDs
    predictions_df = pd.DataFrame(all_data)
    predictions_df.loc[:, "Image_ID"] = predictions_df["Image_ID"].apply(
        lambda x: str(Path(x).stem)
    )

    return predictions_df


# %%
def load_yolo_labels(label_folder):
    """Load YOLO format labels from files."""
    label_data = {}
    label_folder = Path(label_folder)
    paths = [i for i in label_folder.glob("*") if i.suffix != ".npy"]

    for label_file in paths:
        with open(label_file, "r") as file:
            annotations = []
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    annotations.append(
                        {
                            "class_id": int(class_id),
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                        }
                    )
            label_data[label_file.stem] = annotations

    # Convert to DataFrame
    label_df = []
    for image_id, annotations in label_data.items():
        for annotation in annotations:
            label_df.append(
                {
                    "Image_ID": image_id,
                    "class_id": annotation["class_id"],
                    "x_center": annotation["x_center"],
                    "y_center": annotation["y_center"],
                    "width": annotation["width"],
                    "height": annotation["height"],
                }
            )

    return pd.DataFrame(label_df)


# %%
def yolo_to_bbox(image_folder: Path, labels_df: pd.DataFrame):
    """Convert YOLO format labels to bounding box coordinates."""
    image_folder = Path(image_folder)
    converted_bboxes = []

    # Handle images without labels
    paths = [i for i in image_folder.glob("*") if i.suffix != ".npy"]
    for image_file in paths:
        image_id = image_file.stem
        if image_id not in labels_df["Image_ID"].values:
            converted_bboxes.append(
                {
                    "Image_ID": image_id,
                    "class_id": -1,  # No label
                    "xmin": None,
                    "ymin": None,
                    "xmax": None,
                    "ymax": None,
                }
            )

    # Process images with labels
    for _, row in labels_df.iterrows():
        all_ids = [
            i for i in image_folder.glob(f"{row['Image_ID']}*") if i.suffix != ".npy"
        ]
        image_path = all_ids[0] if all_ids else None

        if image_path and image_path.exists():
            img = load_image_(image_path)
            img_width, img_height = img.size

            # Convert normalized coordinates to pixel values
            x_center = row["x_center"] * img_width
            y_center = row["y_center"] * img_height
            width = row["width"] * img_width
            height = row["height"] * img_height

            # Calculate bbox coordinates
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            converted_bboxes.append(
                {
                    "Image_ID": row["Image_ID"],
                    "class_id": row["class_id"],
                    "xmin": x_min,
                    "ymin": y_min,
                    "xmax": x_max,
                    "ymax": y_max,
                }
            )

    return pd.DataFrame(converted_bboxes)


# %%
def load_ground_truth(label_folder, image_folder, class_map=None):
    """Load and prepare ground truth data."""
    # Load class mapping if provided
    if class_map is None:
        train = pd.read_csv(INPUT_DATA_DIR / "Train_df.csv")
        class_map = {
            cls: i for i, cls in enumerate(sorted(train["class"].unique().tolist()))
        }

    # Load or compute labels
    labels_csv_path = DATASETS_DIR / "labels.csv"
    if os.path.exists(labels_csv_path):
        labels = pd.read_csv(labels_csv_path)
    else:
        labels = load_yolo_labels(label_folder)
        labels.to_csv(labels_csv_path, index=False)

    # Convert to bounding box format
    converted_labels_csv_path = DATASETS_DIR / "converted_labels.csv"
    if os.path.exists(converted_labels_csv_path):
        converted_labels = pd.read_csv(converted_labels_csv_path)
    else:
        converted_labels = yolo_to_bbox(image_folder, labels)
        converted_labels.to_csv(converted_labels_csv_path, index=False)

    # Map class IDs to names
    id_class_map = {v: k for k, v in class_map.items()}
    converted_labels["class"] = converted_labels["class_id"].map(id_class_map)

    return converted_labels, class_map, id_class_map


# %%
def convert_df(df: pd.DataFrame):
    """Convert DataFrame to format required for metric computation."""
    df = df.copy().dropna()
    return {
        img_id: {
            "boxes": torch.tensor(
                raw[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
            ),
            "scores": (
                torch.tensor(raw["confidence"].values, dtype=torch.float32)
                if "confidence" in raw.columns
                else None
            ),
            "labels": torch.tensor(raw["class_id"].values, dtype=torch.int32),
        }
        for (img_id,), raw in df.groupby(["Image_ID"])
    }


def default_value():
    """Return default empty tensor values for predictions."""
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int32),
    }


def get_preds_data(preds: pd.DataFrame, thr=None):
    """Filter predictions by confidence threshold and convert to required format."""
    if thr is not None:
        preds = preds[preds["confidence"] >= thr]
    preds = convert_df(preds)
    d = default_value()
    return {i: preds.get(i, d) for i in ground_truth_df["Image_ID"].unique()}


# %%
def calculate_iou_tensor(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    xA = torch.max(box1[0], box2[0])
    yA = torch.max(box1[1], box2[1])
    xB = torch.min(box1[2], box2[2])
    yB = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else torch.tensor(0.0)


def evaluate_detection(
    predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.0
):
    """
    Evaluate detection metrics: Precision, Recall, F1, Accuracy.

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: Threshold for considering a detection as correct
        conf_threshold: Confidence threshold for filtering predictions

    Returns:
        Dictionary with metric scores
    """
    TP = 0
    FP = 0
    FN = 0

    # Ensure predictions and ground_truths are lists
    if isinstance(predictions, dict):
        predictions = list(predictions.values())
    if isinstance(ground_truths, dict):
        ground_truths = list(ground_truths.values())

    for preds, gts in zip(predictions, ground_truths):
        # Sort predictions by confidence score (descending)
        if len(preds["boxes"]) > 0:
            _, sorted_indices = torch.sort(preds["scores"], descending=True)
            pred_boxes = preds["boxes"][sorted_indices]
            pred_labels = preds["labels"][sorted_indices]
            pred_scores = preds["scores"][sorted_indices]
        else:
            pred_boxes = torch.empty((0, 4))
            pred_labels = torch.empty((0,), dtype=torch.int32)
            pred_scores = torch.empty((0,))

        gt_boxes = gts["boxes"]
        gt_labels = gts["labels"]
        matched_gt = set()

        # Match predictions to ground truth
        for i in range(len(pred_boxes)):
            if pred_scores[i] < conf_threshold:
                continue

            pred_box = pred_boxes[i]
            pred_label = pred_labels[i]
            best_iou = iou_threshold
            best_match = None

            # Find best matching GT box
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if pred_label != gt_labels[j]:
                    continue
                iou = calculate_iou_tensor(pred_box, gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_match = j

            if best_match is not None:
                TP += 1
                matched_gt.add(best_match)
            else:
                FP += 1

        # Count unmatched ground truths as false negatives
        FN += len(gt_boxes) - len(matched_gt)

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
    }


# %%
def compute_map(preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]], cfg: dict):
    """
    Compute Mean Average Precision.

    Args:
        preds: List of prediction dicts
        targets: List of ground truth dicts
        cfg: Model configuration with max_det parameter

    Returns:
        mAP results dictionary
    """
    # Initialize the metric
    metric = MeanAveragePrecision(
        iou_thresholds=[0.5],
        max_detection_thresholds=[1, 10, cfg.get("max_det", 100)],
        class_metrics=True,
    )

    # Update metric with predictions and targets
    metric.update(preds, targets)

    # Compute the results
    result = metric.compute()

    return result


# %%
def evaluate_f1_precision_recall(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, class_map: dict, conf_thresholds=None
):
    """
    Evaluate and print F1, Precision, Recall metrics for different confidence thresholds.

    Args:
        predictions_df: DataFrame with predictions
        ground_truth_df: DataFrame with ground truth
        class_map: Dictionary mapping class names to IDs
        conf_thresholds: List of confidence thresholds to evaluate
    """
    # Prepare data
    predictions_df["class_id"] = predictions_df["class"].map(class_map)
    predictions_all_conf = get_preds_data(predictions_df, None)
    ground_truth = convert_df(ground_truth_df)
    ground_truth = {k: ground_truth[k] for k in ground_truth_df["Image_ID"].unique()}

    # Use default thresholds if none provided
    if conf_thresholds is None:
        conf_thresholds = np.linspace(0.05, 0.95, 10)

    # Print header
    print("\n" + "=" * 60)
    print("DETECTION METRICS (F1, PRECISION, RECALL, ACCURACY)".center(60))
    print("=" * 60)
    print(
        f"{'Conf Thr':^10} {'TP':^6} {'FP':^6} {'FN':^6} {'Precision':^10} {'Recall':^10} {'F1 Score':^10} {'Accuracy':^10}"
    )
    print("-" * 60)

    # Evaluate for each threshold
    best_f1 = 0
    best_thr = 0

    for threshold in conf_thresholds:
        scores = evaluate_detection(
            predictions_all_conf,
            ground_truth.values(),
            iou_threshold=0.5,
            conf_threshold=threshold,
        )

        # Track best F1 score
        if scores["F1 Score"] > best_f1:
            best_f1 = scores["F1 Score"]
            best_thr = threshold

        # Print results
        print(
            f"{threshold:^10.2f} {scores['TP']:^6} {scores['FP']:^6} {scores['FN']:^6} "
            f"{scores['Precision']:^10.4f} {scores['Recall']:^10.4f} {scores['F1 Score']:^10.4f} "
            f"{scores['Accuracy']:^10.4f}"
        )

    print("-" * 60)
    print(f"Best F1 Score: {best_f1:.4f} at confidence threshold: {best_thr:.2f}")
    print("=" * 60)

    return best_thr, best_f1


# %%
def evaluate_map(
    predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, class_map: dict, id_class_map: dict, cfg: dict, conf_thresholds: list[float]=None
):
    """
    Evaluate and print mAP metrics for different confidence thresholds.

    Args:
        predictions_df: DataFrame with predictions
        ground_truth_df: DataFrame with ground truth
        class_map: Dictionary mapping class names to IDs
        id_class_map: Dictionary mapping class IDs to names
        cfg: Model configuration
        conf_thresholds: List of confidence thresholds to evaluate
    """
    # Prepare data
    predictions_df["class_id"] = predictions_df["class"].map(class_map)

    # Convert to format for mAP calculation
    ground_truth_tensor = convert_df(ground_truth_df)
    ground_truth_tensor = {
        k: ground_truth_tensor[k] for k in ground_truth_df["Image_ID"].unique()
    }
    targets = list(ground_truth_tensor.values())

    # Use default thresholds if none provided
    if conf_thresholds is None:
        conf_thresholds = [0, 0.01, 0.1, 0.25, 0.5, 0.75]

    # Print header
    print("\n" + "=" * 80)
    print("MEAN AVERAGE PRECISION (mAP)".center(80))
    print("=" * 80)
    print(
        f"{'Conf Thr':^10} {'mAP@0.5 (maxDet=1)':^20} {'mAP@0.5 (maxDet=10)':^20} {'mAP@0.5 (maxDet='+str(cfg.get('max_det',100))+')':^20}"
    )
    print("-" * 80)

    # Evaluate for each threshold
    best_map = 0
    best_thr = 0

    for threshold in conf_thresholds:
        # Filter predictions by confidence threshold
        preds_filtered = get_preds_data(predictions_df, threshold)
        current_preds_list = list(preds_filtered.values())

        # Skip if no predictions at this threshold
        if not any(len(p["scores"]) > 0 for p in current_preds_list):
            print(f"{threshold:^10.2f} {'No predictions':^60}")
            continue

        # Compute mAP
        results = compute_map(current_preds_list, targets, cfg)

        # Track best mAP score (using standard mAP@0.5 with maxDet=100)
        map_score = results["map_50_det_100"].item()
        if map_score > best_map:
            best_map = map_score
            best_thr = threshold

        # Print results
        print(
            f"{threshold:^10.2f} {results['map_50'].item():^20.4f} {results['map_50_det_10'].item():^20.4f} {map_score:^20.4f}"
        )

        # Print per-class AP if available and not empty
        if (
            "map_per_class" in results
            and results["map_per_class"] is not None
            and results["map_per_class"].numel() > 0
        ):
            print("\n  Per-class AP@0.5:")
            class_aps = {
                int(cls_id.item()): ap.item()
                for cls_id, ap in zip(results["classes"], results["map_per_class"])
            }
            for cls_id, ap_val in class_aps.items():
                class_name = id_class_map.get(cls_id, f"Unknown ClassID {cls_id}")
                print(f"    {class_name}: {ap_val:.4f}")
            print("")

    print("-" * 80)
    print(f"Best mAP@0.5: {best_map:.4f} at confidence threshold: {best_thr:.2f}")
    print("=" * 80)

    return best_thr, best_map


# %%
def main():
    """Run the evaluation pipeline."""
    # Find best model
    weights_path, config_path = find_best_model()

    # Load model and prepare configuration
    model, cfg = load_and_prepare_model(weights_path, config_path)

    # Load ground truth data
    ground_truth_df, class_map, id_class_map = load_ground_truth(
        VAL_LABELS_DIR, VAL_IMAGES_DIR
    )

    # Predict on validation images
    print("\nRunning predictions on validation images...")
    predictions_df = predict_images(model, VAL_IMAGES_DIR, cfg)
    print(
        f"Generated {len(predictions_df)} predictions across {predictions_df['Image_ID'].nunique()} images"
    )

    # Confidence thresholds for evaluation
    f1_thresholds = np.linspace(0.05, 0.95, 10)
    map_thresholds = [0.01, 0.1, 0.25, 0.5, 0.75]

    # Evaluate F1, Precision, Recall
    best_f1_thr, best_f1 = evaluate_f1_precision_recall(
        predictions_df, ground_truth_df, class_map, f1_thresholds
    )

    # Evaluate mAP
    best_map_thr, best_map = evaluate_map(
        predictions_df, ground_truth_df, class_map, id_class_map, cfg, map_thresholds
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY".center(60))
    print("=" * 60)
    print(f"Best F1 Score: {best_f1:.4f} at confidence threshold: {best_f1_thr:.2f}")
    print(f"Best mAP@0.5: {best_map:.4f} at confidence threshold: {best_map_thr:.2f}")
    print("=" * 60)


# %%
# Global variables needed for some functions
ground_truth_df: pd.DataFrame = None

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()
