from ultralytics import YOLO
from ultralytics.engine.results import Results
from pathlib import Path
from typing import List, Dict, Any


class MergedYOLOPredictor:
    def __init__(
        self,
        model_paths: List[str],
        class_names_map: Dict[int, str] = {0: "anthracnose", 1: "cssvd", 2: "healthy"},
    ):
        """
        Initializes the predictor with multiple YOLO models.

        Args:
            model_paths (List[str]): A list of paths to the trained YOLO model files.
                                     Each model should correspond to a single class.
            class_names_map (Dict[int, str]): A mapping from the original class index
                                              (e.g., 0, 1, 2) to the class name
                                              (e.g., {0: 'anthracnose', 1: 'cssvd', 2: 'healthy'}).
                                              The order should align with model_paths if cfgs aren't specific.
        """
        if not model_paths:
            raise ValueError("model_paths cannot be empty.")
        self.models = [self.load_model(path) for path in model_paths]
        self.class_names_map = class_names_map
        print(f"Loaded {len(self.models)} models.")
        print(f"Class mapping: {self.class_names_map}")

    def load_model(self, model_path: str):
        """Loads and prepares a single YOLO model for evaluation."""
        model = YOLO(model_path)
        return model  # .eval() is implicitly handled by predict

    def predict(self, images: List[Any], cfgs: List[Dict[str, Any]]):
        """
        Runs prediction using each model on the provided images.

        Args:
            images (List[Any]): A list of images (paths, numpy arrays, PIL images, etc.).
            cfgs (List[Dict[str, Any]]): A list of configuration dictionaries for each model's predict call.
                                         Each cfg MUST include 'classes': [original_class_index]
                                         to link the model back to its original class.

        Returns:
            List[Dict[str, Any]]: Aggregated prediction results per image.
        """
        if len(self.models) != len(cfgs):
            raise ValueError("Number of models and configurations must match.")

        all_results: Dict[int, List[Results]] = {}
        for model, cfg in zip(self.models, cfgs):
            if (
                "classes" not in cfg
                or not isinstance(cfg["classes"], list)
                or len(cfg["classes"]) != 1
            ):
                raise ValueError(
                    "Each cfg must contain 'classes': [original_class_index] (e.g., 'classes': [0])"
                )
            original_class_idx = cfg["classes"][0]
            if original_class_idx not in self.class_names_map:
                raise ValueError(
                    f"Class index {original_class_idx} from cfg not found in class_names_map."
                )

            # Ensure verbose=False unless explicitly requested per model
            predict_cfg = cfg.copy()
            predict_cfg.setdefault("verbose", False)

            results: List[Results] = model.predict(images, **predict_cfg)
            all_results[original_class_idx] = (
                results  # Store results keyed by original index
            )

        return self.aggregate_results(all_results)

    def aggregate_results(
        self, all_results: Dict[int, List[Results]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregates results from multiple single-class models into a per-image format.

        Args:
            all_results (Dict[int, List[Results]]): A dictionary where keys are original class indices
                                                     and values are lists of Results objects (one per image)
                                                     from the corresponding single-class model.

        Returns:
            List[Dict[str, Any]]: A list where each dictionary represents an image and contains:
                                  - 'Image_ID': The filename of the image.
                                  - 'detections': A list of detection dictionaries for that image,
                                                  each with 'bbox', 'class', and 'confidence'.
        """
        if not all_results:
            return []

        # Determine the number of images from the first result list
        first_class_idx = list(all_results.keys())[0]
        num_images = len(all_results[first_class_idx])

        aggregated_data_per_image: List[Dict[str, Any]] = []

        # Iterate through each image index
        for img_idx in range(num_images):
            image_detections = []
            image_id = None

            # Iterate through each model's results (keyed by original class index)
            for original_class_idx, results_list in all_results.items():
                if img_idx >= len(results_list):
                    print(
                        f"Warning: Mismatch in result list length for class {original_class_idx}. Skipping image index {img_idx}."
                    )
                    continue

                result: Results = results_list[img_idx]

                # Store image ID if not already found
                if image_id is None and result.path:
                    image_id = Path(result.path).name

                # Extract detections for the current image from this model
                boxes = result.boxes.xyxy.tolist() if result.boxes else []
                confidences = result.boxes.conf.tolist() if result.boxes else []
                # Note: result.boxes.cls will likely be all 0s for single-class models.
                # We use the original_class_idx to get the correct class name.
                original_class_name = self.class_names_map.get(
                    original_class_idx, f"unknown_class_{original_class_idx}"
                )

                for box, conf in zip(boxes, confidences):
                    image_detections.append(
                        {
                            "bbox": box,
                            "class": original_class_name,  # Use the original class name
                            "confidence": conf,
                        }
                    )

            # Fallback image ID if path wasn't available in results
            if image_id is None:
                image_id = f"image_{img_idx}"

            aggregated_data_per_image.append(
                {"Image_ID": image_id, "detections": image_detections}
            )

        return aggregated_data_per_image
