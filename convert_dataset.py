import os
import yaml
import argparse
import shutil
from pathlib import Path

def get_label_path(img_path_str):
    """Infers label path from image path assuming parallel structure."""
    img_path = Path(img_path_str)
    # Assumes labels are in '../labels/' relative to '../images/'
    label_path = img_path.parent.parent / 'labels' / img_path.name
    if not label_path.exists():
        raise FileNotFoundError(f"Assumed label directory not found: {label_path}")
    return label_path

def convert_labels(original_label_dir, target_label_dir, target_class_idx):
    """
    Copies label files, filtering for a specific class and re-indexing it to 0.
    Only creates a target file if annotations for the target class exist.
    """
    os.makedirs(target_label_dir, exist_ok=True)
    print(f"Processing labels from {original_label_dir} for class index {target_class_idx} -> {target_label_dir}")
    files_created = 0
    for label_file in Path(original_label_dir).glob('*.txt'):
        output_lines = []
        try:
            with open(label_file, 'r') as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        original_idx = int(parts[0])
                        if original_idx == target_class_idx:
                            # Keep the line, change class index to 0
                            output_lines.append(f"0 {' '.join(parts[1:])}\n")
                    except (ValueError, IndexError):
                        print(f"Warning: Skipping malformed line in {label_file}: {line.strip()}")
                        continue # Skip malformed lines

            # Only write the file if there are lines to write
            if output_lines:
                target_file_path = Path(target_label_dir) / label_file.name
                with open(target_file_path, 'w') as f_out:
                    f_out.writelines(output_lines)
                files_created += 1
        except Exception as e:
            print(f"Error processing file {label_file}: {e}")
    print(f"Finished processing. Created {files_created} label files in {target_label_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert multi-class YOLO dataset to single-class datasets.")
    parser.add_argument(
        "--input-yaml",
        type=str,
        default="data.yaml",
        help="Path to the original multi-class data.yaml file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets_single_class",
        help="Base directory to save the new single-class dataset configurations and labels."
    )
    args = parser.parse_args()

    # Load original data configuration
    try:
        with open(args.input_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Input YAML file not found at {args.input_yaml}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {args.input_yaml}: {e}")
        return

    original_classes = data_config.get('names')
    if not original_classes or not isinstance(original_classes, list):
        print("Error: 'names' field missing or invalid in input YAML.")
        return

    try:
        original_train_img = Path(data_config['train']).resolve()
        original_val_img = Path(data_config['val']).resolve()
        original_test_img = None # Path(data_config.get('test', None)) # Test set is optional
        if original_test_img:
            original_test_img = original_test_img.resolve()

        # Infer label directories
        original_train_lbl = get_label_path(original_train_img)
        original_val_lbl = get_label_path(original_val_img)
        original_test_lbl = get_label_path(original_test_img) if original_test_img else None

    except KeyError as e:
        print(f"Error: Missing required key in YAML: {e}")
        return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure label directories exist parallel to image directories (e.g., dataset/labels/train).")
        return
    except Exception as e:
        print(f"An unexpected error occurred while resolving paths: {e}")
        return


    output_base_dir = Path(args.output_dir)
    print(f"Creating single-class datasets in: {output_base_dir.resolve()}")

    new_yaml_paths = []

    for i, class_name in enumerate(original_classes):
        print(f"\nProcessing class: {class_name} (Index: {i})")
        class_output_dir = output_base_dir / class_name
        class_labels_dir = class_output_dir / 'labels'

        # Create new label directories and convert labels
        convert_labels(original_train_lbl, class_labels_dir / 'train', i)
        convert_labels(original_val_lbl, class_labels_dir / 'val', i)
        if original_test_lbl:
            convert_labels(original_test_lbl, class_labels_dir / 'test', i)

        # Create new data.yaml for this class
        new_data_config = {
            'names': [class_name],
            'nc': 1,
            # Important: Point to the *original* image paths
            'train': str(original_train_img),
            'val': str(original_val_img),
            # The 'path' variable helps YOLO find labels relative to images if needed,
            # but often it infers correctly if labels/<split> exists relative to images/<split>
            # Setting it to the new class dir might help if inference fails.
            # Alternatively, leave it out or point to the original dataset root.
            # Let's point it to the new structure containing the labels.
            'path': str(class_output_dir.resolve()),
        }
        if original_test_img:
            new_data_config['test'] = str(original_test_img)

        new_yaml_path = class_output_dir / f"data_{class_name}.yaml"
        try:
            os.makedirs(class_output_dir, exist_ok=True)
            with open(new_yaml_path, 'w') as f:
                yaml.dump(new_data_config, f, default_flow_style=False, sort_keys=False)
            new_yaml_paths.append(str(new_yaml_path.resolve()))
            print(f"Created new YAML config: {new_yaml_path}")
        except Exception as e:
             print(f"Error writing YAML file {new_yaml_path}: {e}")


    print("\nConversion complete.")
    print("New dataset YAML files created:")
    for p in new_yaml_paths:
        print(f"- {p}")
    print(f"\nEach YAML file points to the original images but uses filtered labels located within '{output_base_dir.resolve()}/<class_name>/labels/'.")
    print("You can now train separate models using these YAML files.")

if __name__ == "__main__":
    main()