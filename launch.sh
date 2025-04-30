#!/bin/bash

# Define color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== YOLO Model Training Script =====${NC}"

# Function to display help
function show_help {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./launch.sh [option]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  train          - Train a new model with default settings"
    echo "  train-small    - Train with smaller image size and batch size (for limited resources)"
    echo "  train-yolo11s  - Train using YOLOv11 small model"
    echo "  train-yolo11n  - Train using YOLOv11 nano model"
    echo "  train-yolo11m  - Train using YOLOv11 medium model"
    echo "  train-yolo11l  - Train using YOLOv11 large model"
    echo "  train-yolo11x  - Train using YOLOv11 extra large model"
    echo "  resume         - Resume training from last checkpoint"
    echo "  validate       - Only validate the best model without training"
    echo "  help           - Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./launch.sh train          # Start training with default settings"
    echo "  ./launch.sh train-small    # Train with smaller image size"
    echo "  ./launch.sh train-yolo11l  # Train with YOLOv11 large model"
    echo "  ./launch.sh resume         # Resume from last checkpoint"
    echo ""
}

# Function to execute training with parameters
function train_model {
    echo -e "${GREEN}Starting training with parameters:${NC}"
    echo "$1"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Generate timestamp for unique log file names
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="logs/yolo_train_${timestamp}.log"
    
    echo -e "${YELLOW}Training will continue in background. Output redirected to:${NC}"
    echo -e "${GREEN}${log_file}${NC}"
    echo -e "${YELLOW}You can monitor progress with:${NC} tail -f ${log_file}"
    
    # Run the training in the background with nohup
    nohup python TrainYoloModel.py $1 > ${log_file} 2>&1 &
    
    # Save the process ID for reference
    echo $! > logs/last_training_pid.txt
    echo -e "${GREEN}Process started with PID:${NC} $!"
}

# Process command line arguments
case "$1" in
    train)
        train_model "--model yolov8l.pt --batch_size 16 --img_size 1024 --epochs 200 --patience 20 --dropout 0.3 --mixup 0.2"
        ;;
    train-small)
        train_model "--model yolov8n.pt --batch_size 16 --img_size 1024 --epochs 100 --patience 15 --dropout 0.3 --mixup 0.2"
        ;;
    train-yolo11s)
        train_model "--model yolo11s.pt --batch_size 16 --img_size 1024 --epochs 150 --patience 20 --dropout 0.3 --mixup 0.2"
        ;;
    train-yolo11n)
        train_model "--model yolo11n.pt --batch_size 16 --img_size 1024 --epochs 150 --patience 20 --dropout 0.3 --mixup 0.2"
        ;;
    train-yolo11m)
        train_model "--model yolo11m.pt --batch_size 16 --img_size 1024 --epochs 200 --patience 20 --dropout 0.3 --mixup 0.2"
        ;;
    train-yolo11l)
        train_model "--model yolo11l.pt --batch_size 16 --img_size 1024 --epochs 200 --patience 20 --dropout 0.3 --mixup 0.2"
        ;;
    train-yolo11x)
        train_model "--model yolo11x.pt --batch_size 16 --img_size 1024 --epochs 250 --patience 25 --dropout 0.3 --mixup 0.2"
        ;;
    resume)
        train_model "--model runs/detect/train/weights/last.pt --resume"
        ;;
    validate)
        train_model "--validate_only"
        ;;
    validate-specific)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Please provide a model path to validate${NC}"
            echo "Example: ./launch.sh validate-specific runs/detect/train5/weights/best.pt"
            exit 1
        fi
        train_model "--validate_only --model_path $2"
        ;;
    help|*)
        show_help
        ;;
esac

echo -e "${GREEN}===== Script completed =====${NC}"
