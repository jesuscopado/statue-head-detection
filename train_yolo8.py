import argparse
import os

import wandb
from dotenv import load_dotenv
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


def train_yolov8(
    yaml_path: str = "yolo_config/statues_yolov8.yaml",
    wandb_project_name: str = "statues",
    model_name: str = "yolov8n",
    epochs: int = 100,
    img_size: int = 1280,
    batch_size: int = 32,
    seed: int = 22,
):
    """
    Train a YOLOv8 model with custom configurations.

    Args:
        yaml_path (str): Path to the YAML config file.
        wandb_project_name (str): Wandb project name.
        model_name (str): YOLOv8 model name.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """

    # Init W&B run
    wandb.init(project=wandb_project_name, job_type="training")

    # Init YOLO model with pre-trained weights (COCO)
    model = YOLO(f"{model_name}.pt")

    # Add W&B callback for ultralytics
    add_wandb_callback(
        model,
        enable_model_checkpointing=True,
        enable_train_validation_logging=False,
        enable_validation_logging=False,
    )

    # Train the model
    model.train(
        project=wandb_project_name,
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        seed=seed,
        deterministic=True,
    )

    # Validate the model
    model.val()

    # Finalize W&B run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="yolo_config/statues_yolov8.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="statues",
        help="Wandb project name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolov8n",
        help="YOLOv8 model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1280,
        help="Image size for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed",
    )

    # Load the .env file
    load_dotenv()

    # Login in W&B
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    args = parser.parse_args()
    train_yolov8(
        args.yaml_path,
        args.wandb_project_name,
        args.model_name,
        args.epochs,
        args.img_size,
        args.batch_size,
        args.seed,
    )
