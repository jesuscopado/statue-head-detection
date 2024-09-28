import argparse
import os

import pandas as pd
from ultralytics import YOLO


def run_solution(
    weights_path: str,
    image_folder: str,
    conf_threshold: float,
    save_images: bool = True,
    output_csv: str = "results.csv",
):
    """
    Run YOLOv8 inference on a set of images and save the results to a CSV file.

    Args:
        weights_path (str): Path to YOLO model weights.
        image_folder (str): Folder containing input images.
        conf_threshold (float): Confidence threshold for the detections.
        save_images (bool, optional): Whether to save images with predicted bounding boxes.
        output_csv (str, optional): Path to the CSV file to save the results.

    Returns:
        None
    """

    # Initialize the trained YOLO model
    model = YOLO(weights_path)

    # Get a list of image files in the specified folder
    image_filenames = [
        f
        for f in os.listdir(image_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG"))
    ]

    # Initialize a list to store results as dictionaries
    results_list = []

    # Process each image
    for image_filename in image_filenames:
        # Run inference on the image
        img_path = os.path.join(image_folder, image_filename)
        prediction = model(
            img_path,
            conf=conf_threshold,
            save=save_images,
        )[0]

        # Check if any objects are detected
        num_boxes = prediction.boxes.shape[0]

        if num_boxes == 0:
            # No heads detected, add a row with default values
            results_list.append(
                {
                    "image name": image_filename,
                    "x1": 0,
                    "y1": 0,
                    "x2": 1,
                    "y2": 1,
                    "class": 0,
                }
            )

        elif num_boxes == 1:
            # Extract coordinates of the only detected head and determine the class
            x1, y1, x2, y2 = map(
                lambda v: int(round(v.item())), prediction.boxes.xyxy[0]
            )
            class_id = int(prediction.boxes.cls[0].item())

            # Add the result to the list
            results_list.append(
                {
                    "image name": image_filename,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class": class_id,
                }
            )

        else:
            # For multiple detections, identify Lenin's or Ataturk’s head or find the largest box
            detected_heads = []
            for i in range(num_boxes):
                x1, y1, x2, y2 = map(
                    lambda v: int(round(v.item())), prediction.boxes.xyxy[i]
                )
                cls_id = int(prediction.boxes.cls[i].item())
                head_bbox = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class": cls_id,
                    "area": (x2 - x1) * (y2 - y1),
                }
                detected_heads.append(head_bbox)

            # Identify statues of Lenin or Ataturk
            identified_statues = [
                head for head in detected_heads if head["class"] in [1, 2]
            ]

            # Determine which head to print
            if len(identified_statues) == 1:
                # If there's exactly one Lenin or Ataturk head, use that
                selected_head = identified_statues[0]
            else:
                # If none or more than one, find the largest head
                selected_head = max(detected_heads, key=lambda h: h["area"])

            # Append the selected head to the results list
            results_list.append(
                {
                    "image name": image_filename,
                    "x1": selected_head["x1"],
                    "y1": selected_head["y1"],
                    "x2": selected_head["x2"],
                    "y2": selected_head["y2"],
                    "class": selected_head["class"],
                }
            )

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(output_csv, index=False, sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on a set of images"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="model_weights/best_yolov8n.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="datasets/statues-dataset-processed/images/test",
        help="Folder containing input images",
    )
    # Add a new argument for the confidence threshold with a default value (e.g., 0.5)
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for the detections",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save images with predicted bounding boxes",
    )
    args = parser.parse_args()
    run_solution(
        args.weights_path, args.image_folder, args.conf_threshold, args.save_images
    )
