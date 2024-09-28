import os
from typing import Dict, Tuple, Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Base directory
BASE_DIR = "datasets/statues-train"

# Ouput directory where the processed dataset will be saved (YOLO format)
OUTPUT_DIR = "datasets/statues-dataset-processed"

# Directories for each class
CLASS_DIRECTORIES = {
    "lenin": os.path.join(BASE_DIR, "statues-lenin"),
    "ataturk": os.path.join(BASE_DIR, "statues-ataturk"),
    "other": os.path.join(BASE_DIR, "statues-other"),
}


def fix_labels(df: pd.DataFrame, class_directories: Dict[str, str]) -> pd.DataFrame:
    """
    Adjust the labels in the dataframe by scaling the bounding boxes and
    handling potential image rotations.

    Args:
        df: The dataframe containing the labels.
        class_directories: A dictionary mapping class names to their respective directories.

    Returns:
        A new dataframe with adjusted bounding boxes and a new 'rotated' column.
    """
    fixed_df = df.copy()
    fixed_df["rotated"] = False

    for index, row in fixed_df.iterrows():
        image_path = find_image_path(row["filename"], row["class"], class_directories)
        if image_path is None:
            print(f"Image {row['filename']} not found in any class directory.")
            continue

        with Image.open(image_path) as img:
            rotated = (row["width"] > row["height"]) != (img.width > img.height)
            fixed_df.at[index, "rotated"] = rotated
            scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax = scale_bounding_boxes(
                row, img, rotated
            )
            fixed_df.loc[index, "xmin"] = scaled_xmin
            fixed_df.loc[index, "ymin"] = scaled_ymin
            fixed_df.loc[index, "xmax"] = scaled_xmax
            fixed_df.loc[index, "ymax"] = scaled_ymax

    return fixed_df


def scale_bounding_boxes(
    row: pd.Series, img: Image, rotated: bool
) -> Tuple[int, int, int, int]:
    """
    Scale bounding box coordinates based on the actual size of the image.

    Args:
        row: A series from the dataframe containing bounding box coordinates and image dimensions.
        img: The image corresponding to the bounding box.
        rotated: A boolean indicating if the image is rotated.

    Returns:
        A tuple of scaled bounding box coordinates (xmin, ymin, xmax, ymax).
    """
    img_width, img_height = (
        (img.height, img.width) if rotated else (img.width, img.height)
    )
    x_scale = img_width / row["width"]
    y_scale = img_height / row["height"]
    scaled_xmin = round(row["xmin"] * x_scale)
    scaled_xmax = round(row["xmax"] * x_scale)
    scaled_ymin = round(row["ymin"] * y_scale)
    scaled_ymax = round(row["ymax"] * y_scale)
    return scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax


def find_image_path(
    filename: str, class_name: str, class_directories: Dict[str, str]
) -> Optional[str]:
    """
    Find the path of an image given its filename and class.

    Args:
        filename: The name of the file to find.
        class_name: The class of the image.
        class_directories: A dictionary mapping class names to their respective directories.

    Returns:
        The path to the image if found, otherwise None.
    """
    preferred_path = os.path.join(class_directories.get(class_name, ""), filename)
    if os.path.isfile(preferred_path):
        return preferred_path
    for directory in class_directories.values():
        potential_path = os.path.join(directory, filename)
        if os.path.isfile(potential_path):
            return potential_path
    return None


def unify_labels_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two dataframes with label data into one unified dataframe.

    Args:
        df1: The first dataframe.
        df2: The second dataframe.

    Returns:
        A unified dataframe containing the label data from both input dataframes.
    """
    df1 = df1.drop(["width", "height"], axis=1)
    df2["rotated"] = False  # Align df2 structure with df1
    return pd.concat([df1, df2], ignore_index=True)


def process_dataset_yolo(labels_df_unified: pd.DataFrame) -> None:
    """
    Process the unified label dataframe to prepare the dataset in YOLO format.

    Args:
        labels_df_unified: The unified dataframe with label data.

    Returns:
        None
    """
    # Seed and split ratio setup
    seed = 22
    split_ratio = (0.9, 0.05, 0.05)  # Split ratio for train/val/test

    # Create output directories for images and labels
    for file_type in ["images", "labels"]:
        for split_name in ["train", "val", "test"]:
            os.makedirs(os.path.join(OUTPUT_DIR, file_type, split_name), exist_ok=True)

    # Split data into train, validation, and test sets
    unique_filenames = labels_df_unified["filename"].unique()
    train_filenames, test_filenames = train_test_split(
        unique_filenames, test_size=split_ratio[2], random_state=seed
    )
    train_filenames, val_filenames = train_test_split(
        train_filenames,
        test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
        random_state=seed,
    )

    # Process each set of filenames
    process_data(train_filenames, "train", labels_df_unified)
    process_data(val_filenames, "val", labels_df_unified)
    process_data(test_filenames, "test", labels_df_unified)


def process_data(
    filenames: list, split_name: str, labels_df_unified: pd.DataFrame
) -> None:
    """
    Process a set of filenames to prepare the data in YOLO format for a specific data split (train/val/test).

    Args:
        filenames: The list of filenames to process.
        split_name: The name of the data split ('train', 'val', or 'test').
        labels_df_unified: The unified dataframe with label data.

    Returns:
        None
    """
    print(f"Processing {split_name} split...")
    for filename in tqdm(filenames):
        image_labels = labels_df_unified[labels_df_unified["filename"] == filename]
        if image_labels.empty:
            continue

        process_image_and_labels(filename, image_labels, split_name)


def process_image_and_labels(
    filename: str, image_labels: pd.DataFrame, split_name: str
) -> None:
    """
    Process a single image and its corresponding labels.

    Args:
        filename: The filename of the image to process.
        image_labels: A dataframe containing the labels for the image.
        split_name: The name of the data split ('train', 'val', or 'test').

    Returns:
        None
    """
    first_row = image_labels.iloc[0]
    src_img_path = find_image_path(filename, first_row["class"], CLASS_DIRECTORIES)

    if not src_img_path or not os.path.exists(src_img_path):
        print(f"{filename} not found.")
        return

    img = Image.open(src_img_path)
    if first_row["rotated"]:
        img = img.rotate(-90, expand=True)  # Rotate 90 degrees counter-clockwise

    dst_img_path = os.path.join(OUTPUT_DIR, "images", split_name, filename)
    img.save(dst_img_path)  # Save image to respective directory

    write_labels(filename, image_labels, img.size, split_name, OUTPUT_DIR)


def write_labels(
    filename: str,
    image_labels: pd.DataFrame,
    img_size: Tuple[int, int],
    split_name: str,
    OUTPUT_DIR: str,
) -> None:
    """
    Write the YOLO-formatted labels for a single image to a .txt file.

    Args:
        filename: The filename of the image.
        image_labels: A dataframe containing the labels for the image.
        img_size: A tuple containing the size (width, height) of the image.
        split_name: The name of the data split ('train', 'val', or 'test').
        OUTPUT_DIR: The base directory where the processed dataset will be saved.

    Returns:
        None
    """
    img_width, img_height = img_size
    labels_list = []

    for _, row in image_labels.iterrows():
        # Create label in YOLO format and add to the list
        x_center = (row["xmin"] + row["xmax"]) / 2 / img_width
        y_center = (row["ymin"] + row["ymax"]) / 2 / img_height
        width = (row["xmax"] - row["xmin"]) / img_width
        height = (row["ymax"] - row["ymin"]) / img_height
        class_id = (
            1 if row["class"] == "lenin" else 2 if row["class"] == "ataturk" else 0
        )
        label = f"{class_id} {x_center} {y_center} {width} {height}\n"
        labels_list.append(label)

    # Save labels as a .txt file
    base_filename, _ = os.path.splitext(filename)
    label_file = base_filename + ".txt"
    label_path = os.path.join(OUTPUT_DIR, "labels", split_name, label_file)
    with open(label_path, "w") as file:
        file.writelines(labels_list)


def correct_dataset_issues(labels_df: pd.DataFrame) -> None:
    """
    Correct specific issues in the label dataframe, such as class naming and file renaming.

    Args:
        labels_df: The dataframe containing label data to be corrected.

    Returns:
        None
    """
    # Correct class naming issues
    labels_df.loc[labels_df["class"] == "2", "class"] = "ataturk"

    # Rename files if necessary and update the DataFrame accordingly
    old_name = "Moscow_2017_1408.jpg"
    new_name = "Moscow_2017_1408_0.jpg"
    class_name = "lenin"
    old_path = os.path.join(CLASS_DIRECTORIES[class_name], old_name)
    new_path = os.path.join(CLASS_DIRECTORIES[class_name], new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    labels_df.loc[
        (labels_df["filename"] == old_name) & (labels_df["class"] == class_name),
        "filename",
    ] = new_name


def prepare_dataset() -> None:
    """
    Prepare the dataset by fixing labels, unifying dataframes, correcting specific issues,
    and processing the dataset for YOLO training.
    """
    # Paths to label files and read label data
    labels_path1 = os.path.join(BASE_DIR, "statues_labels.csv")
    labels_path2 = os.path.join(BASE_DIR, "statues_labels2.csv")
    labels_df1 = pd.read_csv(labels_path1)
    labels_df2 = pd.read_csv(
        labels_path2,
        sep=";",
        names=["filename", "xmin", "ymin", "xmax", "ymax", "class"],
    )

    labels_df1_fixed = fix_labels(labels_df1, CLASS_DIRECTORIES)
    labels_df_unified = unify_labels_df(labels_df1_fixed, labels_df2)

    # Correct specific label issues
    correct_dataset_issues(labels_df_unified)

    # Process dataset for YOLO
    process_dataset_yolo(labels_df_unified)


if __name__ == "__main__":
    prepare_dataset()
