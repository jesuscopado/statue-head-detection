import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = "model_weights/best_yolov8n.pt"

# Setting page layout
st.set_page_config(
    page_title="Statue Head Detection using YOLOv8",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Creating sidebar
with st.sidebar:
    st.header("Image Upload")
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    # Model confidence
    confidence = float(st.slider("Select Model Confidence", 25, 100, 50)) / 100

# Creating main page heading
st.title("Statue Head Detection using YOLOv8")

uploaded_image = None
# Adding image to the first column if image is uploaded
if source_img:
    # Opening the uploaded image
    uploaded_image = Image.open(source_img)

model = None
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button("Detect Statue Heads 🗿"):
    try:
        prediction = model.predict(uploaded_image, conf=confidence)[0]
        res_plotted = prediction.plot()[:, :, ::-1]
        st.image(res_plotted, caption="Detected Image", use_column_width=True)
        with st.expander("Detection Results"):
            num_boxes = prediction.boxes.shape[0]
            for i in range(num_boxes):
                x1, y1, x2, y2 = map(
                    lambda v: int(round(v.item())), prediction.boxes.xyxy[i]
                )
                cls_id = int(prediction.boxes.cls[i].item())
                box_dict = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class": cls_id,
                }
                st.write(box_dict)

    except NameError as name_error_exception:
        st.write("No image is uploaded yet!")
