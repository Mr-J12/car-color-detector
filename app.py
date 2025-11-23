import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import io

@st.cache_resource
def load_model(path='yolov8n.pt'):
    return YOLO(path)


def get_car_color(car_image, lower_blue=np.array([100, 50, 50]), upper_blue=np.array([140, 255, 255]), threshold=0.15):
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_pixel_count = int(cv2.countNonZero(mask))
    total_pixel_count = car_image.shape[0] * car_image.shape[1]
    if total_pixel_count == 0:
        return 'other'
    blue_ratio = blue_pixel_count / total_pixel_count
    return 'blue' if blue_ratio > threshold else 'other'


def process_frame(frame, model, threshold=0.15):
    """Run YOLO on frame, annotate and return processed BGR image and stats."""
    results = model(frame)
    car_count = 0
    predictions = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name == 'car':
                car_count += 1
                car_crop = frame[y1:y2, x1:x2]
                color = get_car_color(car_crop, threshold=threshold)
                predictions.append(color)
                if color == 'blue':
                    rect_color = (0, 0, 255)
                    label = 'Blue Car'
                else:
                    rect_color = (255, 0, 0)
                    label = 'Other Car'
                cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), rect_color, cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    info_text = f"Cars: {car_count}"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    return frame, {'total': car_count, 'predictions': predictions}


def to_pil_image(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main():
    st.set_page_config(page_title='Car Color Detector', page_icon='🚗', layout='wide')
    st.title('🚗🔵Car Color Detector & Counter🔵🚗')

    model = load_model()

    # Sidebar controls
    st.sidebar.header('Input')
    uploaded = st.sidebar.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    sample_images = []
    sample_dir = 'sample_data'
    if os.path.isdir(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    selected_sample = None
    if not uploaded and sample_images:
        selected_sample = st.sidebar.selectbox('Or choose a sample image', [''] + sample_images)

    threshold = st.sidebar.slider('Blue threshold (ratio)', 0.0, 1.0, 0.15, 0.01)
    run = st.sidebar.button('Run Detection')

    # Load input image
    input_img = None
    input_name = None
    if uploaded:
        input_name = uploaded.name
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif selected_sample:
        input_name = selected_sample
        input_img = cv2.imread(os.path.join(sample_dir, selected_sample))

    if input_img is None:
        st.info('Upload an image or select a sample to start.')
        return

    # Show original image and processed on button press
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original')
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    if run:
        processed, stats = process_frame(input_img.copy(), model, threshold=threshold)
        pil_processed = to_pil_image(processed)

        with col2:
            st.subheader('Processed')
            st.image(pil_processed, use_container_width=True)

        # Metrics and download
        total = stats['total']
        blue = stats['predictions'].count('blue')
        other = stats['predictions'].count('other')
        st.markdown(f"**Total cars:** {total}  \n                     **Blue:** {blue}  \n                     **Other:** {other}")

        # Provide download
        buf = io.BytesIO()
        pil_processed.save(buf, format='PNG')
        buf.seek(0)
        st.download_button('Download processed image', data=buf, file_name=f'processed_{input_name}.png', mime='image/png')


if __name__ == '__main__':
    main()
