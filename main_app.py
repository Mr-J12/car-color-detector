import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# --- Core Machine Learning and Image Processing Logic ---

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

def get_car_color(car_image):
    # Convert the input image from the BGR color space to the HSV color space.
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color blue in the HSV color space.
    # These values can be adjusted to capture different shades of blue.
    # Hue [100-140] typically covers the blue spectrum.
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a binary 'mask'. This is an image where pixels that fall within the defined
    # blue range are turned white (value 255), and all other pixels are turned black (value 0).
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Count the number of non-zero (i.e., white) pixels in the mask. This gives us the total
    # number of blue pixels in the car image.
    blue_pixel_count = cv2.countNonZero(mask)

    # Calculate the total number of pixels in the car's image area.
    total_pixel_count = car_image.shape[0] * car_image.shape[1]
    # To prevent a division-by-zero error if the cropped image is empty.
    if total_pixel_count == 0:
        return 'other'
    # Calculate the ratio of blue pixels to the total pixels in the image.
    blue_ratio = blue_pixel_count / total_pixel_count
    # If the percentage of blue pixels is above a certain threshold (e.g., 15%),
    # we classify the car as 'blue'. This helps avoid false positives from small blue details
    # or reflections.
    if blue_ratio > 0.15:
        return 'blue'
    else:
        return 'other'

def process_image(image_path):
    """
    This is the main processing function. It loads an image, performs object detection,
    classifies car colors, and annotates the image with bounding boxes and counts.

    Args:
        image_path (str): The file path of the image to be processed.

    Returns:
        np.array: The processed image with all visual annotations, in BGR format.
    """
    # Read the image file from the given path into a NumPy array using OpenCV.
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    # Initialize counters for the objects we want to track.
    car_count = 0
    
    # Run the YOLOv8 model on the entire image frame to get detection results.
    results = model(frame)

    # The results can contain detections for multiple objects. We loop through each result.
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name == 'car':
                car_count += 1
                car_crop = frame[y1:y2, x1:x2]
                color = get_car_color(car_crop)
                if color == 'blue':
                    rect_color = (0, 0, 255)
                    label = "Blue Car"
                else:
                    rect_color = (255, 0, 0)
                    label = "Other Car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 60)
                label_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), rect_color, cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 255, 255), 5)

    # --- Display Car Count Only ---
    info_text = f"Cars: {car_count}"
    font_scale = 1.2
    font_thickness = 2
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 3)
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return frame

# --- Graphical User Interface (GUI) using Tkinter ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Color Detector & Counter")
        self.root.geometry("1200x800")

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10)

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.load_button = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        image_frame = tk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.original_label = tk.Label(image_frame, text="Original Image Preview", font=("Helvetica", 14))
        self.original_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.processed_label = tk.Label(image_frame, text="Processed Image Output", font=("Helvetica", 14))
        self.processed_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        
        original_img = Image.open(file_path)
        original_img.thumbnail((580, 580))
        self.original_photo = ImageTk.PhotoImage(original_img)
        self.original_label.config(image=self.original_photo, text="")
        self.original_label.image = self.original_photo

        processed_frame = process_image(file_path)
        if processed_frame is not None:
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            processed_img = Image.fromarray(processed_rgb)
            processed_img.thumbnail((580, 580))
            self.processed_photo = ImageTk.PhotoImage(processed_img)
            self.processed_label.config(image=self.processed_photo, text="")
            self.processed_label.image = self.processed_photo

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
