import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

def get_car_color(car_image):
    # Convert the input image from the BGR color space to the HSV color space.
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color blue in the HSV color space.
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a binary 'mask'.
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Count the number of non-zero pixels in the mask.
    blue_pixel_count = cv2.countNonZero(mask)

    # Calculate the total number of pixels in the car's image area.
    total_pixel_count = car_image.shape[0] * car_image.shape[1]
    if total_pixel_count == 0:
        return 'other'
    # Calculate the ratio of blue pixels to the total pixels.
    blue_ratio = blue_pixel_count / total_pixel_count
    # Classify based on threshold.
    if blue_ratio > 0.15:
        return 'blue'
    else:
        return 'other'

def process_image_for_evaluation(image_path):
    """
    Process image for evaluation: detect cars, classify colors, return data.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return None, []
    car_data = []
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name == 'car':
                car_crop = frame[y1:y2, x1:x2]
                color = get_car_color(car_crop)
                car_data.append({
                    'bbox': (x1, y1, x2, y2),
                    'color': color
                })
                # Draw on frame
                if color == 'blue':
                    rect_color = (0, 0, 255)
                    label = "Blue Car"
                else:
                    rect_color = (255, 0, 0)
                    label = "Other Car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), (x1 + label_size[0], label_y + 5), rect_color, cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    car_count = len(car_data)
    info_text = f"Cars: {car_count}"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    return frame, car_data

def main():
    # Create evaluation folder if not exists
    evaluation_dir = 'evaluation_result'
    os.makedirs(evaluation_dir, exist_ok=True)

    # Sample data directory
    sample_data_dir = 'sample_data'

    # Ground truth: For demonstration, assuming all cars are 'other'. Replace with actual GT.
    # GT should be a dict: {image_name: [{'color': 'blue/other'}, ...]}
    ground_truth = {}  # Placeholder

    all_predictions = []
    all_true = []

    for image_file in os.listdir(sample_data_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(sample_data_dir, image_file)
            processed_frame, car_data = process_image_for_evaluation(image_path)
            if processed_frame is not None:
                # Save processed image
                output_path = os.path.join(evaluation_dir, f'eval_{image_file}')
                cv2.imwrite(output_path, processed_frame)

                # Collect predictions
                predictions = [car['color'] for car in car_data]
                all_predictions.extend(predictions)

                # Collect true labels if available
                if image_file in ground_truth:
                    true_colors = [car['color'] for car in ground_truth[image_file]]
                    all_true.extend(true_colors)
                else:
                    # If no GT, assume 'other' for all
                    all_true.extend(['other'] * len(predictions))

                print(f"Processed {image_file}: {len(car_data)} cars detected.")

    # Generate classification report
    if all_true and all_predictions:
        labels = ['other', 'blue']
        # Use labels and zero_division to avoid missing-class errors when one class is absent
        report = classification_report(all_true, all_predictions, labels=labels, target_names=labels, zero_division=0)
        print("\nClassification Report:")
        print(report)

        # Save report to file
        with open(os.path.join(evaluation_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # Confusion matrix (ensure order matches labels)
        cm = confusion_matrix(all_true, all_predictions, labels=labels)

        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks and label them with the respective list
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2. if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        cm_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
        fig.savefig(cm_path, bbox_inches='tight')
        plt.close(fig)
    else:
        print("No data for classification report.")

    # Summary
    total_cars = len(all_predictions)
    blue_cars = all_predictions.count('blue')
    other_cars = all_predictions.count('other')
    print(f"\nTotal cars detected: {total_cars}")
    print(f"Blue cars: {blue_cars}")
    print(f"Other cars: {other_cars}")

if __name__ == "__main__":
    main()