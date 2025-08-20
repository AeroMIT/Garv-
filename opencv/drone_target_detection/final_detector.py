import cv2
import tensorflow as tf
import numpy as np
from ultralytics import YOLO


# Path to your trained model
model_path = "drone_target_classifier.h5"

# Load the trained MobileNet model
model = tf.keras.models.load_model(model_path)

# Define class labels (same order as training)
class_labels = ["target", "nothing"]

def target_classifier(frame):
    # Resize to match training size
    img = cv2.resize(frame, (640, 640))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0  # normalize
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    return confidence

def cv_target_detector(frame):
    # Load your trained model
    model = YOLO("runs\detect\train5\weights\best.pt")

    # Run YOLO inference
    result_frame = frame.copy()
    results = model(result_frame)

    # Draw detections
    for result in results:
        is_box = False
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])   # confidence
            cls = int(box.cls[0])       # class id
            label = model.names[cls]    # class name

            if conf > 0.90:
                # Draw bounding box
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label + confidence
                cv2.putText(result_frame, f"{label} {conf:.2f}", 
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
                is_box = True

    return result_frame, is_box


    
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output.mp4", 
    cv2.VideoWriter_fourcc(*'H264'), 10, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    classifier_conf = target_classifier(frame)
    res_frame, is_box = cv_target_detector(frame)

    if not is_box and classifier_conf < 0.75:
        cv2.putText(res_frame, "FAKE", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 255), 4)
            
    elif not is_box and classifier_conf > 0.75:
        cv2.putText(res_frame, "NOTHING", 
            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
            2, (0, 255, 255), 4)

    cv2.imshow('FINAL DETECTION', res_frame)
    out.write(res_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
    

