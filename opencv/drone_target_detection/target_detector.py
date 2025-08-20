import cv2
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def yolo_target_detector(frame):

    model = YOLO("runs\detect\train5\weights\best.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.80, verbose=False)
        annotated_frame = results[0].plot()

    return annotated_frame

def cv_target_detector(frame):

    model = YOLO("runs\detect\train5\weights\best.pt")

    result_frame = frame.copy()
    results = model(result_frame)

    for result in results:
        is_box = False
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > 0.80:
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                is_box = True

    return result_frame, is_box

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res_frame, is_box = cv_target_detector(frame)
        cv2.imshow('YOLO detection', res_frame)
        print(is_box)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

