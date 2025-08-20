import cv2
import tensorflow as tf
import numpy as np

model_path = r"C:\Users\Admin\Desktop\programs\ML\CNN\drone_target_detection\drone_target_classifier.h5"
model = tf.keras.models.load_model(model_path)
class_labels = ["target", "nothing"]

def classify_frame(frame):
    img = cv2.resize(frame, (640, 640))
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return confidence

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        conf = classify_frame(frame)

        if conf < 0.8:
            label = "target"
        else:
            label = "nothing"

        text = f"{label} ({conf:.2f})"

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Webcam Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
