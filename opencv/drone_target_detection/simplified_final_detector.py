from target_detector import *
from target_classifier import *


cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#out = cv2.VideoWriter(
#    "output.mp4", 
#    cv2.VideoWriter_fourcc(*'H264'), 15, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    classifier_conf = classify_frame(frame)
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
    #out.write(res_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()