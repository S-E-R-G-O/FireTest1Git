from ultralytics import YOLO
import cv2
import math


cap = cv2.VideoCapture('fire2.mp4')

out = cv2.VideoWriter('testx.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (640,480))


model = YOLO('best (1).pt')

# Reading the classes
classnames = ['fire']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classnames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
