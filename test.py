import torch
import cv2

model = torch.hub.load(r'C:\Users\AN\Desktop\pyqt_yolo\model', 'custom',
                           path=r"C:\Users\AN\Desktop\pyqt_yolo\weights\best.pt", source='local')
model.conf = 0.5
model.iou = 0.45


def box(frame,results):
    # print(results)
    results = results.pandas().xyxy[0].to_numpy()
    # print(results)
    color = (251, 238, 1)
    for box in results:
        cls = box[6]
        l, t, r, b = box[:4].astype('int')
        cv2.putText(frame, str(cls), (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(frame, (l, t), (r, b), color, 1)

    return frame


def run():
    cap = cv2.VideoCapture(r'demo.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = box(frame,results)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    run()










