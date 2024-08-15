import torch
from tracker.byte_tracker import BYTETracker
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.7
    track_buffer: int = 40
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.5
    min_box_area: float = 1.0
    mot20: bool = False

byte_tracker = BYTETracker(BYTETrackerArgs())

model = torch.hub.load(r'C:\Users\AN\Desktop\pyqt_yolo\model', 'custom',
                           path=r"C:\Users\AN\Desktop\pyqt_yolo\weights\best.pt", source='local')
model.conf = 0.5
model.iou = 0.45

def box(frame,results):
    results = results.pandas().xyxy[0].to_numpy()

    confidences = results[:, 4]
    # aaa = confidences[:, None]
    # print(aaa)
    byte_track_input = np.hstack((results[:, :4], confidences[:, None]))
    tracks = byte_tracker.update(byte_track_input, frame.shape, frame.shape)

    for track in tracks:
        track_id = track.track_id
        bbox = track.tlbr.astype(np.int32)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (251, 238, 1), 1)
        cv2.putText(frame, str(track_id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def run():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20, (int(1344), int(756)))
    cap = cv2.VideoCapture(r'test_video.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = box(frame,results)

        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()










