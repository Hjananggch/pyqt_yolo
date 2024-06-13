import sys
import cv2
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout,
                             QFileDialog, QLabel, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

#加载模型
model = torch.hub.load(r'C:\Users\AN\Desktop\pyqt_yolo\model', 'custom',
                       path=r"C:\Users\AN\Desktop\pyqt_yolo\weights\best.pt", source='local')
model.conf = 0.5
model.iou = 0.45

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv5 Video Player')
        self.setGeometry(100, 100, 1000, 700)

        self.layout = QVBoxLayout()

        title = QLabel('YOLOv5 Object Detection Video Player')
        title.setFont(QFont('Arial', 20))
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)

        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.videoLabel)

        buttonLayout = QHBoxLayout()
        self.openButton = QPushButton('Open Video')
        self.openButton.setFont(QFont('Arial', 14))
        self.openButton.clicked.connect(self.openVideoFile)
        buttonLayout.addWidget(self.openButton)

        self.quitButton = QPushButton('Quit')
        self.quitButton.setFont(QFont('Arial', 14))
        self.quitButton.clicked.connect(self.close)
        buttonLayout.addWidget(self.quitButton)

        self.layout.addLayout(buttonLayout)
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)

        self.cap = None

    def openVideoFile(self):
        videoPath, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if videoPath != '':
            self.cap = cv2.VideoCapture(videoPath)
            self.timer.start(30)

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        if ret:
            results = model(frame)
            frame = self.box(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.videoLabel.setPixmap(QPixmap.fromImage(image))
        else:
            self.timer.stop()
            self.cap.release()

    def box(self, frame, results):
        results = results.pandas().xyxy[0].to_numpy()
        color = (251, 238, 1)
        for box in results:
            cls = box[6]
            l, t, r, b = box[:4].astype('int')
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, str(cls), (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        return frame

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())

