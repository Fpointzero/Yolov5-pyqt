import argparse
import sys
import threading
import tkinter as tk
from tkinter import filedialog

import cv2
import torch
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QPixmap, QBrush
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from tqdm import tqdm

from detect import ROOT
from ui.mainUi import Ui_Form

model_path = 'weights/yolov5s.pt'


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 初始化窗口
        self.setWindowTitle("My Window")  # 设置窗口标题
        self.resize(1280, 720)  # 设置窗口大小
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)  # 禁用全屏
        layout = QVBoxLayout()  # 创建垂直布局

        # 创建按钮
        form = Ui_Form()
        form.setupUi(self)
        form.pushButton_2.clicked.connect(self.btn_click_detect_img)  # 选择图片按钮
        form.pushButton_3.clicked.connect(self.btn_click_detect_video)  # 选择视频按钮
        form.pushButton_4.clicked.connect(self.btn_click_detect_camera)  # 选择摄像机按钮
        # 将按钮添加到布局中
        # layout.addWidget(form)

        # 设置窗口背景图像
        palette = self.palette()
        background_image = QPixmap("ui/img/backgroud.jpg")  # 请将 "img/backgroud.jpg" 替换为实际的图像文件路径
        scaled_image = background_image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        background_brush = QBrush(scaled_image)
        palette.setBrush(QPalette.Background, background_brush)
        self.setPalette(palette)

        # 将布局设置为窗口的主布局
        self.setLayout(layout)

    def paintEvent(self, event):
        pass

    def btn_click_detect_img(self):
        root = tk.Tk()
        root.withdraw()

        # 打开文件对话框，只允许选择图片文件
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])

        if file_path:
            thread = threading.Thread(target=detect_img, args=(file_path,))
            thread.start()

    def btn_click_detect_video(self):
        root = tk.Tk()
        root.withdraw()

        # 打开文件对话框，只允许选择视频文件
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])

        if file_path:
            thread = threading.Thread(target=detect_video, args=(file_path,))
            thread.start()

    def btn_click_detect_camera(self):
        thread = threading.Thread(target=camera_detect, args=())
        thread.start()


def detect_img(file_path):
    results = model(file_path, size=640)
    results.show()
    # subprocess.run(
    #     ['venv/Scripts/python', 'detect.py', '--weights', f'{model_path}', '--device', '0',
    #      '--conf-thres', '0.05'] + ['--source', f'{file_path}'])
    # subprocess.run('start .\\runs\\detect', shell=True)


def detect_video(file_path):
    video_capture = cv2.VideoCapture(file_path)
    if not video_capture.isOpened():
        print("视频文件")
        return None
    # total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # pbar = tqdm(total=total_frames)
    while True:
        # 逐帧捕获视频
        ret, frame = video_capture.read()

        if not ret:
            break
        # 将图像转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = model(image, size=640)

        # results.show()
        frame = draw_boxes(frame, results, model)
        # 显示图像
        cv2.imshow('Video', frame)
        # pbar.update(1)
        # 按下 'Esc' 键退出循环
        if cv2.waitKey(1) & 0XFF == 27:
            break

    # 释放摄像头资源
    try:
        video_capture.release()
        cv2.destroyAllWindows()
    except:
        pass


# def camera_detect():
#     # 加载模型权重
#     # model = torch.hub.load(".", 'custom', path="weights/yolov5s.pt", source='local')
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     checkpoint = torch.load(model_path, map_location=device)
#     # model = torch.load("weights/yolov5s.pt", map_location=torch.device("cpu"))
#     model = checkpoint['model']
#     model.to(device)
#     # 打开摄像头
#     video_capture = cv2.VideoCapture(1)
#
#     while True:
#         if not video_capture.isOpened():
#             print("无法打开摄像头")
#             exit()
#         # 逐帧捕获视频
#         ret, frame = video_capture.read()
#
#         # 将图像转换为PIL图像
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#         # 将图像转换为Tensor格式
#         # image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
#         # 进行目标检测
#         results = model(image, size=640)
#
#         frame = draw_boxes(frame, results, model)
#
#         # 显示图像
#         cv2.imshow('Object Detection', frame)
#
#         # 按下 'q' 键退出循环
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # 释放摄像头资源
#     video_capture.release()
#     cv2.destroyAllWindows()


def camera_detect():
    # 打开摄像头
    # model = torch.hub.load(".", 'custom', path=model_path, source='local')
    # 用网络摄像头所以是1，本地的摄像头用0
    video_capture = cv2.VideoCapture(camera_no)

    while True:
        # 逐帧捕获视频
        if not video_capture.isOpened():
            print("无法打开摄像头")
            return None
        ret, frame = video_capture.read()
        # 将图像转换为PIL图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = model(image, size=640)

        # results.show()
        frame = draw_boxes(frame, results, model)
        # 显示图像
        cv2.imshow('Camera', frame)

        # 按下 'Esc' 键退出循环
        if cv2.waitKey(1) & 0XFF == 27:
            break

    # 释放摄像头资源
    video_capture.release()
    cv2.destroyAllWindows()


def draw_boxes(frame, results, model):
    for _detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = _detection
        label = model.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
    return frame


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / f'{model_path}',
                        help='model path or triton URL')
    args = parser.parse_args()
    model_path = args.weights[0]
    # 加载模型
    model = torch.hub.load(".", 'custom', path=model_path, source='local')

    # 设置摄像头
    camera_no = 1
    # windows
    main()
