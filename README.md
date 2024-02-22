# 项目简介

本项目基于YOLOv5为算法基础，采用PyQt作为UI基础制作，能够实现通用物体检测，达到障碍物检测功能。没有距离传感器所以无法精确测距



使用摄像头之前的设置，

```
# 设置摄像头，设置为0则为第一个摄像头
camera_no = 1
```



如何运行：

```
python main.py --weights weights/yolov5s.pt
```

