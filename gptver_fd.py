import cv2
import time

def video_demo():
    # 加载Haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 打开摄像头
    capture = cv2.VideoCapture(0)
    
    if not capture.isOpened():
        print("无法打开摄像头")
        return

    prev_time = time.time()
    while True:
        # 读取帧
        ret, frame = capture.read()
        if not ret:
            print("无法接收帧 (流可能已结束？)")
            break

        # 转换为灰度图像以加快处理速度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Haar级联分类器检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # 绘制矩形框以突出显示人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 翻转图像以正常显示
        frame = cv2.flip(frame, 1)
        
        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示视频帧
        cv2.imshow("video", frame)

        # 检查按键
        c = cv2.waitKey(1)
        if c == 27:  # 如果按下ESC键，则退出循环
            break

    # 释放摄像头
    capture.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()

video_demo()
