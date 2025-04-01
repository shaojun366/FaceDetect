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

    # 设置摄像头分辨率
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 设置处理分辨率
    process_width = 320
    process_height = 240

    prev_time = time.time()
    frame_count = 0
    fps_update_interval = 30  # 每30帧更新一次FPS显示

    while True:
        # 读取帧
        ret, frame = capture.read()
        if not ret:
            print("无法接收帧 (流可能已结束？)")
            break

        # 调整处理分辨率
        frame = cv2.resize(frame, (process_width, process_height))
        
        # 转换为灰度图像以加快处理速度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Haar级联分类器检测人脸，优化参数
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # 增加scaleFactor以减少检测次数
            minNeighbors=3,   # 降低minNeighbors以提高检测速度
            minSize=(20, 20), # 减小最小检测尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 绘制矩形框以突出显示人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 翻转图像以正常显示
        frame = cv2.flip(frame, 1)
        
        # 计算FPS（降低更新频率以减少计算开销）
        frame_count += 1
        if frame_count % fps_update_interval == 0:
            curr_time = time.time()
            fps = fps_update_interval / (curr_time - prev_time)
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
