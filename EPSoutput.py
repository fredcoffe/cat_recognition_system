import cv2
import socket
import numpy as np
import os
import time

# 设置猫脸检测分类器路径
cat_path = 'haarcascade/haarcascade_frontalcatface.xml'
face_cascade = cv2.CascadeClassifier(cat_path)

# 创建UDP套接字并绑定
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 9090))  # 监听所有接口的9090端口

cat_detected_time = 0  # 检测到猫的时间计数
detection_interval = 3  # 设定检测到猫的持续时间（秒）
screenshot_folder = 'cat_save'  # 截图保存文件夹
os.makedirs(screenshot_folder, exist_ok=True)  # 创建文件夹

def manage_screenshots():
    # 获取当前文件夹中的所有截图
    screenshots = sorted(os.listdir(screenshot_folder), key=lambda x: os.path.getmtime(os.path.join(screenshot_folder, x)))
    if len(screenshots) > 4:
        # 删除最早的一个截图
        os.remove(os.path.join(screenshot_folder, screenshots[0]))

while True:
    try:
        # 接收数据
        data, addr = s.recvfrom(65507)  # 最大UDP数据包大小

        # 转换为字节流并读取图像
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码图像

        if img is None:
            print("无法解码图像")
            continue

        # 转换为灰度图进行猫脸检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 检测到猫脸后，增加时间计数
        if len(faces) > 0:
            cat_detected_time += 1
            if cat_detected_time >= detection_interval * 30:  # 假设30fps
                # 截图并保存
                screenshot_path = os.path.join(screenshot_folder, f'cat_{time.time()}.jpg')
                cv2.imwrite(screenshot_path, img)
                print(f"检测到猫并截图: {screenshot_path}")
                manage_screenshots()  # 管理截图数量

            # 绘制猫脸矩形
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        else:
            cat_detected_time = 0  # 重置计数

        # 显示图像
        cv2.imshow("ESP32 Capture Image", img)

        # 按下 'q' 键退出
        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print("发生错误:", e)

# 释放资源
s.close()
cv2.destroyAllWindows()
