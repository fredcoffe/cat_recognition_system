import socket
import io
from PIL import Image
import numpy as np
import cv2

# 创建UDP套接字并绑定
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 9090))  # 监听所有接口的9090端口

while True:
    try:
        # 接收数据
        data, addr = s.recvfrom(65507)  # 最大UDP数据包大小
        print("接收到数据大小:", len(data))  # 打印接收到的数据大小

        bytes_stream = io.BytesIO(data)  # 转换为字节流
        image = Image.open(bytes_stream)  # 使用PIL打开图像
        img = np.asarray(image)  # 转换为NumPy数组
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR格式

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
