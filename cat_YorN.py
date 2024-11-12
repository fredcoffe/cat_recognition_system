# coding=utf-8
import cv2
import os

# 设置猫脸检测分类器路径
cat_path = 'haarcascade/haarcascade_frontalcatface.xml'
face_cascade = cv2.CascadeClassifier(cat_path)

def detect_and_tag_image(image_path):
    # 加载图像
    frame = cv2.imread(image_path)
    if frame is None:
        print("无法加载图像:", image_path)
        return

    # 转换为灰度图进行猫脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.02,
        minNeighbors=3,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 检测到猫脸并标记
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # 绘制猫脸矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 保存标记后的图像
        output_path = 'output_marked_image.jpg'
        cv2.imwrite(output_path, frame)
        print(f"检测到猫，处理完成，输出图像保存为: {output_path}")
    else:
        print("未检测到猫脸。")

if __name__ == "__main__":
    # 图片路径
    image_path = 'demo9.jpg'  # 确保此路径正确
    detect_and_tag_image(image_path)
