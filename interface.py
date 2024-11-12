import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import pickle
import torch
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
import time  # 导入时间库
import socket  # 添加导入

'''''
****************************************************************************

10.30
交互网站可以与数控版交互，可以有摄像头等，可以发送指令（最后再弄）


****************************************************************************
'''''

class Cat:
    """猫类，用于表示一只猫的基本信息，包括名字、图像路径和标签。"""
    def __init__(self, name, image_path):
        self.name = name  # 猫的名字
        self.image_path = image_path  # 猫的图像路径
        self.labels = set()  # 标签集合

    def add_label(self, label):
        """添加标签到猫的标签集合中。"""
        self.labels.add(label)

    def remove_label(self, label):
        """从猫的标签集合中移除指定标签。"""
        self.labels.discard(label)


class CatDatabase:
    """猫数据库类，用于管理猫的集合，包括添加、删除和查找相似猫的功能。"""
    def __init__(self, db_path='cat_database.pkl'):
        self.db_path = db_path  # 数据库文件路径
        self.cats = []  # 猫的列表
        self.load_database()  # 加载数据库

    def add_cat(self, cat):
        """将一只猫添加到数据库中。"""
        self.cats.append(cat)  # 添加猫到数据库
        self.save_database()  # 保存数据库

    def remove_cat(self, cat):
        """从数据库中移除指定的猫。"""
        self.cats.remove(cat)  # 从数据库中移除猫
        self.save_database()  # 保存数据库

    def find_similar_cats(self, labels):
        """根据标签查找相似的猫，并返回相似猫的列表。"""
        similar_cats = []  # 存储相似猫的列表
        for cat in self.cats:
            similarity = self.compute_similarity(cat, labels)  # 计算相似度
            if similarity > 0:
                similar_cats.append((cat, similarity))  # 添加相似猫及其相似度

        # 按相似度排序
        similar_cats.sort(key=lambda x: x[1], reverse=True)

        # 确保至少有4只相似猫
        if len(similar_cats) < 4:
            additional_needed = 4 - len(similar_cats)
            remaining_cats = [cat for cat in self.cats if cat not in [c[0] for c in similar_cats]]
            additional_cats = remaining_cats[:additional_needed]
            similar_cats.extend((cat, 0) for cat in additional_cats)  # 添加额外猫

        return similar_cats

    def compute_similarity(self, cat, labels):
        """计算猫与给定标签的相似度。"""
        matching_labels = cat.labels.intersection(labels)  # 找到匹配的标签
        total_cats = len(self.cats)  # 总猫数
        similarity_score = len(matching_labels) / total_cats if total_cats > 0 else 0  # 计算相似度分数
        return similarity_score

    def save_database(self):
        """将猫数据库保存到文件中。"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.cats, f)  # 将猫列表保存到文件

    def load_database(self):
        """从文件加载猫数据库。"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.cats = pickle.load(f)  # 从文件加载猫列表
        else:
            self.cats = []  # 如果文件不存在，初始化为空列表


class CatRecognitionApp:
    """猫识别应用程序类，负责创建用户界面和处理用户交互。"""
    def __init__(self, master):
        self.master = master  # 主窗口
        self.master.title("猫身份录入系统")  # 设置窗口标题
        self.database = CatDatabase()  # 创建猫数据库实例
        self.recognized_labels = []  # 识别的标签
        self.selected_labels = set()  # 选中的标签集合
        self.selected_cat = None  # 选中的猫

        self.create_main_interface()  # 创建主界面

    def create_main_interface(self):
        """创建主界面，显示猫数据库和操作按钮。"""
        # 清空窗口
        for widget in self.master.winfo_children():
            widget.destroy()

        tk.Label(self.master, text="猫数据库").pack(pady=10)  # 显示标题
        self.cat_frame = tk.Frame(self.master)  # 创建猫框架
        self.cat_frame.pack()

        # 显示数据库中的猫
        for idx, cat in enumerate(self.database.cats):
            frame = tk.Frame(self.cat_frame)  # 创建每只猫的框架
            frame.grid(row=idx // 4, column=idx % 4, padx=10, pady=10)
            img = Image.open(cat.image_path)  # 打开猫的图像
            img = img.resize((100, 100))  # 调整图像大小
            photo = ImageTk.PhotoImage(img)  # 创建可显示的图像
            btn = tk.Button(frame, image=photo, command=lambda c=cat: self.edit_cat(c))  # 创建按钮以编辑猫
            btn.image = photo  # 保持对图像的引用
            btn.pack()
            tk.Label(frame, text=cat.name).pack()  # 显示猫的名字

        # 识别新猫按钮
        tk.Button(self.master, text="识别新猫", command=self.recognize_new_cat).pack(pady=10)

        # 摄像头检测按钮
        tk.Button(self.master, text="摄像头猫脸检测", command=self.start_camera_detection).pack(pady=10)

        # 退出按钮
        tk.Button(self.master, text="退出", command=self.master.quit).pack(pady=10)

    def start_camera_detection(self):
        """启动摄像头检测界面。"""
        self.camera_window = tk.Toplevel(self.master)  # 创建新窗口
        self.camera_window.title("摄像头猫脸检测")  # 设置窗口标题

        # 创建UDP套接字并绑定
        self.video_source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_source.bind(("0.0.0.0", 9090))  # 监听所有接口的9090端口

        self.face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalcatface.xml')  # 加载猫脸检测分类器

        self.cat_detected_time = 0  # 检测到猫的时间计数
        self.detection_interval = 3  # 设定检测到猫的持续时间（秒）
        self.screenshot_folder = 'cat_save'  # 截图保存文件夹
        os.makedirs(self.screenshot_folder, exist_ok=True)  # 创建文件夹

        # 创建一个标签用于显示摄像头画面
        self.camera_label = tk.Label(self.camera_window)
        self.camera_label.pack()

        self.update_camera_frame()  # 开始更新摄像头画面

    def update_camera_frame(self):
        """更新摄像头画面并进行猫脸检测。"""
        try:
            if not self.camera_window.winfo_exists():  # 检查窗口是否仍然存在
                return

            data, addr = self.video_source.recvfrom(65507)  # 最大UDP数据包大小

            # 转换为字节流并读取图像
            img_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码图像

            if frame is None:
                print("无法解码图像")
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # 检测猫脸

            # 检测到猫脸后，增加时间计数
            if len(faces) > 0:
                self.cat_detected_time += 1
                print(f"检测到猫脸: {self.cat_detected_time / 3:.1f} 秒")  # 打印检测到的时间（假设30fps）

                if self.cat_detected_time >= self.detection_interval * 3:  # 假设30fps
                    # 截图并保存
                    screenshot_path = os.path.join(self.screenshot_folder, f'cat_{time.time()}.jpg')
                    cv2.imwrite(screenshot_path, frame)
                    print(f"检测到猫并截图: {screenshot_path}")
                    self.cat_detected_time = 0  # 重置计数

                    # 询问用户是否进行猫脸识别
                    if messagebox.askyesno("猫脸识别", "检测到猫脸，是否进行猫脸识别？"):
                        self.recognize_cat(screenshot_path)  # 进��猫脸识别

                # 绘制猫脸矩形
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 绘制矩形框

            else:
                self.cat_detected_time = 0  # 重置计数

            # 将图像转换为Tkinter可用的格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色通道
            img = Image.fromarray(frame)  # 将NumPy数组转换为PIL图像
            img = img.resize((640, 480))  # 调整图像大小
            self.photo = ImageTk.PhotoImage(img)  # 创建可显示的图片
            self.camera_label.config(image=self.photo)  # 更新标签的图像

        except KeyboardInterrupt:
            print("摄像头检测已停止。")
            self.stop_camera_detection()  # 停止摄像头检测
        except Exception as e:
            print("发生错误:", e)

        # 继续更新画面
        if self.camera_window.winfo_exists():  # 确保窗口仍然存在
            self.camera_window.after(10, self.update_camera_frame)  # 继续更新画面

    def stop_camera_detection(self):
        """停止摄像头检测并关闭窗口。"""
        self.video_source.close()  # 关闭UDP套接字
        self.camera_window.destroy()  # 关闭摄像头检测窗口

    def recognize_cat(self, image_path):
        """进行猫脸识别并处理识别结果。"""
        labels = self.run_inference(image_path)  # 运行推理并获取标签
        if not labels:
            messagebox.showerror("错误", "未能识别任何标签。")
            return

        self.recognized_image_path = image_path  # 保存识别的图片路径
        self.recognized_labels = set(labels)  # 保存识别的标签
        self.selected_labels = set()  # 初始化选中的标签集合

        # 进入界面1
        self.create_interface1()

    def run_inference(self, image_path):
        """运行推理以识别猫的标签。"""
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 图像预处理
        transform = get_transform(image_size=384)

        # 加载模型
        model = ram_plus(pretrained='pretrained/ram_plus_swin_large_14m.pth',
                         image_size=384,
                         vit='swin_l',
                         text_encoder_type='resources/bert-base-uncased')
        model.eval()  # 设置模型为评估模式
        model.to(device)  # 将模型移动到指定设备

        # 加载并处理图像
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)  # 转换为张量并移动到设备

        # 推理
        result = inference(image, model)

        # result[1] 包含中文标签
        labels_string = result[1]  # 这是一个用'|'分隔的标签字符串

        if labels_string is None:
            messagebox.showerror("错误", "未能识别任何标签。")
            return []

        # 分割标签并去除空白
        labels = [label.strip() for label in labels_string.split('|') if label.strip()]

        return labels

    def create_interface1(self):
        """创建界面1，显示识别的猫图像和标签。"""
        # 清空窗口
        for widget in self.master.winfo_children():
            widget.destroy()

        # 显示识别的图像
        image_frame = tk.Frame(self.master)
        image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        img = Image.open(self.recognized_image_path)
        img = img.resize((200, 200))
        self.photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(image_frame, image=self.photo)
        img_label.pack()

        # 显示标签
        label_frame = tk.Frame(self.master)
        label_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(label_frame, text="识别的标签:").pack()
        self.label_buttons = {}
        for label in self.recognized_labels:
            btn = tk.Button(label_frame, text=label, relief=tk.RAISED,
                            command=lambda l=label: self.toggle_label(l))
            btn.pack(pady=2)
            self.label_buttons[label] = btn

        # 显示相似猫的图像和相似度
        similar_cats_frame = tk.Frame(self.master)
        similar_cats_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(similar_cats_frame, text="相似的猫:").pack()
        similar_cats = self.database.find_similar_cats(self.recognized_labels)  # 查找相似猫
        self.similar_cats = similar_cats  # 存储以备后用
        self.cat_buttons = {}
        for idx, (cat, similarity) in enumerate(similar_cats):
            frame = tk.Frame(similar_cats_frame)
            frame.pack(pady=5)
            img = Image.open(cat.image_path)
            img = img.resize((100, 100))
            photo = ImageTk.PhotoImage(img)
            btn = tk.Button(frame, image=photo, relief=tk.RAISED, command=lambda c=cat: self.select_cat(c))
            btn.image = photo  # 保持对图像的引用
            btn.pack()
            # 计算相似度百分比
            total_cats = len(self.database.cats)
            if total_cats > 0:
                similarity_percent = (similarity / total_cats) * 100
            else:
                similarity_percent = 0
            tk.Label(frame, text=f"{cat.name}").pack()
            # 使用进度条可视化相似度
            progress = tk.Canvas(frame, width=100, height=10)
            progress.pack()
            progress.create_rectangle(0, 0, similarity_percent, 10, fill='green')
            tk.Label(frame, text=f"相似度: {similarity_percent:.1f}%").pack()
            self.cat_buttons[cat.name] = btn

        # 决定是现有猫还是新猫的按钮
        decision_frame = tk.Frame(self.master)
        decision_frame.pack(side=tk.BOTTOM, pady=10)

        tk.Button(decision_frame, text="这是库中的猫", command=self.confirm_existing_cat).pack(side=tk.LEFT, padx=5)
        tk.Button(decision_frame, text="这是一只新猫", command=self.create_new_cat).pack(side=tk.LEFT, padx=5)
        tk.Button(decision_frame, text="退出", command=self.master.quit).pack(side=tk.LEFT, padx=5)

    def toggle_label(self, label):
        """切换标签的选择状态。"""
        if label in self.selected_labels:
            self.selected_labels.remove(label)  # 移除标签
            self.label_buttons[label].config(relief=tk.RAISED)  # 更新按钮状态
        else:
            self.selected_labels.add(label)  # 添加标签
            self.label_buttons[label].config(relief=tk.SUNKEN)  # 更新按钮状态

    def select_cat(self, cat):
        """切换猫的选择状态。"""
        if self.selected_cat == cat:
            self.selected_cat = None  # 取消选择
            self.cat_buttons[cat.name].config(relief=tk.RAISED)  # 更新按钮状态
        else:
            if self.selected_cat:
                self.cat_buttons[self.selected_cat.name].config(relief=tk.RAISED)  # 取消之前选择的猫
            self.selected_cat = cat  # 选择新猫
            self.cat_buttons[cat.name].config(relief=tk.SUNKEN)  # 更新按钮状态

    def confirm_existing_cat(self):
        """确认选择的猫是现有猫，并更新标签。"""
        if self.selected_cat:
            # 更新猫的标签
            for label in self.selected_labels:
                self.selected_cat.add_label(label)
            # 移除未选择的标签
            for label in self.recognized_labels:
                if label not in self.selected_labels and label in self.selected_cat.labels:
                    self.selected_cat.remove_label(label)
            self.database.save_database()  # 保存数据库
            self.create_interface2()  # 进入界面2
        else:
            messagebox.showwarning("警告", "请选择一只猫。")  # 提示用户选择猫

    def create_interface2(self):
        """创建界面2，显示选中猫的图像和信息。"""
        # 清空窗口
        for widget in self.master.winfo_children():
            widget.destroy()

        cat = self.selected_cat  # 获取选中的猫

        img = Image.open(cat.image_path)  # 打开猫的图像
        img = img.resize((200, 200))  # 调整图像大小
        self.cat_photo = ImageTk.PhotoImage(img)  # 创建可显示的图像
        img_label = tk.Label(self.master, image=self.cat_photo)  # 显示猫的图像
        img_label.pack(pady=10)

        tk.Label(self.master, text=f"猫的名字: {cat.name}").pack(pady=5)  # 显示猫的名字

        tk.Button(self.master, text="完成", command=self.create_main_interface).pack(pady=10)  # 返回主界面
        tk.Button(self.master, text="退出", command=self.master.quit).pack(pady=10)  # 退出程序

    def create_new_cat(self):
        """进入创建新猫的界面。"""
        self.create_new_cat_interface()

    def create_new_cat_interface(self):
        """创建新猫的界面。"""
        # 清空窗口
        for widget in self.master.winfo_children():
            widget.destroy()

        tk.Label(self.master, text="创建新猫").pack(pady=5)  # 显示标题

        # 显示识别的图像
        img = Image.open(self.recognized_image_path)
        img = img.resize((200, 200))
        self.new_cat_photo = ImageTk.PhotoImage(img)  # 创建可显示的图像
        img_label = tk.Label(self.master, image=self.new_cat_photo)
        img_label.pack(pady=10)

        tk.Label(self.master, text="给猫起名字:").pack()  # 提示用户输入猫的名字
        self.cat_name_entry = tk.Entry(self.master)  # 输入框
        self.cat_name_entry.pack(pady=5)

        tk.Button(self.master, text="保存", command=self.save_new_cat).pack(pady=10)  # 保存新猫
        tk.Button(self.master, text="退出", command=self.master.quit).pack(pady=10)  # 退出程序

    def save_new_cat(self):
        """保存新猫的信息到数据库。"""
        cat_name = self.cat_name_entry.get()  # 获取猫的名字
        if not cat_name:
            messagebox.showwarning("警告", "请输入猫的名字。")  # 提示用户输入名字
            return
        # 保存图像到猫的文件夹
        cats_folder = 'cats'
        if not os.path.exists(cats_folder):
            os.makedirs(cats_folder)  # 创建文件夹
        image_filename = os.path.join(cats_folder, f"{cat_name}.jpg")  # 图像文件名
        img = Image.open(self.recognized_image_path).convert('RGB')  # 转换图像模式
        img.save(image_filename)  # 保存图像
        # 创建新猫并添加到数据库
        cat = Cat(cat_name, image_filename)  # 创建猫实例
        for label in self.selected_labels:
            cat.add_label(label)  # 添加标签
        self.database.add_cat(cat)  # 添加猫到数据库
        messagebox.showinfo("成功", f"猫 '{cat_name}' 已被添加到库中。")  # 提示成功消息
        self.create_main_interface()  # 返回主界面

    def recognize_new_cat(self):
        """让用户选择一张图片进行猫脸识别。"""
        image_path = filedialog.askopenfilename(title="选择猫的图片",  # 弹出文件对话框
                                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_path:
            return  # 如果没有选择图片，返回

        # 运行推理代码
        labels = self.run_inference(image_path)  # 运行推理并获取标签
        if not labels:
            messagebox.showerror("错误", "未能识别任何标签。")  # 显示错误消息
            return

        self.recognized_image_path = image_path  # 保存识别的图片路径
        self.recognized_labels = set(labels)  # 保存识别的标签
        self.selected_labels = set()  # 初始化选中的标签集合

        # 进入界面1
        self.create_interface1()

    def edit_cat(self, cat):
        """编辑选中猫的信息。"""
        # 清空当前窗口
        for widget in self.master.winfo_children():
            widget.destroy()

        # 显示猫的信息
        img = Image.open(cat.image_path)  # 打开猫的图像
        img = img.resize((200, 200))  # 调整图像大小
        self.cat_photo = ImageTk.PhotoImage(img)  # 创建可显示的图像
        img_label = tk.Label(self.master, image=self.cat_photo)  # 显示猫的图像
        img_label.pack(pady=10)

        # 编辑猫的名字
        tk.Label(self.master, text="猫的名字:").pack(pady=5)  # 提示用户输入猫的名字
        self.cat_name_entry = tk.Entry(self.master)  # 输入框
        self.cat_name_entry.insert(0, cat.name)  # 填入当前猫的名字
        self.cat_name_entry.pack(pady=5)

        # 更换猫的照片按钮
        tk.Button(self.master, text="更换照片", command=lambda: self.change_cat_photo(cat)).pack(pady=5)

        # 删除猫的按钮
        tk.Button(self.master, text="删除猫", command=lambda: self.remove_cat(cat)).pack(pady=5)  # 新增删除按钮

        # 显示标签
        tk.Label(self.master, text="标签:").pack(pady=5)  # 显示标签标题
        self.label_buttons = {}
        for label in cat.labels:
            frame = tk.Frame(self.master)  # 创建标签框架
            frame.pack(pady=2)
            tk.Label(frame, text=label).pack(side=tk.LEFT)  # 显示标签
            btn = tk.Button(frame, text="删除", command=lambda l=label: self.remove_cat_label(cat, l))  # 删除标签按钮
            btn.pack(side=tk.RIGHT)
            self.label_buttons[label] = btn  # 保持对按钮的引用

        # 保存更改按钮
        tk.Button(self.master, text="保存更改", command=lambda: self.save_cat_changes(cat)).pack(pady=10)

        # 返回按钮
        tk.Button(self.master, text="返回", command=self.create_main_interface).pack(pady=10)

    def change_cat_photo(self, cat):
        """更换选中猫的照片。"""
        # 让用户选择新的图片
        new_image_path = filedialog.askopenfilename(title="选择新的猫的图片", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if new_image_path:
            cat.image_path = new_image_path  # 更新猫的图片路径
            self.edit_cat(cat)  # 重新显示猫的信息以更新图片

    def remove_cat_label(self, cat, label):
        """从猫的标签中删除指定标签。"""
        cat.remove_label(label)  # 从猫的标签中删除
        self.edit_cat(cat)  # 重新显示猫的信息以更新标签

    def save_cat_changes(self, cat):
        """保存对选中猫的更改。"""
        new_name = self.cat_name_entry.get()  # 获取新的猫名字
        if new_name:
            cat.name = new_name  # 更新猫的名字
            self.database.save_database()  # 保存数据库
            messagebox.showinfo("成功", "猫的信息已更新。")  # 提示成功消息
            self.create_main_interface()  # 返回主界面
        else:
            messagebox.showwarning("警告", "猫的名字不能为空。")  # 提示用户输入名字

    def remove_cat(self, cat):
        """从数据库中删除选中的猫。"""
        self.database.remove_cat(cat)  # 从数据库中移除猫
        messagebox.showinfo("成功", f"猫 '{cat.name}' 已被删除。")  # 提示成功消息
        self.create_main_interface()  # 返回主界面


def main():
    """主函数，启动应用程序。"""
    root = tk.Tk()  # 创建主窗口
    app = CatRecognitionApp(root)  # 创建猫识别应用实例
    root.mainloop()  # 运行主循环


if __name__ == '__main__':
    main()  # 调用主函数
