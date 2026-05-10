# Cat Recognition System

这是一个猫脸/猫身份识别实验项目。它包含一个 Tkinter 桌面界面，可以维护猫咪图片数据库，并用 RAM（Recognize Anything Model）提取图片标签来辅助匹配相似猫咪。

## 它做什么

- 维护本地猫咪图片和标签数据库。
- 选择新图片后识别标签，并查找相似猫咪。
- 可接收摄像头/UDP 图像流，检测猫脸并保存截图。
- 附带 Arduino 草图，可作为硬件端实验参考。

## 项目结构

```text
interface.py              桌面主界面
inference_ram_plus.py     RAM 模型推理示例
cat_YorN.py               OpenCV 猫脸检测示例
EPSoutput.py              UDP 图像接收示例
cats/                     示例猫咪图片
cat_save/                 运行时截图目录
sketch_jul10a/            Arduino 示例
requirements.txt          Python 依赖
```

## 快速运行

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python interface.py
```

macOS/Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

## 模型文件

项目代码会使用 RAM 模型。模型权重通常较大，不建议直接提交到 GitHub。请按 RAM 项目说明下载权重，并在代码或启动参数中指向本地模型路径。

## 公开前检查

- 未发现明文 API Key。
- 原依赖文件包含本机 `F:\...` 绝对路径，已改成可复现安装方式。
- `cat_database.pkl`、运行截图和模型权重建议作为本地数据，不要继续提交。
