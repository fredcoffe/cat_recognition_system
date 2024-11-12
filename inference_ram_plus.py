'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

#创建命令行参数解释器
parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()  #解析命令行参数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)  #获取图像变换

    #######load model
    model = ram_plus(pretrained='resources/ram_plus_swin_large_14m.pth',
                     image_size=384,
                     vit='swin_l',
                     text_encoder_type='resources/bert-base-uncased')  #加载模型
    model.eval()  #设置模型为评估模式

    model = model.to(device)  #将模型移动到指定设备

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)  #将图像转换为张量并移动到指定设备

    res = inference(image, model)
    print("Image Tags: ", res[0])
    print("图像标签: ", res[1])
