import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
def imshow(inp, title=None):
    inp = inp.cpu()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.show()
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = '/home/liqi/net_maybe_can_use/Terrain-classification/frame_1713496092220.812.jpg'
image = Image.open(image_path)
transform = data_transforms['test']
transformed_image = transform(image)
t_img = transformed_image.view(1, 3, 224, 224)
t_img = t_img.cuda()
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# for child in model.named_children():
#     print(child)
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(1280, 32),
    nn.Linear(32, 5),
)
model.classifier = classifier
Label_list = ['brick', 'grass', 'gravel', 'others', 'sand']
# My_net = AlexNet()
model = model.to(device)
test_net = model
test_net = test_net.to(device)
test_net.eval()
test_net.load_state_dict(torch.load('./model_ALexNet.pth'))
t_output = test_net(t_img)
_, t_predict_label = torch.max(t_output, 1)
t_predict_label = t_predict_label.cpu()
t_predict_label = t_predict_label.numpy()
Predict_label = Label_list[t_predict_label[0]]
title_str = 'Predict label:' + Predict_label
imshow(transformed_image, title_str)
