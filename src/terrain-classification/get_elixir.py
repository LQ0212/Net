import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.models as models
from tqdm import tqdm
from time import sleep
# from arguments import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device name:', device)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
}
data_dir = './Mesona2'

train_sets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
train_loader = torch.utils.data.DataLoader(train_sets, batch_size=16, shuffle=True)
train_size = len(train_sets)
train_classes = train_sets.classes

val_sets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
val_loader = torch.utils.data.DataLoader(val_sets, batch_size=16, shuffle=False)
val_size = len(val_sets)

test_sets = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_sets, batch_size=16, shuffle=False)
test_size = len(test_sets)

print('train_size:', train_size)
print('val_size:', val_size)
print('test_size:', test_size)


# build model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# for child in model.named_children():
#     print(child)
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(1280, 128),
    nn.Linear(128, 32),
    nn.Linear(32, 4),
)
model.classifier = classifier
print("classifier changes: ", model.classifier)

# My_net = AlexNet()
model = model.to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)


# 画出数据集的图像
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


#  画出损失函数随着epoch的变化图
def show_loss(epoch_list, train_loss_list, val_loss_list):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_list, train_loss_list, color='blue', linestyle="-", label="Train_Loss")
    plt.plot(epoch_list, val_loss_list, color='red', linestyle="-", label="Val_Loss")
    plt.legend()
    plt.show()


#  画出准确率随着epoch的变化图
def show_accurate(epoch_list, train_accurate_list, val_accurate_list):
    plt.xlabel('epochs')
    plt.ylabel('accurate')
    plt.plot(epoch_list, train_accurate_list, color='blue', linestyle="-", label="Train_accurate")
    plt.plot(epoch_list, val_accurate_list, color='red', linestyle="-", label="Val_accurate")
    plt.legend()
    plt.show()


#  训练模型
def train(Epoch):
    best_loss = 1000
    epoch_list = [i for i in range(Epoch)]
    train_loss_list = []
    val_loss_list = []
    train_accurate_list = []
    val_accurate_list = []

    print('************************************Start training************************************')
    sleep(1)
    for epoch in range(Epoch):
        model.train()
        running_loss = 0.0
        accurate_train = 0
        for i, data in tqdm(enumerate(train_loader)):
            inputs, label = data
            inputs = inputs.to(device)  # 传入GPU
            label = label.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            accurate_train += (outputs.argmax(1) == label).sum()
        running_loss = running_loss / len(train_loader)
        model.eval()  # 评估模型
        val_loss = 0  # 验证集的loss
        accurate_val = 0  # 验证集正确个数
        with torch.no_grad():
            for i, data1 in enumerate(val_loader):
                inputs, labels = data1
                inputs1 = inputs.to(device)
                labels1 = labels.to(device)
                outputs1 = model(inputs1)
                loss1 = loss_function(outputs1, labels1)
                val_loss += loss1.item()
                accurate_val += (outputs1.argmax(1) == labels1).sum()
            val_loss = val_loss / len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './model_Elixir.pth')
        print('epoch:', epoch + 1, '|train_loss:%.3f' % running_loss, 'val_loss:%.3f' % val_loss,
              'train_accurate:{}%.'.format(accurate_train / len(train_sets) * 100),
              'val_accurate:{}%.'.format(accurate_val / len(val_sets) * 100))
        train_loss_list.append(running_loss)
        accurate_train=accurate_train.cpu()
        train_accurate_list.append(accurate_train / len(train_sets))
        val_loss_list.append(val_loss)
        accurate_val=accurate_val.cpu()
        val_accurate_list.append(accurate_val / len(val_sets))
        

    print('************************************Training Finish************************************')
    #  可视化
    show_loss(epoch_list, train_loss_list, val_loss_list)
    show_accurate(epoch_list, train_accurate_list, val_accurate_list)


def test_accurate():
    test_net = model
    test_net = test_net.to(device)
    test_net.eval()
    test_net.load_state_dict(torch.load('./model_Elixir.pth'))
    accurate_test = 0  # 正确个数
    with torch.no_grad():
        for i, data1 in enumerate(test_loader):
            inputs, labels = data1
            inputs1 = inputs.to(device)
            labels1 = labels.to(device)
            outputs1 = test_net(inputs1)
            accurate_test += (outputs1.argmax(1) == labels1).sum()
    print('test_correct_num:', accurate_test.item())
    print('test_accurate:{}%'.format(accurate_test / test_size * 100))


def train_accurate():
    train_net = model
    train_net = train_net.to(device)
    train_net.eval()
    train_net.load_state_dict(torch.load('./model_Elixir.pth'))
    accurate_train = 0  # 正确个数
    with torch.no_grad():
        for i, data1 in enumerate(train_loader):
            inputs, labels = data1
            inputs1 = inputs.to(device)
            labels1 = labels.to(device)
            outputs1 = train_net(inputs1)
            accurate_train += (outputs1.argmax(1) == labels1).sum()
    print('train_correct_num:', accurate_train.item())
    print('train_accurate:{}%'.format(accurate_train / train_size * 100))


def val_accurate():
    val_net = model
    val_net = val_net.to(device)
    val_net.eval()
    val_net.load_state_dict(torch.load('./model_Elixir.pth'))
    accurate_val = 0  # 正确个数
    with torch.no_grad():
        for i, data1 in enumerate(val_loader):
            inputs, labels = data1
            inputs1 = inputs.to(device)
            labels1 = labels.to(device)
            outputs1 = val_net(inputs1)
            accurate_val += (outputs1.argmax(1) == labels1).sum()
    print('val_correct_num:', accurate_val.item())
    print('val_accurate:{}%'.format(accurate_val / val_size * 100))


def show_test_img(num):  # 挑选测试集的前num张图像进行输出测试
    Label_list = ['Asphalt', 'Grass', 'Rock', 'Sand']
    for i in range(num):
        test_net = model
        test_net = test_net.to(device)
        test_net.eval()
        test_net.load_state_dict(torch.load('./model_Elixir.pth'))
        t_index = random.randint(0, 1448)
        t_img = test_sets[t_index][0]
        t_img_to_show = t_img
        t_true_label = test_sets[t_index][1]
        t_img = t_img.view(1, 3, 224, 224)
        t_img = t_img.cuda()
        t_output = test_net(t_img)
        _, t_predict_label = torch.max(t_output, 1)
        t_predict_label = t_predict_label.cpu()
        t_predict_label = t_predict_label.numpy()
        True_label = Label_list[t_true_label]
        Predict_label = Label_list[t_predict_label[0]]
        title_str = 'True label:' + True_label + '  ' + 'Predict label:' + Predict_label
        imshow(t_img_to_show, title_str)

# train(Epoch=15)
# train_accurate()
# # # if args.run_mode=='train':
# # #     train(Epoch=args.train_epochs)
# # #     train_accurate()
# # if args.run_mode=='test':
show_test_img(10)
test_accurate()
val_accurate()
