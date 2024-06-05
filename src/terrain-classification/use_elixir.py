#!/home/liqi/anaconda3/envs/torch-3.9/bin/python3
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from std_msgs.msg import Int16
from PIL import Image
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
}
if __name__ == '__main__':
    # step1 检查并初始化设备可用性
    pub = rospy.Publisher('terrain', Int16, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(30)
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No realsense D435i was detected.")
        exit()
    device = ctx.devices[0]
    serial_number = device.get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial_number)

    # step2 根据配置文件设置数据流
    config.enable_stream(rs.stream.color,
                        848, 480,
                        rs.format.bgr8, 30)

    # step3 启动相机流水线并设置是否自动曝光
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    color_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]  # 0-depth(两个infra)相机, 1-rgb相机,2-IMU
    # 自动曝光设置
    color_sensor.set_option(rs.option.enable_auto_exposure, True)
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
    Label_list = ['Asphalt', 'Grass',  'Rock', 'Sand']
    # My_net = AlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_net = model
    test_net = test_net.to(device)
    test_net.eval()
    test_net.load_state_dict(torch.load('./model_Elixir.pth'))
    # step4 循环读取帧内容，如果需要并输出
    print("Shooting ...")
    while not rospy.is_shutdown():
        # 读取帧内容
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将帧数据转换为可用的格式
        frame_data = np.asanyarray(color_frame.get_data())
        timestamp_ms = color_frame.get_timestamp()
        height, width, _ = frame_data.shape
        start_row = height - 224
        start_col = (width - 224) // 2
        cropped_data = frame_data[start_row:height, start_col:start_col + 224, :]
        transform = data_transforms['test']
        pil_image = Image.fromarray(cropped_data)
        transformed_image = transform(pil_image)
        t_img = transformed_image.view(1, 3, 224, 224)
        t_img = t_img.cuda()
        t_output = test_net(t_img)
        _, t_predict_label = torch.max(t_output, 1)
        t_predict_label = t_predict_label.cpu()
        t_predict_label = t_predict_label.numpy()
        pub.publish(t_predict_label[0])
        rate.sleep()
        Predict_label = Label_list[t_predict_label[0]]
        title_str = 'Predict label:' + Predict_label
        # rospy.INFO(title_str)
        # # cv2.imwrite("frame_{}.jpg".format(timestamp_ms), cropped_data)
        # # 显示图像
        # x1, y1 = start_row, start_col
        # x2, y2 = start_row + 224, start_col + 224
        # color = (0, 255 ,0)
        # thickness = 2
        # cv2.rectangle(frame_data, (y1,x1), (y2,x2), color, thickness)
        # cv2.imshow(title_str,frame_data)
        # # 退出循环的条件（按下 'q' 键）
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # 保存图像
        #     break

    # 停止并释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
