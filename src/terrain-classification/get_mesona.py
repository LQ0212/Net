import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from PIL import Image
if __name__ == '__main__':
    # step1 检查并初始化设备可用性
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
    # for child in model.named_children():
    #     print(child)
    # step4 循环读取帧内容，如果需要并输出
    print("Shooting ...")
    while True:
        # 读取帧内容
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        folder_path = "./Mesona/Sand/"
        # 将帧数据转换为可用的格式
        frame_data = np.asanyarray(color_frame.get_data())
        timestamp_ms = color_frame.get_timestamp()
        height, width, _ = frame_data.shape
        start_row = height - 224
        start_col = (width - 224) // 2
        cropped_data = frame_data[start_row:height, start_col:start_col + 224, :]
        # 显示图像
        x1, y1 = start_row, start_col
        x2, y2 = start_row + 224, start_col + 224
        color = (0, 255 ,0)
        thickness = 2
        cv2.rectangle(frame_data, (y1,x1), (y2,x2), color, thickness)
        cv2.imshow("get train set",frame_data)
        # file_name = "frame_{}.jpg".format(timestamp_ms)
        # file_path = os.path.join(folder_path, file_name)
        # cv2.imwrite(file_path, cropped_data)
        # 退出循环的条件（按下 'q' 键）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 保存图像
            break

    # 停止并释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
    # 裁剪下方中间的 224x224 区域
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # 归一化 RGB 数据
    # normalized_data = cropped_data.astype(np.float32) / 255.0
    # # build model
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # # for child in model.named_children():
    # #     print(child)
    # classifier = nn.Sequential(
    #     nn.Dropout(0.25),
    #     nn.Linear(1280, 32),
    #     nn.Linear(32, 5),
    # )
    # model.classifier = classifier
    # print("classifier changes: ", model.classifier)
    # Label_list = ['brick', 'grass', 'gravel', 'others', 'sand']
    # # My_net = AlexNet()
    # model = model.to(device)
    # test_net = model
    # test_net = test_net.to(device)
    # test_net.eval()
    # test_net.load_state_dict(torch.load('./model_ALexNet.pth'))
    # t_output = test_net(normalized_data)
    # _, t_predict_label = torch.max(t_output, 1)
    # t_predict_label = t_predict_label.cpu()
    # t_predict_label = t_predict_label.numpy()
    # Predict_label = Label_list[t_predict_label[0]]
    # title_str = 'Predict label:' + Predict_label
    # imshow(t_img_to_show, title_str)
