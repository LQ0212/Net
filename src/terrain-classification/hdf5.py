import h5py
import numpy as np
from PIL import Image

# 打开 HDF5 文件
with h5py.File('testSet_c7_ver_2.hdf5', 'r') as file:
    # 遍历图像数据集
    for dataset_name in file:
        # 读取图像数据集
        image_data = file[dataset_name][()]

        # 将图像数据转换为图像对象
        image = Image.fromarray(image_data)

        # 保存图像
        image.save(f'extracted_images/{dataset_name}.jpg')