# Terrain classification
***Author***: Li Qi

***E-mail***: liqi0037@e.ntu.edu.sg

## About this program


## Requirements
1. You need a [conda](https://www.anaconda.com) enviroment with a suitable Python version:
    ```
    conda create -n YOUR_ENV_NAME python==3.9
    conda activate YOUR_ENV_NAME
    ```
2. Install pytorch(GPU version with your CUDA version), matplotlib, tqdm and numpy:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install matplotlib
   pip install tqdm
   pip install numpy
   ```
## Run this program
1. I provide this terrian dataset in the project file"dixing2" which is pre-seperated by **data_seperate.py**. If you wants to use the source dataset in your way [here](https://drive.google.com/file/d/1hNgvmXk9PifjSBLo2L4cESjqi8ncOcG7/view?usp=sharing) it is.
2. To train this model, run:
   ```
   python Building.py -- run_mode train --train_epochs YOUR_TRAIN_EPOCHS
   ```
   To test the result, run:
   ```
   python Building.py -- run_mode test
   ```
## Result
