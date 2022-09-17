import cv2
import numpy as np
import os
import glob
from PIL import Image

os.makedirs('./assets/resize',exist_ok=True)
files = glob.glob('./assets/origin/*.png')

for i, file_path in enumerate(files):
  try:
    # 画像の読み込み(RGB)
    img = cv2.imread(file_path)
    print(img.shape)
    resize_img = cv2.resize(img, (100,100))
    print(resize_img.shape)
    cv2.imwrite("./assets/resize/{}rds_0{}_r.png".format(str(i).zfill(2),i), resize_img)
  except Exception:
    continue