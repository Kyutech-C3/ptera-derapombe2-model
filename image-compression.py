import cv2
import numpy as np
import os
import glob
from PIL import Image

os.makedirs('compression',exist_ok=True)
files = glob.glob('./assets/resize/*.png')

for i, file_path in enumerate(files):
  try:
    # 画像の読み込み(RGB)
    img = cv2.imread(file_path)
    cv2.imwrite("./assets/compression/rds_0{}_rc.png".format(i), img)
    for j in range(5,10):
      compression_x_img = cv2.resize(img, dsize=None, fx=0.1*j, fy=1)
      compression_y_img = cv2.resize(img, dsize=None, fx=1, fy=0.1*j)
      # compression_x_img = cv2.resize(compression_x_img, (100,100))
      # compression_y_img = cv2.resize(compression_y_img, (100,100))
      cv2.imwrite("./assets/compression/rds_0{}_rcx{}.png".format(i,j), compression_x_img)
      cv2.imwrite("./assets/compression/rds_0{}_rcy{}.png".format(i,j), compression_y_img)
  except Exception:
    continue