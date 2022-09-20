import cv2
import numpy as np
import os
import glob
from PIL import Image

os.makedirs('./assets/test',exist_ok=True)
file = files = glob.glob('./assets/test/IMG_*.jpg')

for i, file in enumerate(files):
# 画像の読み込み(RGB)
  img = cv2.imread(file)
  print(img.shape)
  x, y, z = img.shape
  print(f"x: {x}, y: {y}, z: {z}")
  resize_img = cv2.resize(img, dsize = None, fx = 0.25, fy = 0.25)
  print(resize_img.shape)
  name = file.split('/')[-1]
  name = name.split('.')[0] 
  print(name)
  cv2.imwrite(f"./assets/test/{name}_r.jpg", resize_img)