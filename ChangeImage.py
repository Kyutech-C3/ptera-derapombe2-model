import cv2
import numpy as np
from IPython.display import Image, display


# def imshow(img):
#     ret, encoded = cv2.imencode(".jpg", img)
#     display(Image(encoded))

def adjust(img, alpha=1.0, beta=0.0):
    # 積和演算を行う。
    dst = alpha * img + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)


# 画像を読み込む。
src = cv2.imread("./assets/resize/X100-1/01rds_01_r.png")

# コントラスト、明るさを変更する。
dst = adjust(src, alpha=1.0, beta=100.0)
# imshow(dst)
cv2.imshow("Image", dst)
cv2.waitKey()