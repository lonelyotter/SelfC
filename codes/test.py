from data.util import read_img1
from utils.util import tensor2img, save_img
import torch
import numpy as np
import cv2

# HWC, BGR, [0, 1]
img1 = cv2.imread("/home/songhc/SelfC/results/test_SelfC_small_h264/city/1/1.jpg") / 255

# CHW, RGB, [0, 1]
img1_1 = np.transpose(img1[:, :, [2, 1, 0]], (2, 0, 1))

# CHW, RGB, [0, 1]
tensor1 = torch.from_numpy(np.ascontiguousarray(img1_1)).float()

# CHW, RGB, [0, 255]
img2 = (tensor1.numpy() * 255).round().astype(np.uint8)

# HWC, BGR, [0, 255], 到这一步还没什么误差
img2 = np.transpose(img2[[2, 1, 0], :, :], (1, 2, 0))

save_img(img2, "./test.jpg")

# HWC, BGR, [0, 1]
img3 = cv2.imread("./test.jpg") / 255

# CHW, RGB, [0, 1]
img3_1 = np.transpose(img3[:, :, [2, 1, 0]], (2, 0, 1))

print(np.mean(np.abs(img1_1 - img3_1)))
