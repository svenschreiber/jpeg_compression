import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
from scipy.fftpack import dct, idct

css_block_size = 2
dct_block_size = 8
Q = 50

QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

S = 5000/Q if Q < 50 else 200 - 2*Q
QTY = np.clip(np.floor((S * QTY + 50) / 100), 1, None)
QTC = np.clip(np.floor((S * QTC + 50) / 100), 1, None)


def rgb2ycbcr(img):
    coeffs = np.array([[.299, .587, .114], [-.168736, -.331264, .5], [.5, -.418688, -.081312]])
    ycbcr = img.dot(coeffs.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def chroma_subsample(ycbcr):
    w, h = ycbcr.shape[:2]
    block_sz = css_block_size
    a = np.array(ycbcr)
    for y in range(0, h, block_sz):
        for x in range(0, w, block_sz):
            block = a[x:x + block_sz, y:y + block_sz, 1:]
            a[x:x + block_sz, y:y + block_sz, 1:] = np.full((block_sz, block_sz,2), block[0,0])
    return np.uint8(a)

img = cv2.imread("data/cat.bmp")[:,:,::-1]
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Original")

(w, h) = img.shape[:2]

(padded_w, padded_h) = (ceil(w / dct_block_size) * dct_block_size, ceil(h / dct_block_size) * dct_block_size)

img = np.pad(img, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode="mean")

img = rgb2ycbcr(img)
img = chroma_subsample(img)

img_dct = img.astype(np.int16)
for y in range(0, padded_h, dct_block_size):
    for x in range(0, padded_w, dct_block_size):
        for z in range(3):
            block = img_dct[x:x + dct_block_size, y:y + dct_block_size, z] - 128
            block = cv2.dct(block.astype(np.float32))
            block /= QTY if z == 0 else QTC
            img_dct[x:x + dct_block_size, y:y + dct_block_size, z] = block

# decode
for y in range(0, padded_h, dct_block_size):
    for x in range(0, padded_w, dct_block_size):
        for z in range(3):
            block = img_dct[x:x + dct_block_size, y:y + dct_block_size, z]
            block = (block * (QTY if z == 0 else QTC)).astype(np.int16)
            block = cv2.idct(block.astype(np.float32))
            img[x:x + dct_block_size, y:y + dct_block_size, z] = np.clip(block + 128, 0, 255).astype(np.uint8)

img = ycbcr2rgb(img)

fig.add_subplot(rows, columns, 2)
plt.imshow(img)
plt.axis('off')
plt.title("Decoded")

plt.imsave("data/out.bmp", img)