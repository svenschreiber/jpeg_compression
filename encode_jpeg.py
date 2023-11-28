import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import ceil
from scipy.fftpack import dct, idct

css_block_size = 2
Q = 100

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

def ycbcr2rgb(img):
    coeffs = np.array([[1, 0, 1.402], [1, -.344136, -.714136], [1, 1.772, 0]])
    rgb = img.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(coeffs.T)
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

zz_indices = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

def reorder_zigzag(block):
    return block.flatten()[zz_indices]

img = cv2.imread("data/cat.bmp")[:,:,::-1]
fig.add_subplot(rows, columns, 1)
plt.imshow(img)
plt.axis('off')
plt.title("Original")

(w, h) = img.shape[:2]

# add padding to the image if it's size is not a multiple of 8
(padded_w, padded_h) = (ceil(w / 8) * 8, ceil(h / 8) * 8)
n_blocks = padded_w * padded_h / 64
img = np.pad(img, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode="mean")

# convert color space to YCbCr
img = rgb2ycbcr(img)

# apply chroma subsampling (4:2:0)
img = chroma_subsample(img)

# encode dct + quantization
img_dct = img.astype(np.int16)
dc = np.zeros((n_blocks, 3), dtype=np.int16)
ac = np.zeros((n_blocks, 63, 3), dtype=np.int16)
block_idx = 0
for y in range(0, padded_h, 8):
    for x in range(0, padded_w, 8):
        for z in range(3):
            block = img_dct[x:x + 8, y:y + 8, z] - 128
            #block = dct(dct(block.astype(np.float32).T, norm="ortho").T, norm="ortho")
            block = cv2.dct(block.astype(np.float32))
            block /= QTY if z == 0 else QTC
            img_dct[x:x + 8, y:y + 8, z] = block 
            zigzag = reorder_zigzag(block.astype(np.int16))
            dc[block_idx, z] = zigzag[0]
            ac[block_idx, :, z] = zigzag[1:]
        block_idx += 1


# decode dct + quantization
for y in range(0, padded_h, 8):
    for x in range(0, padded_w, 8):
        for z in range(3):
            block = img_dct[x:x + 8, y:y + 8, z]
            block = block * (QTY if z == 0 else QTC)
            block = cv2.idct(block)
            block = np.clip(block + 128, 0, 255).astype(np.uint8)
            img[x:x + 8, y:y + 8, z] = block

# convert back to rgb
img = ycbcr2rgb(img)

fig.add_subplot(rows, columns, 2)
plt.imshow(img)
plt.axis('off')
plt.title("Decoded")

plt.imsave("data/out.bmp", img)