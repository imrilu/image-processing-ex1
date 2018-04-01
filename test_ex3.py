from os.path import relpath
import numpy as np
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as signal
import matplotlib.pyplot as plt


def read_image(filename, representation):
    """
    this functions reads an image using imread
    :param filename: a given image file path
    :param representation: 1 or 2 representing gray scale or RGB
    :return: None if failed, the read image as float64 type
    """
    im = imread(filename)
    if representation == 1:
        return rgb2gray(im).astype(np.float64) / 255
    elif representation == 2:
        return im.astype(np.float64) / 255

def create_gaussian_filter(size):
    if(size == 1):
        return np.array([1])
    filter = np.array([1, 1]).astype(np.float64)
    filter2 = np.array([1, 1]).astype(np.float64)
    #convoles [1,1] with itself the appropriate amount of times.
    for i in range(1, size - 1):
        filter = np.convolve(filter, filter2)
    return np.divide(filter, filter.sum())

def build_gaussian_pyramid(im, max_levels, filter_size):
    max_levels = int(min(max_levels, round(np.log2(im.shape[0]) - 3), round(np.log2(im.shape[1]) - 3)))
    filter_vec = create_gaussian_filter(filter_size)
    pyr = [im]
    im = signal.convolve2d(im, [filter_vec], mode='same')
    im = signal.convolve2d(im, np.array([filter_vec]).T, mode='same')
    for i in range(1, max_levels):
        im = im[::2, ::2]
        pyr.append(im)
    filter_vec = np.reshape(filter_vec, (1, filter_size))
    return pyr, filter_vec

def enlarge_image(im, filter):
    temp_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    temp_im[::2, ::2] = im
    temp_im = signal.convolve2d(temp_im, [filter * 2], mode='same')
    temp_im = signal.convolve2d(temp_im, np.array([filter * 2]).T, mode='same')
    return temp_im

def build_laplacian_pyramid(im, max_levels, filter_size):
    max_levels = int(min(max_levels, round(np.log2(im.shape[0]) - 3), round(np.log2(im.shape[1]) - 3)))
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    filter_vec = np.reshape(filter_vec,  filter_size)
    for i in range(1, max_levels):
        temp_im = enlarge_image(gauss_pyr[i], filter_vec)
        temp_im = temp_im[:gauss_pyr[i-1].shape[0], :gauss_pyr[i-1].shape[1]]
        pyr.append(gauss_pyr[i - 1] - temp_im)
    pyr.append(gauss_pyr[-1])
    filter_vec = np.reshape(filter_vec, (1, filter_size))
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    filter_vec = np.reshape(filter_vec, filter_vec.shape[1])
    image = np.zeros(lpyr[0].shape)
    for i in range(0, len(coeff)):
        lpyr[i] *= coeff[i]
        temp = lpyr[i]
        for j in range(i, 0, -1):
            temp = enlarge_image(temp, filter_vec)
            temp = temp[:image.shape[0], :image.shape[1]]
        image += temp
    return image

pic = read_image("C:\ex1\gray_orig.png",1)
lplc = build_laplacian_pyramid(pic, 5, 5)
gauss = build_gaussian_pyramid(pic, 5, 5)

res = laplacian_to_image(lplc[0], lplc[1], [1,1,1,1,1])
# new = render_pyramid(lplc[0], 5)

plt.imshow(res, cmap='gray')
plt.show()