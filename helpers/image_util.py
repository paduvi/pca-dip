import numpy as np
from PIL import Image, ImageStat


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    if is_grayscale(infilename) and data.ndim == 3:
        data = np.dot(data[..., :3], [1.0 / 3, 1.0 / 3, 1.0 / 3])
    return data


def is_grayscale(path):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)

    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False


def rgb2gray(rgb):
    return np.around(np.dot(rgb[..., :3], [0.299, 0.587, 0.114]))


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def img2vector(in_file_name):
    if is_grayscale(in_file_name):
        gray_data = load_image(in_file_name)
    else:
        origin_data = load_image(in_file_name)
        gray_data = rgb2gray(origin_data)
    return matrix2vector(gray_data)


def matrix2vector(data):
    h, w = data.shape
    result = data.reshape((w * h, 1))
    return result, w, h


def resize_image(data, width, height):
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img = img.resize((width, height), Image.ANTIALIAS)
    return np.asarray(img, dtype="uint8")
