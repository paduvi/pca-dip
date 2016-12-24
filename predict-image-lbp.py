import sys
from helpers import image_util
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helpers import lbp_extract_feature
import math
from os.path import join


def append_column(arr, values):
    if len(arr) == 0:
        return values
    return np.append(arr, values, axis=1)


def get_histograms(file_path='lbp-model/histogram.txt'):
    f = open(file_path, 'r')
    samples = list(f.readlines())
    f.close()

    samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]  # m images

    histograms = []
    for sample in samples:
        u = []
        sample = sample.strip(';')
        data = [x.strip() for x in sample.split(";") if len(sample.strip(";")) > 0]
        for histogram in data:
            u = append_column(u, np.matrix(histogram).transpose())  # 59x20
        histograms.append(u.transpose())  # 20x59
    return np.array(histograms)


def get_mean_histograms(file_path='lbp-model/mean-histogram.txt'):
    f = open(file_path, 'r')
    sample = f.read().strip(';')
    f.close()
    u = []
    data = [x.strip() for x in sample.split(";") if len(sample.strip(";")) > 0]
    for histogram in data:
        u = append_column(u, np.matrix(histogram).transpose())  # 59x20
    return u.transpose()  # 20x59


def chi_square(s, m, threshold):
    if s.shape != m.shape:
        raise ArithmeticError('Size of 2 tuples not fit')
    rows, cols = s.shape
    result = 0
    for row in range(rows):
        for col in range(cols):
            x = s[row, col]
            y = m[row, col]
            if abs(x + y) != 0:
                result += math.pow((x - y), 2) / abs(x + y)
    similarity = result / (rows * cols)
    print similarity
    return similarity < threshold


def get_info(file_path='lbp-model/info.txt'):
    f = open(file_path, 'r')
    data = list(f.readlines())
    height = int(data[0].strip())
    width = int(data[1].strip())
    return height, width


if __name__ == '__main__':
    filename = sys.argv[1]
    try:
        THRESHOLD = float(sys.argv[2])
    except IndexError:
        THRESHOLD = 0.03
    data = image_util.load_image(filename)
    if not image_util.is_grayscale(filename):
        data = image_util.rgb2gray(data)
    height, width = get_info()

    resize_data = image_util.resize_image(data, width, height)
    local_histograms = lbp_extract_feature.extract_face(data=resize_data, log=join("output/", filename),
                                                        debug=True)  # 20x59
    norm = la.norm(local_histograms, axis=1)
    norm = norm.reshape((20, 1))
    local_histograms = np.divide(local_histograms, norm)

    mean_histograms = get_mean_histograms()

    result = chi_square(local_histograms, mean_histograms, threshold=THRESHOLD)
    print result
