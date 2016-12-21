import sys
from helpers import image_util
import numpy as np
from numpy import linalg as la
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helpers import lbp_extract_feature
import math


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


def chi_square(s, m):
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
    return result / (rows * cols)


if __name__ == '__main__':
    filename = sys.argv[1]
    data = image_util.load_image(filename)
    if not image_util.is_grayscale(filename):
        data = image_util.rgb2gray(data)
    training_histograms = get_histograms()
    mean_histograms = np.mean(training_histograms, axis=0)
    local_histograms = lbp_extract_feature.extract_face(file_path=filename, debug=False)  # 20x59
    norm = la.norm(local_histograms, axis=1)
    norm = norm.reshape((20, 1))
    local_histograms = np.divide(local_histograms, norm)

    similarity = chi_square(local_histograms, mean_histograms)
    print similarity
