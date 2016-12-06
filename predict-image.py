import sys
from helpers import image_util
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

THRESHOLD = 0.55

def predict(data):
    # Step 1: Get normalize face and info
    mean_face = get_mean_face('model/mean-face.txt')
    height, width = get_info('model/info.txt')

    # Step 2: Resize and convert to vector
    resize_data = image_util.resize_image(data, width, height)
    vector, _, _ = image_util.matrix2vector(resize_data)

    # Plot mean face
    plt.figure(2)
    plt.axis("off")
    plt.title('Mean Image')
    plt.imshow(mean_face.reshape((height, width)), cmap=plt.get_cmap('gray'))

    normalize_face = np.subtract(vector, mean_face)  # N^2 x 1

    # Step 3: Get eigenvectors from model then calculate simplified face
    u = get_eigenvectors('model/list-eigenvectors.txt')  # N^2 x k
    w = np.dot(u.transpose(), normalize_face)  # (k x N^2) x (N^2 x 1) = k x 1
    simplified_normalize_face = np.dot(u, w)  # (N^2 x k) x (k x 1) = N^2 x 1
    simplified_face = np.add(simplified_normalize_face, mean_face)  # N^2 x 1

    # Plot simplified face:
    plt.figure(3)
    plt.axis("off")
    plt.title('Simplified Image')
    plt.imshow(simplified_face.reshape((height, width)), cmap=plt.get_cmap('gray'))

    # Step 4: Calculate error detection
    error = la.norm(np.subtract(normalize_face, simplified_normalize_face), axis=0)
    return error / (width * height) < THRESHOLD


def get_mean_face(file_path='model/mean-face.txt'):
    f = open(file_path, 'r')
    data = f.read().strip(';')
    return np.matrix(data)


def get_info(file_path='model/info.txt'):
    f = open(file_path, 'r')
    data = list(f.readlines())
    height = int(data[0].strip())
    width = int(data[1].strip())
    return height, width


def get_eigenvectors(file_path='model/list-eigenvectors.txt'):
    f = open(file_path, 'r')
    samples = list(f.readlines())
    f.close()
    u = []
    samples = [sample.strip() for sample in samples if len(sample.strip()) > 0]
    for sample in samples:
        sample = sample.strip(';')
        u = append_column(u, np.matrix(sample).transpose())
    return u


def append_column(arr, values):
    if len(arr) == 0:
        return values
    return np.append(arr, values, axis=1)


if __name__ == '__main__':
    filename = sys.argv[1]
    data = image_util.load_image(filename)
    if not image_util.is_grayscale(filename):
        data = image_util.rgb2gray(data)

    # Plot gray scale image
    plt.figure(1)
    plt.axis("off")
    plt.title('Gray Scale Image')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    print predict(data)

    # Show Plot
    plt.show()
