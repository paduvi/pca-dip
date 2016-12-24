from helpers import image_util
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_gray_scale_image(file_path):
    data = image_util.load_image(file_path)
    if not image_util.is_grayscale(file_path):
        data = image_util.rgb2gray(data)
    return data


def calculate_decimal(block):
    line1 = block[0, 0] * (2 ** 0) + block[0, 1] * (2 ** 1) + block[0, 2] * (2 ** 2)
    line2 = block[1, 2] * (2 ** 3) + block[0, 0] * (2 ** 7)
    line3 = block[2, 2] * (2 ** 4) + block[2, 1] * (2 ** 5) + block[2, 0] * (2 ** 6)
    return line1 + line2 + line3


def present_block(block):
    result = np.empty([3, 3], dtype=int)
    for row in range(3):
        for col in range(3):
            if block[row, col] > block[1, 1]:
                result[row, col] = 0
            else:
                result[row, col] = 1
    return result


def is_uniform(decimal):
    times = 0
    old_bit = decimal & 1
    for i in range(1, 8):
        new_bit = (decimal >> i) & 1
        if new_bit != old_bit:
            times += 1
        old_bit = new_bit
    if times > 2:
        return False
    return True


def get_histogram(present_data):
    bins = []
    nums_none_uniform = 0
    height, width = present_data.shape
    for i in range(256):
        times = 0
        for row in range(height):
            for col in range(width):
                if present_data[row, col] == i:
                    times += 1
        if is_uniform(i):
            bins.append(times)
        else:
            nums_none_uniform += times
    bins.insert(0, nums_none_uniform)  # none_uniform index = 0
    return np.array(bins)


def extract_segment(data, debug=False):
    height, width = data.shape
    present_image = np.empty([height - 2, width - 2], dtype=int)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            block = present_block(data[row - 1:row + 2, col - 1:col + 2])
            present_image[row - 1, col - 1] = calculate_decimal(block)

    if debug:
        plt.figure(1)
        plt.title('Present Face')
        plt.imshow(present_image, cmap=plt.get_cmap('gray'))

    histogram = get_histogram(present_image)

    if debug:
        plt.figure(2)

        plt.hist(x=np.arange(59), weights=histogram, bins=59, color='green', alpha=0.8)
        plt.xlabel("Bin No.")
        plt.ylabel("Frequency")

    if debug:
        plt.show()

    return histogram


def extract_face(data, debug=False, log=None):
    height, width = data.shape

    present_image = np.empty([height - 2, width - 2], dtype=int)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            block = present_block(data[row - 1:row + 2, col - 1:col + 2])
            present_image[row - 1, col - 1] = calculate_decimal(block)

    if log is not None:
        directory = os.path.dirname(log)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_util.save_image(present_image, log)

    local_histograms = np.empty([20, 59], dtype=int)
    global_histogram = np.zeros(59, dtype=int)
    for row in range(5):
        for col in range(4):
            histogram = extract_segment(
                data[height * row / 5: height * (row + 1) / 5, width * col / 4:width * (col + 1) / 4])
            local_histograms[row * 4 + col] = histogram
            global_histogram = np.add(global_histogram, histogram)

    if debug:
        plt.figure()
        plt.title("Global Histogram")
        plt.hist(x=np.arange(59), weights=global_histogram, bins=59, color='green', alpha=0.8)
        plt.xlabel("Bin No.")
        plt.ylabel("Frequency")

    if debug:
        plt.show()
    return local_histograms


if __name__ == '__main__':
    data = load_gray_scale_image('../train_data/s1/1.pgm')
    extract_segment(data, True)

    # extract_face('train_data/s1/1.pgm', True)
