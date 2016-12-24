from helpers import file_util, lbp_extract_feature, image_util
from os.path import join
from numpy import linalg as la
import numpy as np

if __name__ == '__main__':
    train_folder = 'train_data'

    files = file_util.listFile(train_folder)
    m = len(files)

    # Extract feature histograms from all training images
    # Log histograms to text file
    log_file = open("lbp-model/histogram.txt", "w")
    count = 0

    for filePath in files:
        data = image_util.load_image(join(train_folder, filePath))
        if not image_util.is_grayscale(join(train_folder, filePath)):
            data = image_util.rgb2gray(data)
        local_histograms = lbp_extract_feature.extract_face(data, log=join("lbp-model/image", filePath))  # 20x59
        norm = la.norm(local_histograms, axis=1)
        norm = norm.reshape((20, 1))
        local_histograms = np.divide(local_histograms, norm)
        if count == 0:
            mean_histograms = local_histograms
        else:
            mean_histograms = np.add(mean_histograms, local_histograms)
        count += 1
        file_util.log_data(local_histograms, log_file)
        log_file.write('\n')
    log_file.close()

    # Log width, height of pca-model
    height, width = data.shape
    log_file = open("lbp-model/info.txt", "w")
    log_file.write(str(height) + '\n')
    log_file.write(str(width) + '\n')
    log_file.close()

    mean_histograms = np.divide(mean_histograms, count)

    log_file = open("lbp-model/mean-histogram.txt", "w")
    file_util.log_data(mean_histograms, log_file)
    log_file.close()
