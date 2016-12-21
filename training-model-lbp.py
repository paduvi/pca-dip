from helpers import file_util, lbp_extract_feature
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

    for filePath in files:
        local_histograms = lbp_extract_feature.extract_face(join(train_folder, filePath),
                                                            log=join("lbp-model/image", filePath))  # 20x59
        norm = la.norm(local_histograms, axis=1)
        norm = norm.reshape((20, 1))
        local_histograms = np.divide(local_histograms, norm)
        file_util.log_data(local_histograms, log_file)
        log_file.write('\n')
    log_file.close()
