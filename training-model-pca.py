from helpers import image_util, file_util
import sys, os
import numpy as np
from numpy import linalg as la
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def append_column(arr, values):
    if len(arr) == 0:
        return values
    return np.append(arr, values, axis=1)


if __name__ == '__main__':
    if sys.argv[1]:
        try:
            k = int(sys.argv[1])
        except ValueError:
            k = 8
    else:
        k = 8
    train_folder = 'train_data'

    log_file = open("model/list-file.txt", "w")
    files = file_util.listFile(train_folder)
    m = len(files)

    # Step 1: Convert to vectors:
    raw_vectors = []  # N^2 x M

    for filePath in files:
        vector, width, height = image_util.img2vector(join(train_folder, filePath))  # N^2 x 1
        raw_vectors = append_column(raw_vectors, vector)
        file_util.log_data(filePath, log_file)  # Log to text file
        log_file.write('\n')

    log_file.close()
    size = width * height

    # Log width, height of model
    log_file = open("model/info.txt", "w")
    log_file.write(str(height) + '\n')
    log_file.write(str(width) + '\n')
    log_file.close()

    # Step 2: Get mean-face:
    mean_face = np.mean(raw_vectors, axis=1).reshape((size, 1))  # N^2 x 1
    plt.figure(1)
    plt.title('Mean Face')
    plt.imshow(mean_face.reshape((height, width)), cmap=plt.get_cmap('gray'))

    log_file = open("model/mean-face.txt", "w")
    file_util.log_data(mean_face, log_file)
    log_file.close()
    mean_face_repeat = np.repeat(mean_face, m, axis=1)

    # Step 3: Get normalized-face:
    normalized_faces = np.subtract(raw_vectors, mean_face_repeat)  # N^2 x M

    # Step 4: Calculate A^T.A
    covariance = np.dot(normalized_faces.transpose(), normalized_faces)

    # Step 5: Calculate Eigenvalues & Eigenvectors (normalized) of covariance. Sort by Eigenvalues ASC order:
    eigenvalues, v = la.eig(covariance)
    idx = np.argsort(eigenvalues)
    eigenvalues = np.array(eigenvalues)[idx]
    v = np.array(v)[idx]

    # Step 6: Choose K greatest Eigenvalues, then get Eigenvectors u of A.A^T:
    eigenvalues = eigenvalues[-k:]
    v = v[-k:]  # k x M
    u = np.dot(normalized_faces, v.transpose())  # (N^2 x M) x (M x k) = N^2 x k
    norm = la.norm(u, axis=0)
    u = np.divide(u, norm)  # Normalize u

    # Log Eigenvectors to text file
    log_file = open("model/list-eigenvectors.txt", "w")
    for i in range(k):
        file_util.log_data(u[:, i], log_file)
        log_file.write('\n')
    log_file.close()

    # Calculate and log coefficients to text file:
    coefficients = np.dot(u.transpose(), normalized_faces)  # (k x N^2) x (N^2 x M) = k x M

    log_file = open("model/list-coefficient.txt", "w")
    for i in range(m):
        file_util.log_data(coefficients[:, i], log_file)
        log_file.write('\n')
    log_file.close()

    # Calculate and save simplified-normalize-faces:
    simplified_normalize_faces = np.dot(u, coefficients)  # (N^2 x k) x (k x M) = N^2 x M
    simplified_faces = np.add(simplified_normalize_faces, mean_face_repeat)  # N^2 x M
    # print simplified_normalize_faces
    # print mean_face_repeat
    # print simplified_faces
    for i in range(m):
        filepath = 'model/image/' + files[i]
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_util.save_image(simplified_faces[:, i].reshape((height, width)), filepath)

    # Show plot
    plt.show()
