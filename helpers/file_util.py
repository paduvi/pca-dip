from os import listdir
from os.path import isfile, join
import numpy as np


def listFile(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles


def log_data(data, logger):
    if type(data) in [list, tuple, np.ndarray]:
        for value in data:
            log_data(value, logger)
        logger.write(';')
    else:
        logger.write('\t' + str(data))
