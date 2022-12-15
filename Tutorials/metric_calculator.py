import numpy as np


def calculate_mse(x,y,ppm):
    # region of interest: 1.5ppm - 4ppm
    min_ind = np.argmin(ppm<=4)
    max_ind = np.argmax(ppm>=1.5)

    x_crop = x[:,min_ind:max_ind]
    y_crop = y[:,min_ind:max_ind]

    return np.sum(np.square(y_crop-x_crop)).mean(axis=1).mean()

def calculate_mse(x,y,ppm):
    # region of interest: 1.5ppm - 4ppm
    min_ind = np.argmin(ppm<=4)
    max_ind = np.argmax(ppm>=1.5)

    x_crop = x[:,min_ind:max_ind]
    y_crop = y[:,min_ind:max_ind]

    return np.sum(np.square(y_crop-x_crop)).mean(axis=1).mean()