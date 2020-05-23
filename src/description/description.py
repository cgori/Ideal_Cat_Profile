import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import cv2
import os
import collections

here = os.path.dirname(os.path.abspath(__file__))


def run(images):
    if isinstance(images, collections.Iterable):
        results = []
        for image in images:
            results.append(linear_regression(image))
        return results
    else:

        return linear_regression(images)


def linear_regression(image):
    if image.mask_crop is None:
        return
    y = np.asarray(image.mask_crop[0])
    x = np.asarray(image.mask_crop[1])

    # x = 2 - 3 * np.random.normal(0, 1, 20)
    # y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    polynomial_features = PolynomialFeatures(degree=5)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    # print(rmse)
    # print(r2)

    plt.scatter(x, y, s=5)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    # with open("array_final.csv", "wb") as f:
    #     f.write(b'x,y\n')
    #     np.savetxt(f, arr.astype(int), fmt='%i', delimiter=",")

    arr = []
    for values in zip(x, y_poly_pred):
        arr.append([values[0][0], values[1][0]])
    arr = np.asarray(arr)
    # print(arr)
    # for row in arr:
    # print(row)
    # with open("output.csv", "wb") as f:
    #     f.write(b'x,y\n')
    #     np.savetxt(f, arr.astype(float), fmt='%f', delimiter=",")
    # plt.plot(x, y_poly_pred, color='m')
    # plt.show()

    image.curve = arr
    image.coef = norm(model.coef_)
    return image


def norm(profile):
    if np.unique(profile).shape[0] == 1:
        pass
    else:
        result_array = (profile - np.min(profile)) / np.ptp(profile)
        return result_array
