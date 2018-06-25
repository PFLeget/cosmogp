"""compute mean and interpolate mean for GP."""

import scipy.interpolate as inter
import numpy as np


def interpolate_mean_1d(old_binning, mean_function, new_binning):
    """
    Interpoalte 1d function.

    Function to interpolate 1D mean function on the new grid
    Interpolation is done using cubic spline from scipy

    old_binning : 1D numpy array or 1D list. Represent the
    binning of the mean function on the original grid. Should not
    be sparce. For SNIa it would be the phases of the Mean function

    mean_function : 1D numpy array or 1D list. The mean function
    used inside Gaussian Process, observed at the Old binning. Would
    be the average Light curve for SNIa.

    new_binning : 1D numpy array or 1D list. The new grid where you
    want to project your mean function. For example, it will be the
    observed SNIa phases.

    output : mean_interpolate, Mean function on the new grid (New_binning)
    """
    cubic_spline = inter.InterpolatedUnivariateSpline(old_binning,
                                                      mean_function)

    mean_interpolate = cubic_spline(new_binning)

    return mean_interpolate


def interpolate_mean_2d(old_binning, mean_function, new_binning):
    """
    Interpolate 2d function.

    Function to interpolate 2D mean function on the new grid
    Interpolation is done using cubic spline from scipy.

    old_binning : 2D numpy array or 2D list. Represent the
    binning of the mean function on the original grid. Should not
    be sparce. For Weak-lensing it would be the pixel coordinqtes
    for the Mean function.

    mean_function : 2D numpy array or 2D list. The mean function
    used inside Gaussian Process, observed at the Old binning. Would
    be the average value of PSF size for example in Weak-lensing.

    new_binning : 2D numpy array or 2D list. The new grid where you
    want to project your mean function. For example, it will be the
    galaxy position for Weak-Lensing.

    output : mean_interpolate,  Mean function on the new grid (New_binning)
    """
    tck = inter.bisplrep(old_binning[:, 0], old_binning[:, 1],
                         mean_function, task=1)

    mean_interpolate = np.zeros(len(new_binning))

    for i in range(len(new_binning)):
        mean_interpolate[i] = inter.bisplev(new_binning[i, 0],
                                            new_binning[i, 1], tck)

    return mean_interpolate


def return_mean(y, x, new_x=None, mean_y=None, mean_xaxis=None, diff=None):
    """
    Substract the mean function.

    in order to avoid systematic difference between
    average function and the data
    """
    if mean_y is not None:
        assert mean_xaxis is not None, 'you should provide an x axis for the average'
        assert len(mean_y) == len(mean_xaxis), 'mean_y and mean_xaxis should have the same len'

        if type(x[0]) is np.float64:
            mean_y_shape = interpolate_mean_1d(mean_xaxis, mean_y, x)
        else:
            mean_y_shape = interpolate_mean_2d(mean_xaxis, mean_y, x)
    else:
        mean_y_shape = 0

    if diff is None:
        diff = np.mean(y - mean_y_shape)

    y0 = mean_y_shape + diff

    if new_x is not None:
        if mean_y is not None:
            if type(x[0]) is np.float64:
                mean_y_shape = interpolate_mean_1d(mean_xaxis, mean_y, new_x)
            else:
                mean_y_shape = interpolate_mean_2d(mean_xaxis, mean_y, new_x)
            new_y0 = mean_y_shape + diff
        else:
            new_y0 = y0
        return new_y0
    else:
        return y0
