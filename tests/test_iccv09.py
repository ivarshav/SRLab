from scipy.optimize import curve_fit

from .fixtures import *

NOISE = 100


def test_lstsq_default(kernel):
    from random import randint
    high_lines = np.ndarray((25, 25))
    for i in xrange(len(high_lines)):
        for j in xrange(len(high_lines[0])):
            high_lines[i][j] = randint(0, 255)

    flatten_kernel = kernel.ravel()
    low_lines = np.dot(high_lines, flatten_kernel)
    unnormalized_kernel = np.linalg.lstsq(high_lines, low_lines)[0]
    k = unnormalized_kernel / np.sum(unnormalized_kernel)
    k = np.reshape(k, (5, 5))
    np.testing.assert_array_almost_equal(kernel, k)


def test_lstsq(high_lines, kernel):
    flatten_kernel = kernel.ravel()
    low_lines = np.dot(high_lines, flatten_kernel)
    unnormalized_kernel = np.linalg.lstsq(high_lines, low_lines)[0]
    k = unnormalized_kernel / np.sum(unnormalized_kernel)
    k = np.reshape(k, (5, 5))
    np.testing.assert_array_almost_equal(kernel, k)


def test_fit_default(twoD_kernel, x_y, sigma_x, sigma_y, theta):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel(), p0=[sigma_x, sigma_y, theta])
    np.testing.assert_array_equal(popt, [sigma_x, sigma_y, theta])


def test_fit_parameters_with_noise(twoD_kernel, x_y, sigma_x, sigma_y, theta, noise=NOISE):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel(),
                           p0=[sigma_x + noise, sigma_y + noise, theta + noise])
    new_kernel = sr_image_util.twoD_gaussian((x, y), *popt)
    np.testing.assert_array_almost_equal(twoD_kernel, new_kernel)


def test_fit_kernel_with_noise(twoD_kernel, x_y, sigma_x, sigma_y, theta, noise=NOISE):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel() + noise,
                           p0=[sigma_x, sigma_y, theta])
    new_kernel = sr_image_util.twoD_gaussian((x, y), *popt)
    np.testing.assert_array_almost_equal(twoD_kernel, new_kernel)
    np.testing.assert_array_almost_equal(popt, [sigma_x, sigma_y, theta])
