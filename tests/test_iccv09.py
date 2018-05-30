import numpy as np
import pytest
from PIL import Image
from scipy.optimize import curve_fit

from sr_factory.sr_image_factory import SRImageFactory
from sr_util import sr_image_util
from sr_util.sr_image_util import gaussian_kernel

RADIUS = 2
NOISE = 5


@pytest.fixture
def sr_image():
    image = Image.open("test_data/letter.png")
    sr_image = SRImageFactory.create_sr_image(image)
    return sr_image


@pytest.fixture
def high_lines(sr_image):
    """

    :param sr_image:
    :return: 25 by 25 array
    """
    high_res_patches, _ = sr_image_util.get_patches_from(sr_image, interval=4)
    start = 550
    return high_res_patches[start:start + 25]


@pytest.fixture
def kernel():
    return gaussian_kernel()


def test_lstsq(high_lines, kernel):
    flatten_kernel = kernel.ravel()
    low_lines = np.dot(high_lines, flatten_kernel)
    unnormalized_kernel = np.linalg.lstsq(high_lines, low_lines)[0]
    k = unnormalized_kernel / np.sum(unnormalized_kernel)
    k = np.reshape(k, (5, 5))
    print kernel
    print k
    np.testing.assert_array_almost_equal(kernel, k)


@pytest.fixture
def sigma_x():
    return 1.0


@pytest.fixture
def sigma_y():
    return 1.0


@pytest.fixture
def theta():
    return 15


@pytest.fixture
def x_y(radius=RADIUS):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x, y


@pytest.fixture
def twoD_kernel(x_y, sigma_x, sigma_y, theta):
    return sr_image_util.twoD_gaussian(x_y, sigma_x, sigma_y, theta)


def test_fit_dafault(twoD_kernel, x_y, sigma_x, sigma_y, theta):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel(), p0=[sigma_x, sigma_y, theta])
    np.testing.assert_array_equal(popt, [sigma_x, sigma_y, theta])


def test_fit_parameters_with_noise(twoD_kernel, x_y, sigma_x, sigma_y, theta, noise=NOISE):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel(),
                           p0=[sigma_x + noise, sigma_y + noise, theta + noise])
    print popt
    new_kernel = sr_image_util.twoD_gaussian((x, y), *popt)
    np.testing.assert_array_almost_equal(twoD_kernel, new_kernel)
    np.testing.assert_array_almost_equal(popt, [sigma_x, sigma_y, theta])


def test_fit_kernel_with_noise(twoD_kernel, x_y, sigma_x, sigma_y, theta, noise=NOISE):
    x, y = x_y
    xdata = np.vstack((x.ravel(), y.ravel()))
    popt, pcov = curve_fit(sr_image_util.twoD_gaussian, xdata, ydata=twoD_kernel.ravel() * noise,
                           p0=[sigma_x, sigma_y, theta])
    print popt
    new_kernel = sr_image_util.twoD_gaussian((x, y), *popt)
    np.testing.assert_array_almost_equal(twoD_kernel, new_kernel)
    np.testing.assert_array_almost_equal(popt, [sigma_x, sigma_y, theta])
