import numpy as np
import pytest
from PIL import Image

from sr_factory.sr_image_factory import SRImageFactory
from sr_util import sr_image_util
from sr_util.sr_image_util import gaussian_kernel

RADIUS = 2


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
    high_res_patches, dc_patches = sr_image_util.get_patches_from(sr_image, interval=4)
    high_res_patches = high_res_patches + dc_patches
    start = 550
    return high_res_patches[start:start + 200]


@pytest.fixture
def kernel():
    return gaussian_kernel()


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
