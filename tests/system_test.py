import pytest
from PIL import Image

from sr_util import sr_image_util
from sr_util.kernel import Kernel
from sr_util.sr_image_util import asymmetric_gaussian_kernel
from src.sr_factory.sr_image_factory import SRImageFactory
import matplotlib.pyplot as plt


def _create_small_image(path, radius=2, sigma_x=1.0, sigma_y=3.0, theta=15):
    with open("test_data/log1.txt", "ab") as f:
        f.write("\n\n" + path.split("/")[-1] + ":\n")
        f.write("\tinput kernel:\n")
        f.write("\t\tsigma_x = " + str(sigma_x) + " sigma_y = " + str(sigma_y) + " theta = " + str(theta))
    image = Image.open(path)
    sr_image = SRImageFactory.create_sr_image(image)
    downgrade_ratio = 2
    kernel = Kernel(asymmetric_gaussian_kernel(radius, sigma_x, sigma_y, theta), radius, sigma_x, sigma_y, theta)
    print "input kernel"
    print kernel
    size = sr_image_util.create_size(sr_image.size, 1.0 / downgrade_ratio)
    downgraded_image = sr_image.downgrade(size, kernel.kernel)
    return downgraded_image


def test_system_tree():
    pyramid = _create_small_image("test_data/tree_original.png")
    pyramid.save("./test_data/tree0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/tree_reconstructed.png", "png")


def test_system_tulip():
    pyramid = _create_small_image("test_data/tulip_original.png")
    pyramid.save("./test_data/tulip0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/tulip_reconstructed.png", "png")


def test_system_bush():
    pyramid = _create_small_image("test_data/bush_original.png")
    pyramid.save("./test_data/bush0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/bush_reconstructed.png", "png")


def test_system_earth():
    pyramid = _create_small_image("test_data/earth_original.png")
    pyramid.save("./test_data/earth0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/earth_reconstructed.png", "png")


def test_system_flowerfield():
    pyramid = _create_small_image("test_data/examples/flowerfield.png")
    pyramid.save("./test_data/examples/flowerfield0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/examples/flowerfield_res.png", "png")


def test_bkl():
    # p = _create_small_image("test_data/fff.png")
    # p.save("./test_data/ffff.png", "png")
    image = Image.open("./test_data/synthetic/ff.png")
    p = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = p.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/synthetic/pp.png", "png")


@pytest.mark.parametrize("name", [
    "chip.png",
    "old_man.png",
    "sculpture.png",
    "butterfly.png",
    "freckles2.png",
    "kitchen2.png",
    "street3.png",
    "letter.png",
    "hands.png",

    # "babyface_4.png",
    # "colorblind.png",
    # "monarch.png",
    # "temple.png", #slow
    # "koala.png",
    # "kitchen.png",
    # "auditorium.png",
])
def test_system_examples(name):
    pyramid = _create_small_image("test_data/examples/{}".format(name))
    pyramid.save("./test_data/examples/{}0.png".format(name), "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/examples/{}_res.png".format(name), "png")
