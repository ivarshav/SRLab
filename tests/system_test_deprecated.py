from tests.system_test import _create_small_image


def test_system_grass():
    pyramid = _create_small_image("test_data/grass_original.png")
    pyramid.save("./test_data/grass0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/grass_reconstructed.png", "png")


def test_system_beach():
    pyramid = _create_small_image("test_data/beach_original.jpg")
    pyramid.save("./test_data/beach0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/beach_reconstructed.png", "png")


def test_system_skyline():
    pyramid = _create_small_image("test_data/skyline_original.png")
    pyramid.save("./test_data/skyline0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/skyline_reconstructed.png", "png")


def test_system_space():
    pyramid = _create_small_image("test_data/space_original.jpg")
    pyramid.save("./test_data/space0.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/space_reconstructed.png", "png")


def test_system_space_1():
    pyramid = _create_small_image("test_data/space_original.jpg", theta=20)
    pyramid.save("./test_data/space01.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/space_reconstructed1.png", "png")


def test_system_space_2():
    pyramid = _create_small_image("test_data/space_original.jpg", sigma_y=5, theta=10)
    pyramid.save("./test_data/space02.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/space_reconstructed2.png", "png")


def test_system_space_3():
    pyramid = _create_small_image("test_data/space_original.jpg", sigma_x=10, sigma_y=3, theta=10)
    pyramid.save("./test_data/space03.png", "png")
    reconstructed_sr_image = pyramid.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("test_data/space_reconstructed3.png", "png")