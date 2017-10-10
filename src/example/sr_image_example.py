from PIL import Image
from sr_factory.sr_image_factory import SRImageFactory


def letter_example():
    image = Image.open("../test_data/letter.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(3, 'iccv09')
    reconstructed_sr_image.save("../test_data/letter_3x.png", "png")


def babyface_example():
    image = Image.open("../test_data/babyface_4.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/babyface_2x.png", "png")


def monarch_example():
    image = Image.open("../test_data/monarch.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/monarch_2x.png", "png")


def temple_example():
    image = Image.open("../test_data/temple.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/temple_2x.png", "png")


def chip_example():
    image = Image.open("../test_data/chip.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/chip_2x.png", "png")


def koala_example():
    image = Image.open("../test_data/koala.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/koala_2x.png", "png")


def old_man_example():
    image = Image.open("../test_data/old_man.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/old_man_2x.png", "png")


def sculpture_example():
    image = Image.open("../test_data/sculpture.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/sculpture_2x.png", "png")


def flowerfield_example():
    image = Image.open("../test_data/flowerfield.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/flowerfield_2x.png", "png")


def kitchen_example():
    image = Image.open("../test_data/kitchen.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/kitchen_2x.png", "png")


def auditorium_example():
    image = Image.open("../test_data/auditorium.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/auditorium_2x.png", "png")


def butterfly_example():
    image = Image.open("../test_data/butterfly.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/butterfly_2x.png", "png")


def colorblind_example():
    image = Image.open("../test_data/colorblind.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/colorblind_2x.png", "png")


def freckles2_example():
    image = Image.open("../test_data/freckles2.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/freckles2_2x.png", "png")


def kitchen2_example():
    image = Image.open("../test_data/kitchen2.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/kitchen2_2x.png", "png")


def street3_example():
    image = Image.open("../test_data/street3.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/street3_2x.png", "png")


def hands_example():
    image = Image.open("../test_data/hands.png")
    sr_image = SRImageFactory.create_sr_image(image)
    reconstructed_sr_image = sr_image.reconstruct(2, 'iccv09')
    reconstructed_sr_image.save("../test_data/hands_2x.png", "png")


examples = [temple_example, chip_example, koala_example, old_man_example, sculpture_example, flowerfield_example,
            kitchen_example, auditorium_example, butterfly_example, colorblind_example, freckles2_example,
            kitchen2_example, street3_example]

if __name__ == '__main__':
    # hands_example()
    letter_example()
    # babyface_example()
    # monarch_example()
    # colorblind_example()
    # for ex in examples:
    #     try:
    #         ex()
    #     except Exception:
    #         pass
    #     else:
    #         print ex
