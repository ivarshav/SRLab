import math
import sys

import numpy as np

from sr_util import profiler
from src.sr_util import sr_image_util
from sr_dataset import SRDataSet

DEFAULT_RECONSTRUCT_LEVEL = 6
ALPHA = 2 ** (1.0 / 3)


class ICCV09(object):
    def __init__(self):
        self._method_type = "iccv09"
        self._kernel = None
        # self._kernel = sr_image_util.create_gaussian_kernel()

    def get_method_type(self):
        return self._method_type

    @profiler.profiler
    def reconstruct(self, ratio, sr_image):
        """
        Reconstruct the SR image by the given ratio.

        :param ratio: reconstruct ratio
        :type ratio: float
        :param sr_image: original SR image
        :type sr_image: L{sr_image.SRImage}
        :return: reconstructed SR image
        :rtype: L{sr_image.SRImage}
        """
        # import matplotlib.pyplot as plt

        sr_dataset = SRDataSet.from_sr_image(sr_image)
        reconstructed_sr_image = sr_image
        construct_level = 1  # int(math.log(ratio, ALPHA) + 0.5)
        r = ALPHA
        for level in xrange(construct_level):
            reconstructed_sr_image = self._reconstruct(r, reconstructed_sr_image, sr_dataset)
            reconstructed_sr_image = sr_image_util.back_project(reconstructed_sr_image, sr_image, 3, level + 1)
            new_sr_dataset = SRDataSet.from_sr_image(reconstructed_sr_image)
            # plt.imshow(reconstructed_sr_image)
            # plt.show()
            sr_dataset.merge(new_sr_dataset)
            sys.stdout.write("\rReconstructing %.2f%%" % (float(level + 1) /
                                                          construct_level * 100))
            sys.stdout.flush()
        return reconstructed_sr_image

    def _reconstruct(self, ratio, sr_image, sr_dataset):
        """
        Reconstruct a SRImage using the given SRDataset by the given ratio.

        :param ratio: reconstruct ratio
        :type ratio: float
        :param sr_image: original SRImage
        :type sr_image: L{sr_image.SRImage}
        :param sr_dataset:
        :type sr_dataset: L{sr_dataset.SRDataset}
        :return: reconstructed SRImage
        :rtype: L{sr_image.SRImage}
        """
        resized_sr_image = sr_image.resize(ratio)
        patches_without_dc, patches_dc = sr_image_util.get_patches_from(resized_sr_image, interval=4)
        high_res_patches_without_dc = sr_dataset.query(patches_without_dc)
        high_res_patches = high_res_patches_without_dc + patches_dc
        if self._kernel is None:
            self._kernel = sr_image_util.create_gaussian_kernel()
        high_res_data = sr_image_util.unpatchify(high_res_patches, resized_sr_image.size, self._kernel)
        self.update_kernel(high_res_patches, patches_dc)
        resized_sr_image.putdata(high_res_data)
        return resized_sr_image

    def update_kernel(self, high_res_patches, low_res_patches):
        print "hi"
        start = 550
        high_lines = high_res_patches[start:start + 25]
        low_lines = np.transpose(low_res_patches[start:start + 25][12:13])
        unnormalized_kernel = np.linalg.lstsq(high_lines, low_lines)[0]
        k = unnormalized_kernel / np.sum(unnormalized_kernel)
        print "**"
        print np.transpose(np.dot(high_lines, k))
        print np.transpose(np.dot(high_lines, unnormalized_kernel))
        print np.transpose(low_lines)
        assert (np.dot(high_lines, unnormalized_kernel) == low_lines).any(), "hmm"
        # assert np.dot(high_lines, low_lines) == k, "hmm"
        print k
        kernel = np.reshape(k, (5, 5))
