import math
import sys

import numpy as np
from scipy.optimize import curve_fit

from sr_dataset import SRDataSet
from sr_util import profiler
from sr_util.kernel import Kernel
from sr_util.sr_image_util import twoD_gaussian
from src.sr_util import sr_image_util

DEFAULT_RECONSTRUCT_LEVEL = 6
ALPHA = 2 ** (1.0 / 3)


class ICCV09(object):
    def __init__(self):
        self._method_type = "iccv09"
        radius = 2
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        self._kernel = sr_image_util.create_asymmetric_gaussian_kernel(x, y, theta=5)

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
        sr_dataset = SRDataSet.from_sr_image(sr_image, kernel=self._kernel)
        reconstructed_sr_image = sr_image
        construct_level = int(math.log(ratio, ALPHA) + 0.5)
        r = ALPHA
        for level in xrange(construct_level):
            reconstructed_sr_image = self._reconstruct(r, reconstructed_sr_image, sr_dataset)
            reconstructed_sr_image = sr_image_util.back_project(reconstructed_sr_image, sr_image, 3, level + 1,
                                                                kernel_params=self._kernel)
            new_sr_dataset = SRDataSet.from_sr_image(reconstructed_sr_image, kernel=self._kernel)
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
        low_res_patches = patches_without_dc + patches_dc
        high_res_data = sr_image_util.unpatchify(high_res_patches, resized_sr_image.size, self._kernel.kernel)
        self.update_kernel(high_res_patches, low_res_patches)
        resized_sr_image.putdata(high_res_data)
        return resized_sr_image

    def pick_best(self, high_res_patches, low_res_patches):
        """
        Pick best tuple of (`high_res_patches`, `low_res_patches`) such that they maximize the sum of squared differences,
        until we get matrix rank that matches `RANK SIZE`.

        :param high_res_patches: high resolution patches from the given low resolution patches
        :type high_res_patches: L{numpy.array}
        :param low_res_patches: low resolution patches
        :type low_res_patches: L{numpy.array}
        :return: A list of best tuples
        :rtype: tuple(np.array(L{numpy.array}),np.array(L{numpy.array}))
        """
        RANK_SIZE = 25
        high_patches, low_patches = [], []
        dist_sq = np.sum((high_res_patches[:, :] - low_res_patches[:, :]) ** 2, axis=1)
        sorted_indexes = np.argsort(dist_sq)
        for i, index in enumerate(sorted_indexes):
            if np.linalg.matrix_rank(high_patches) >= RANK_SIZE and np.linalg.matrix_rank(low_patches) >= RANK_SIZE:
                break
            high_patches.append(high_res_patches[index])
            low_patches.append(low_res_patches[index])
        return np.array(high_patches), np.array(low_patches)

    def update_kernel(self, high_res_patches, low_res_patches):
        """
        Update kernel using lstsq and curve_fit, in order to get closer to the real kernel of the image.

        :param high_res_patches: high resolution patches from the given low resolution patches
        :type high_res_patches: L{numpy.array}
        :param low_res_patches: low resolution patches
        :type low_res_patches: L{numpy.array}
        """
        print "hi"
        start = 550
        diff = 75
        high_lines = high_res_patches[start:start + diff]
        low_lines = np.transpose(np.transpose(low_res_patches[start:start + diff])[12:13])
        unnormalized_kernel = np.linalg.lstsq(high_lines, low_lines)[0]
        k = unnormalized_kernel / np.sum(unnormalized_kernel)

        radius = self._kernel.radius
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]

        xdata = np.vstack((x.ravel(), y.ravel()))
        popt, pcov = curve_fit(twoD_gaussian, xdata, ydata=k.ravel(),
                               p0=[self._kernel.sigma_x, self._kernel.sigma_y, self._kernel.theta])
        fitted_kernel = twoD_gaussian((x, y), *popt)

        kernel = np.reshape(fitted_kernel, (5, 5))
        self._kernel = Kernel(kernel, radius, *popt)
