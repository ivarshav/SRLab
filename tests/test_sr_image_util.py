from .fixtures import *


def test_create_gaussian(twoD_kernel, asymmetric_kernel):
    np.testing.assert_array_equal(twoD_kernel, asymmetric_kernel.ravel())
