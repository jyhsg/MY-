"""Check the geometric construction against synthetic projective ground truth.

No photographed object's real height is known; these checks validate formulas,
not the scene's physical scale or the equal-height assumption.
"""
import unittest

import numpy as np

from analyze import (
    hpoint, join, meet, normalize_point, project_top_to_vertical,
    projective_height_ratio, reconstruct_equal_height_base,
    transfer_reference_height,
)


class GeometryChecks(unittest.TestCase):
    def setUp(self):
        # Ground is Z=0; height is world Z. This projective camera has finite Vz
        # and a ground direction at infinity (an ordinary case, not an error).
        self.camera = np.array([[800., 512., 30., 0.],
                                [10., 340., -820., 0.],
                                [0., 1., 0.2, 8.]])
        self.vz = normalize_point(self.camera[:, 2])
        self.horizon = join(self.camera[:, 0], self.camera[:, 1])

    def project(self, xyz):
        return normalize_point(self.camera @ np.array([*xyz, 1.]))

    def test_height_ratio_with_finite_vertical_vp(self):
        person_b = self.project((2., 5., 0.))
        person_t = self.project((2., 5., 1.7))
        lamp_b = self.project((-3., 10., 0.))
        lamp_t = self.project((-3., 10., 4.6))
        equal_t, _ = transfer_reference_height(person_t, person_b, lamp_b,
                                                self.horizon, self.vz)
        self.assertTrue(np.allclose(equal_t, self.project((-3., 10., 1.7))))
        self.assertAlmostEqual(projective_height_ratio(lamp_b, lamp_t, equal_t,
                                                       self.vz) * 1.7, 4.6, places=10)

    def test_equal_height_hidden_base_with_infinite_ground_vp(self):
        ref_b = self.project((3., 5., 0.))
        ref_t = self.project((3., 5., 4.6))
        target_t = self.project((-2., 5., 4.6))
        recovered, ground_vp = reconstruct_equal_height_base(ref_t, ref_b,
                                    target_t, self.horizon, self.vz)
        self.assertAlmostEqual(ground_vp[2], 0., places=10)
        self.assertTrue(np.allclose(recovered, self.project((-2., 5., 0.))))

    def test_top_projection_and_infinite_vertical_ratio(self):
        b = hpoint((10., 100.))
        vz = np.array([0., 1., 0.])
        corrected = project_top_to_vertical(hpoint((13., 60.)), b, vz)
        self.assertTrue(np.allclose(corrected, hpoint((10., 60.))))
        self.assertAlmostEqual(projective_height_ratio(b, corrected,
                                                       hpoint((10., 80.)), vz), 2.)


if __name__ == "__main__":
    unittest.main()
