from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from garmi_parti.teleoperation import utils

# Set the seed for reproducibility (optional)
rng = np.random.default_rng(42)


class TestUtils(unittest.TestCase):
    def test_TwoArmContainer(self):
        left = utils.Displacement(
            rng.random(3), R.from_euler("XYZ", rng.uniform(np.zeros(3), [np.pi] * 3))
        )
        right = utils.Displacement(
            rng.random(3), R.from_euler("XYZ", rng.uniform(np.zeros(3), [np.pi] * 3))
        )
        disp = utils.TwoArmDisplacement(left, right)
        assert left == disp.left
        assert right == disp.right
