from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from garmi_parti.teleoperation import containers

# Set the seed for reproducibility (optional)
rng = np.random.default_rng(42)


class TestUtils(unittest.TestCase):
    def test_TwoArmContainer(self):
        left = containers.Displacement(
            rng.random(3), R.from_euler("XYZ", rng.uniform(np.zeros(3), [np.pi] * 3))
        )
        right = containers.Displacement(
            rng.random(3), R.from_euler("XYZ", rng.uniform(np.zeros(3), [np.pi] * 3))
        )
        disp = containers.TwoArmDisplacement(left, right)
        assert left == disp.left
        assert right == disp.right
