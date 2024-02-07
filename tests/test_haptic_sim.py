from __future__ import annotations

import unittest

import numpy as np

from garmi_parti import parti_haptic_sim
from garmi_parti.teleoperation import utils


class TestHapticSim(unittest.TestCase):
    def test_entrypoint(self):
        parti_haptic_sim.simulate()

    def test_move_arms(self):
        parti_haptic_sim.move_arms(
            "left-hostname",
            "right-hostname",
            utils.JointPositions(np.zeros(7)),
            utils.JointPositions(np.zeros(7)),
            0.2,
        )
