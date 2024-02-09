from __future__ import annotations

import unittest

import numpy as np

from garmi_parti.launchers import parti_haptic_sim
from garmi_parti.parti import haptic_simulation
from garmi_parti.teleoperation import containers


class TestHapticSim(unittest.TestCase):
    def test_entrypoint(self):
        parti_haptic_sim.main()

    def test_move_arms(self):
        haptic_simulation.move_arms(
            "left-hostname",
            "right-hostname",
            containers.JointPositions(np.zeros(7)),
            containers.JointPositions(np.zeros(7)),
            0.2,
        )
