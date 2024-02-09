"""
Model-mediated teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

import typing

import numpy as np

from .. import garmi


class Follower(garmi.JointFollower):
    """
    Use GARMI or a similar system as a model-mediated teleoperation follower device.
    """

    stiffness: typing.ClassVar[np.ndarray] = [600, 600, 600, 600, 250, 150, 50]
    damping: typing.ClassVar[np.ndarray] = [50, 50, 50, 20, 20, 20, 10]
    filter_coeff = 1.0

    def pause(self) -> None:
        pass

    def unpause(self) -> None:
        pass
