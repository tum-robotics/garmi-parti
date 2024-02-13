"""
Model-mediated teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

from .. import garmi


class Follower(garmi.JointFollower):
    """
    Use GARMI or a similar system as a model-mediated teleoperation follower device.
    """

    def pause(self) -> None:
        pass

    def unpause(self) -> None:
        pass
