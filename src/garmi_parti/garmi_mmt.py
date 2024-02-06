"""
Demos running on the GARMI system.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
from scipy.spatial import transform

from . import garmi
from .teleoperation import server, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("garmi")


class Follower(garmi.JointFollower):
    """
    Use GARMI or a similar system as a model-mediated teleoperation follower device.
    """

    stiffness = [600, 600, 600, 600, 250, 150, 50]
    damping = [50, 50, 50, 20, 20, 20, 10]
    filter_coeff = 1.0

    def pause(self) -> None:
        pass

    def unpause(self) -> None:
        pass


def teleop() -> None:
    """
    GARMI teleoperation demo.
    The GARMI system acts as a network server and teleoperation
    follower that accepts connections from teleoperation clients.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=13701)
    args = parser.parse_args()

    left, right = os.environ.get("PANDA_LEFT"), os.environ.get("PANDA_RIGHT")
    if left is None or right is None:
        raise RuntimeError(
            "Please make sure the environment variables "
            + "PANDA_LEFT and PANDA_RIGHT are set to the respective robot hostnames."
        )
    left_params = utils.TeleopParams(
        left,
        transform.Rotation.from_euler(
            "XYZ", [0, 90 / 180 * np.pi, -90 / 180 * np.pi]
        ).inv(),
    )
    right_params = utils.TeleopParams(
        right,
        transform.Rotation.from_euler(
            "XYZ", [0, 90 / 180 * np.pi, 90 / 180 * np.pi]
        ).inv(),
    )
    srv = server.Server(Follower(left_params, right_params, True, True), args.port)
    server.user_interface(srv)
    srv.shutdown()
