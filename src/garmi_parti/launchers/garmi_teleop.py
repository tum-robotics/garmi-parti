from __future__ import annotations

import argparse
import logging

import numpy as np
from scipy.spatial import transform

from .. import garmi
from ..teleoperation import containers, interfaces, server, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("garmi")


Q_IDLE_LEFT = containers.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_IDLE_RIGHT = Q_IDLE_LEFT
Q_TELEOP_LEFT = containers.JointPositions(
    [
        -0.08769599,
        -1.25825236,
        -0.04485461,
        -1.77655956,
        0.62575622,
        1.92354307,
        2.0303398,
    ]
)
# [0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = containers.JointPositions(
    [
        0.07436611,
        -1.2718769,
        -0.01988506,
        -1.74632818,
        -0.37870186,
        1.96712763,
        -0.59191199,
    ]
)
# [-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84])

TRANSFORM_LEFT = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, -90 / 180 * np.pi]
).inv()
TRANSFORM_RIGHT = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, 90 / 180 * np.pi]
).inv()


def main() -> None:
    """
    GARMI teleoperation demo.
    The GARMI system acts as a network server and teleoperation
    follower that accepts connections from teleoperation clients.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=13701)
    parser.add_argument(
        "--mode",
        choices=["joint", "cartesian", "one_arm_joint"],
        default="cartesian",
        help="Specify teleoperation mode",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="Which arm to use in one armed teleoperation",
    )
    args = parser.parse_args()

    left, right = utils.get_robot_hostnames()
    left_params = containers.TeleopParams(
        left, TRANSFORM_LEFT, Q_IDLE_LEFT, Q_TELEOP_LEFT, nullspace_stiffness=10
    )
    right_params = containers.TeleopParams(
        right, TRANSFORM_RIGHT, Q_IDLE_RIGHT, Q_TELEOP_RIGHT, nullspace_stiffness=10
    )

    follower: interfaces.TwoArmPandaInterface
    if args.mode == "joint":
        follower = garmi.JointFollower(left_params, right_params, True, True)
    elif args.mode == "cartesian":
        follower = garmi.CartesianFollower(left_params, right_params, True, True)

    srv = server.Server(follower, args.port)
    logger = interfaces.TwoArmLogger(follower)
    server.user_interface(srv)
    srv.shutdown()
    logger.stop()
