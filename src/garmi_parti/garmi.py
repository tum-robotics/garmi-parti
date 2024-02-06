"""
Demos running on the GARMI system.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import typing

import numpy as np
import panda_py
from scipy.spatial import transform

from . import panda
from .teleoperation import interface, server, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("garmi")

Q_IDLE_LEFT = utils.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_IDLE_RIGHT = Q_IDLE_LEFT
Q_TELEOP_LEFT = utils.JointPositions(
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
Q_TELEOP_RIGHT = utils.JointPositions(
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


class CartesianFollower(panda.CartesianFollower, interface.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in Cartesian space.
    """

    def get_command(self) -> bytes:
        return pickle.dumps(
            utils.TwoArmWrench(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        displacement: utils.TwoArmDisplacement = pickle.loads(command)
        self._set_command(displacement.left, self.left)
        self._set_command(displacement.right, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()


class JointFollower(panda.JointFollower, interface.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in joint space.
    """

    stiffness: typing.ClassVar[np.ndarray] = [600, 600, 600, 600, 250, 150, 50]
    damping: typing.ClassVar[np.ndarray] = [50, 50, 50, 20, 20, 20, 10]
    filter_coeff = 1.0

    def get_command(self) -> bytes:
        return pickle.dumps(
            utils.TwoArmJointTorques(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_positions: utils.TwoArmJointPositions = pickle.loads(command)
        if joint_positions.left is not None:
            self._set_command(joint_positions.left, self.left)
        if joint_positions.right is not None:
            self._set_command(joint_positions.right, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()

    def set_sync_command(self, command: bytes) -> None:
        self.move_arms(pickle.loads(command))


class OneArmCartesianFollower(CartesianFollower):
    """
    Teleoperation interface that controls one of the arms of Garmi
    in Cartesian space.
    """

    def __init__(
        self,
        side: typing.Literal["left", "right"],
        left: utils.TeleopParams,
        right: utils.TeleopParams,
        has_left_gripper: bool = False,
        has_right_gripper: bool = False,
    ) -> None:
        self.side = side
        super().__init__(left, right, has_left_gripper, has_right_gripper)

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(getattr(self, self.side)))

    def set_command(self, command: bytes) -> None:
        displacement: utils.Displacement = pickle.loads(command)
        if self.side == "left":
            self._set_command(displacement, self.left)
            self._set_command(utils.compute_displacement(self.right), self.right)
        elif self.side == "right":
            self._set_command(utils.compute_displacement(self.left), self.left)
            self._set_command(displacement, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()


class OneArmJointFollower(JointFollower):
    """
    Teleoperation interface that controls one of the arms of Garmi
    in joint space.
    """

    def __init__(
        self,
        side: typing.Literal["left", "right"],
        left: utils.TeleopParams,
        right: utils.TeleopParams,
        has_left_gripper: bool = False,
        has_right_gripper: bool = False,
    ) -> None:
        self.side = side
        super().__init__(left, right, has_left_gripper, has_right_gripper)

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(getattr(self, self.side)))

    def set_command(self, command: bytes) -> None:
        joint_positions: utils.JointPositions = pickle.loads(command)
        if self.side == "left":
            self._set_command(joint_positions, self.left)
            self._set_command(
                utils.JointPositions(self.right.arm.get_state().q), self.right
            )
        elif self.side == "right":
            self._set_command(
                utils.JointPositions(self.left.arm.get_state().q), self.left
            )
            self._set_command(joint_positions, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()

    def set_sync_command(self, command: bytes) -> None:
        arm: panda_py.Panda = getattr(self, self.side).arm
        arm.move_to_joint_position(
            pickle.loads(command).positions, arm.params.speed_factor
        )


def teleop() -> None:
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
    left_params = utils.TeleopParams(
        left, TRANSFORM_LEFT, Q_IDLE_LEFT, Q_TELEOP_LEFT, nullspace_stiffness=10
    )
    right_params = utils.TeleopParams(
        right, TRANSFORM_RIGHT, Q_IDLE_RIGHT, Q_TELEOP_RIGHT, nullspace_stiffness=10
    )

    follower: interface.TwoArmPandaInterface
    if args.mode == "joint":
        follower = JointFollower(left_params, right_params, True, True)
    elif args.mode == "cartesian":
        follower = CartesianFollower(left_params, right_params, True, True)
    elif args.mode == "one_arm_joint":
        follower = OneArmJointFollower(args.side, left_params, right_params, True, True)

    srv = server.Server(follower, args.port)
    logger = interface.TwoArmLogger(follower)
    server.user_interface(srv)
    srv.shutdown()
    logger.stop()
