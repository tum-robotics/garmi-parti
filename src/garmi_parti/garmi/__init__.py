"""
Teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

import pickle
import typing

import numpy as np
import panda_py

from .. import panda
from ..teleoperation import containers, interfaces, utils


class CartesianFollower(panda.CartesianFollower, interfaces.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in Cartesian space.
    """

    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmWrench(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        displacement: containers.TwoArmDisplacement = pickle.loads(command)
        self._set_command(displacement.left, self.left)
        self._set_command(displacement.right, self.right)

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()


class JointFollower(panda.JointFollower, interfaces.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in joint space.
    """

    stiffness: typing.ClassVar[np.ndarray] = [600, 600, 600, 600, 250, 150, 50]
    damping: typing.ClassVar[np.ndarray] = [50, 50, 50, 20, 20, 20, 10]
    filter_coeff = 1.0

    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointTorques(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_positions: containers.TwoArmJointPositions = pickle.loads(command)
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
        left: containers.TeleopParams,
        right: containers.TeleopParams,
        has_left_gripper: bool = False,
        has_right_gripper: bool = False,
    ) -> None:
        self.side = side
        super().__init__(left, right, has_left_gripper, has_right_gripper)

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(getattr(self, self.side)))

    def set_command(self, command: bytes) -> None:
        displacement: containers.Displacement = pickle.loads(command)
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
        left: containers.TeleopParams,
        right: containers.TeleopParams,
        has_left_gripper: bool = False,
        has_right_gripper: bool = False,
    ) -> None:
        self.side = side
        super().__init__(left, right, has_left_gripper, has_right_gripper)

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(getattr(self, self.side)))

    def set_command(self, command: bytes) -> None:
        joint_positions: containers.JointPositions = pickle.loads(command)
        if self.side == "left":
            self._set_command(joint_positions, self.left)
            self._set_command(
                containers.JointPositions(self.right.arm.get_state().q), self.right
            )
        elif self.side == "right":
            self._set_command(
                containers.JointPositions(self.left.arm.get_state().q), self.left
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
