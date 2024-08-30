"""
Teleoperation interfaces for the Garmi robot.
"""

from __future__ import annotations

import pickle

from .. import panda
from ..teleoperation import containers, interfaces


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

    # pylint: disable=duplicate-code
    def pause(self, end_effector: str = "") -> None:
        if end_effector in ("left", ""):
            self.left.arm.stop_controller()
        if end_effector in ("right", ""):
            self.right.arm.stop_controller()

    def unpause(self, end_effector: str = "") -> None:
        if end_effector in ("left", ""):
            self._start_teleop(self.left)
        if end_effector in ("right", ""):
            self._start_teleop(self.right)


class JointFollower(panda.JointFollower, interfaces.TwoArmPandaInterface):
    """
    Teleoperation interface that controls the two arms of Garmi
    in joint space.
    """

    # pylint: disable=duplicate-code
    def get_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointStates(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_states: containers.TwoArmJointStates = pickle.loads(command)
        if joint_states.left is not None:
            self._set_command(joint_states.left, self.left)
        if joint_states.right is not None:
            self._set_command(joint_states.right, self.right)

    def pause(self, end_effector: str = "") -> None:
        if end_effector in ("left", ""):
            self.left.arm.stop_controller()
        if end_effector in ("right", ""):
            self.right.arm.stop_controller()

    def pre_teleop(self) -> bool:
        self._pre_teleop(self.left)
        self._pre_teleop(self.right)
        return True

    def unpause(self, end_effector: str = "") -> None:
        if end_effector in ("left", ""):
            self._start_teleop(self.left)
        if end_effector in ("right", ""):
            self._start_teleop(self.right)

    def set_sync_command(self, command: bytes, end_effector: str = "") -> None:
        joint_positions: containers.TwoArmJointPositions = pickle.loads(command)
        if end_effector == "":
            self.move_arms(joint_positions)
        if end_effector == "left" and joint_positions.left is not None:
            self.left.arm.move_to_joint_position(
                joint_positions.left.positions, self.left.params.speed_factor
            )
        if end_effector == "right" and joint_positions.right is not None:
            self.right.arm.move_to_joint_position(
                joint_positions.right.positions, self.right.params.speed_factor
            )
