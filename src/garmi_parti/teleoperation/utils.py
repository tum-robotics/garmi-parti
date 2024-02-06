"""
Utility module.
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import os
import time
import typing

import numpy as np
import panda_py
from panda_py import controllers, libfranka
from scipy.spatial import transform as tr


@dataclasses.dataclass
class JointPositions:
    """
    Joint positions container.
    """

    positions: np.ndarray


def _default_stiffness() -> np.ndarray:
    return np.array([600, 600, 600, 600, 250, 150, 50])


def _default_damping() -> np.ndarray:
    return np.array([50, 50, 50, 20, 20, 20, 10])


def _default_position() -> np.ndarray:
    return np.zeros(3)


@dataclasses.dataclass
class TeleopParams:
    """
    Parameters for a teleoperator.
    """

    hostname: str
    transform: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)
    q_idle: JointPositions | None = None
    q_teleop: JointPositions | None = None
    lock_translation: bool = False
    lock_rotation: bool = False
    # Joint parameters
    stiffness: np.ndarray = dataclasses.field(default_factory=_default_stiffness)
    damping: np.ndarray = dataclasses.field(default_factory=_default_damping)
    # Cartesian parameters
    linear_stiffness: float = 400
    angular_stiffness: float = 20
    nullspace_stiffness: float = 0.5
    damping_ratio: float = 1.0
    # Filter parameters
    filter_coeff: float = 1.0
    gain_force: float = 0.4
    gain_torque: float = 0.0
    gain_joint_torque: float = 0.8
    speed_factor: float = 0.2


@dataclasses.dataclass
class TeleopContainer:
    """
    Container utility class for multi-arm teleoperation.
    """

    arm: panda_py.Panda
    gripper: libfranka.Gripper
    controller: controllers.TorqueController = None
    params: TeleopParams = dataclasses.field(
        default_factory=lambda: TeleopParams("localhost")
    )
    position_init: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    orientation_init: tr.Rotation = dataclasses.field(
        default_factory=tr.Rotation.identity
    )
    orientation_init_inv: tr.Rotation = dataclasses.field(
        default_factory=tr.Rotation.identity
    )
    transform: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)
    transform_inv: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)

    def reinitialize(self) -> None:
        """
        Reinitialize the reference pose to the current robot pose.
        """
        self.position_init = self.arm.get_position()
        self.orientation_init = self.get_rotation()
        self.orientation_init_inv = self.orientation_init.inv()

    def get_rotation(self) -> tr.Rotation:
        """
        Get a `scipy.spatial.tr.Rotation` from the robot's current
        end effector orientation.

        Returns:
          scipy.spatial.tr.Rotation: Unit quaternion representing the
            the robot end effector's orientation.
        """
        return tr.Rotation.from_quat(self.arm.get_orientation(scalar_first=False))


@dataclasses.dataclass
class Displacement:
    """
    Container holding a relative pose, i.e. SE3 displacement.
    """

    linear: np.ndarray = dataclasses.field(default_factory=_default_position)
    angular: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)


@dataclasses.dataclass
class Wrench:
    """
    Wrench container.
    """

    force: np.ndarray
    torque: np.ndarray


@dataclasses.dataclass
class JointTorques:
    """
    Joint torque container.
    """

    torques: np.ndarray


@dataclasses.dataclass
class Pose:
    """
    Pose container.
    """

    position: np.ndarray = dataclasses.field(default_factory=_default_position)
    orientation: tr.Rotation = dataclasses.field(default_factory=tr.Rotation.identity)


T = typing.TypeVar("T")


@dataclasses.dataclass
class TwoArmContainer(abc.ABC, typing.Generic[T]):
    """
    Abstract base class for two-arm containers.
    """

    left: T
    right: T


@dataclasses.dataclass
class TwoArmTeleopContainer(TwoArmContainer[TeleopContainer]):
    """
    Two arm teleoperation container.
    """


@dataclasses.dataclass
class TwoArmDisplacement(TwoArmContainer[Displacement]):
    """
    Two-arm displacement container.
    """


@dataclasses.dataclass
class TwoArmWrench(TwoArmContainer[Wrench]):
    """
    Two-arm wrench container.
    """


@dataclasses.dataclass
class TwoArmJointPositions(TwoArmContainer[typing.Optional[JointPositions]]):
    """
    Two-arm joint positions container.
    """


@dataclasses.dataclass
class TwoArmPose(TwoArmContainer[Pose]):
    """
    Two-arm pose container.
    """


@dataclasses.dataclass
class TwoArmJointTorques(TwoArmContainer[JointTorques]):
    """
    Two-arm joint torques container.
    """


class TeleopTimeoutError(RuntimeError):
    """
    Teleoperation network timeout error.
    """


class Timer:
    """
    Utility class to track timeouts.
    """

    def __init__(self, timestep: float, timeout: float, buffer_size: int = 100):
        self.timestep = timestep
        self.timeout = timeout
        self.last_tick_time: float | None = None
        self.buffer: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.last_sleep_time: float = 0

    def tick(self) -> None:
        """
        Tick the internal clock.
        This function will keep track of the time between calls
        and block if necessary to keep the required timestep interval.
        """
        current_time = time.perf_counter()

        if self.last_tick_time is not None:
            time_delta = current_time - self.last_tick_time
            self.buffer.append(time_delta - self.last_sleep_time)
            average_time_delta = (
                sum(self.buffer) / len(self.buffer) if self.buffer else 0.0
            )
            sleep_time = max(0, self.timestep - average_time_delta)
            self.last_sleep_time = sleep_time
            time.sleep(sleep_time)

        self.last_tick_time = current_time

    def check_timeout(self) -> None:
        """
        Check whether a timeout occurred since the last call to `tick`.
        """
        current_time = time.perf_counter()

        if self.last_tick_time is not None:
            time_since_last_tick = current_time - self.last_tick_time
            if time_since_last_tick > self.timeout:
                raise TeleopTimeoutError()


def compute_displacement(container: TeleopContainer) -> Displacement:
    """
    Compute the displacement between the teleoperator's
    initial and its current pose.

    Args:
      container: The teleoperator container.

    Returns:
      Displacement: Container holding the relative pose.
    """
    return Displacement(
        linear=container.transform.apply(
            container.arm.get_position() - container.position_init
        ),
        angular=container.transform
        * container.get_rotation()
        * container.orientation_init_inv
        * container.transform_inv,
    )


def get_robot_hostnames(required: bool = True) -> tuple[str, str]:
    """
    Retrieve the left and right robot ips (hostnames) from
    the environment variables `PANDA_LEFT` and `PANDA_RIGHT`.
    """
    left, right = os.environ.get("PANDA_LEFT"), os.environ.get("PANDA_RIGHT")
    if required and (left is None or right is None):
        raise RuntimeError(
            "Please make sure the environment variables "
            + "PANDA_LEFT and PANDA_RIGHT are set to the respective robot hostnames."
        )
    return left, right  # type: ignore[return-value]
