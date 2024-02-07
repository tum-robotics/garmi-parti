"""
Demos running on the PARTI system.
"""

from __future__ import annotations

import argparse
import collections
import logging
import pickle
import time

import numpy as np

# from pypartigp.zmq import Publisher
from scipy.spatial import transform

from . import panda
from .peripherals import gamepad
from .teleoperation import client, interface, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti")

Q_IDLE_LEFT = utils.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_IDLE_RIGHT = Q_IDLE_LEFT
Q_TELEOP_LEFT = utils.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = utils.JointPositions([-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84])

LEFT_TRANSFORM = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, -90 / 180 * np.pi]
)
RIGHT_TRANSFORM = transform.Rotation.from_euler(
    "XYZ", [0, 90 / 180 * np.pi, 90 / 180 * np.pi]
)

DAMPING = [80, 0, 0, 0, 0, 0, 0]


class Tickable:
    """
    Tickable class that can be mixed in to track runtime.
    """

    def __init__(self, window_size: int = 1000, interval: int = 2000) -> None:
        self._t_last = time.perf_counter()
        self._t_window: collections.deque[float] = collections.deque(maxlen=window_size)
        self.num_ticks = 0
        self.interval = interval

    def tick(self) -> None:
        """
        Reset the internal time since the last tick was called.
        """
        t_curr = time.perf_counter()
        self._t_window.append(t_curr - self._t_last)
        self._t_last = t_curr
        self.num_ticks += 1

        if self.num_ticks % self.interval == 0:
            _logger.debug("Current frequency: %fHz", 1 / np.average(self._t_window))


class CartesianLeader(panda.CartesianLeader, interface.TwoArmPandaInterface, Tickable):
    """
    Use PARTI or a similar system as a cartesian teleoperation leader device.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        left_hostname: str,
        right_hostname: str,
        window_size: int = 1000,
        interval: int = 2000,
    ) -> None:
        Tickable.__init__(self, window_size, interval)  # type: ignore[call-arg]

        self.paused = False

        q_idle = utils.TwoArmJointPositions(left=Q_IDLE_LEFT, right=Q_IDLE_RIGHT)
        q_teleop = utils.TwoArmJointPositions(left=Q_TELEOP_LEFT, right=Q_TELEOP_RIGHT)

        left_transform = LEFT_TRANSFORM
        right_transform = RIGHT_TRANSFORM

        interface.TwoArmPandaInterface.__init__(
            self,
            utils.TeleopParams(
                left_hostname,
                left_transform,
                q_idle=q_idle.left,
                q_teleop=q_teleop.left,
                damping=DAMPING,
            ),
            utils.TeleopParams(
                right_hostname,
                right_transform,
                q_idle=q_idle.right,
                q_teleop=q_teleop.right,
                damping=DAMPING,
            ),
        )

    def get_command(self) -> bytes:
        if self.paused:
            return pickle.dumps(
                utils.TwoArmDisplacement(utils.Displacement(), utils.Displacement())
            )
        displacement = utils.TwoArmDisplacement(
            left=utils.compute_displacement(self.left),
            right=utils.compute_displacement(self.right),
        )
        return pickle.dumps(displacement)

    def set_command(self, command: bytes) -> None:
        wrench: utils.TwoArmWrench = pickle.loads(command)
        self._set_command(self.left, wrench.left)
        self._set_command(self.right, wrench.right)
        self.tick()

    def unpause(self) -> None:
        self.left.reinitialize()
        self.right.reinitialize()
        self.paused = False


class JointLeader(panda.JointLeader, interface.TwoArmPandaInterface, Tickable):
    """
    Use PARTI or a similar system as a joint-space teleoperation leader device.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        left_hostname: str,
        right_hostname: str,
        window_size: int = 1000,
        interval: int = 2000,
    ) -> None:
        Tickable.__init__(self, window_size, interval)  # type: ignore[call-arg]

        q_idle = utils.TwoArmJointPositions(left=Q_IDLE_LEFT, right=Q_IDLE_RIGHT)
        q_teleop = utils.TwoArmJointPositions(left=Q_TELEOP_LEFT, right=Q_TELEOP_RIGHT)

        left_transform = LEFT_TRANSFORM
        right_transform = RIGHT_TRANSFORM

        interface.TwoArmPandaInterface.__init__(
            self,
            utils.TeleopParams(
                left_hostname,
                left_transform,
                q_idle=q_idle.left,
                q_teleop=q_teleop.left,
                damping=DAMPING,
            ),
            utils.TeleopParams(
                right_hostname,
                right_transform,
                q_idle=q_idle.right,
                q_teleop=q_teleop.right,
                damping=DAMPING,
            ),
        )

    def get_command(self) -> bytes:
        return pickle.dumps(
            utils.TwoArmJointPositions(
                left=self._get_command(self.left), right=self._get_command(self.right)
            )
        )

    def set_command(self, command: bytes) -> None:
        joint_torques: utils.TwoArmJointTorques = pickle.loads(command)
        self._set_command(self.left, joint_torques.left)
        self._set_command(self.right, joint_torques.right)
        self.tick()

    def pause(self) -> None:
        self.left.arm.stop_controller()
        self.right.arm.stop_controller()

    def get_sync_command(self) -> bytes:
        return self.get_command()


def teleop() -> None:
    """
    PARTI teleoperation demo.
    The PARTI system acts as a network client and teleoperation
    leader that connects to a teleoperation server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=13701)
    parser.add_argument("-gp", "--gamepad-port", type=int, default=13702)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["joint", "cartesian"],
        default="cartesian",
        help="Specify teleoperation mode",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    left, right = utils.get_robot_hostnames()
    leader: interface.TwoArmPandaInterface
    if args.mode == "joint":
        leader = JointLeader(left, right)
    elif args.mode == "cartesian":
        leader = CartesianLeader(left, right)

    cli = client.Client(leader, args.host, args.port)
    # gp_publisher = Publisher(args.gamepad_port, 30)
    gamepad_handle = gamepad.GamepadHandle(cli, "localhost", args.gamepad_port)
    logger = interface.TwoArmLogger(leader)
    client.user_interface(cli)
    cli.shutdown()
    gamepad_handle.stop()
    # gp_publisher.stop()
    logger.stop()
