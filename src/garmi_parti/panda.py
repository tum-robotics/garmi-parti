"""
Demos running on the Panda system.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
from panda_py import controllers
from scipy.spatial import transform as tr

from .teleoperation import client, interface, server, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("panda")

Q_IDLE = utils.JointPositions(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
)
Q_TELEOP = utils.JointPositions(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
)


class CartesianLeader(interface.PandaInterface):
    """
    Teleoperation interface that uses a Panda robot as a haptic input
    device in Cartesian space.
    """

    def __init__(self, params: utils.TeleopParams) -> None:
        super().__init__(params)
        self.paused = False

    def _pre_teleop(self, container: utils.TeleopContainer) -> None:
        super()._pre_teleop(container)
        container.controller = controllers.AppliedForce(
            damping=container.params.damping, filter_coeff=container.params.filter_coeff
        )

    def get_command(self) -> bytes:
        if self.paused:
            return pickle.dumps(utils.Displacement())
        return pickle.dumps(utils.compute_displacement(self.panda))

    def set_command(self, command: bytes) -> None:
        wrench: utils.Wrench = pickle.loads(command)
        self._set_command(self.panda, wrench)

    def _set_command(
        self, container: utils.TeleopContainer, wrench: utils.Wrench
    ) -> None:
        self.fdir(container)
        force_d = -container.params.gain_force * (
            container.transform_inv.apply(wrench.force)
        )
        torque_d = -container.params.gain_torque * (
            container.transform_inv.apply(wrench.torque)
        )
        container.controller.set_control(np.r_[force_d, torque_d])

    def pause(self) -> None:
        self.paused = True

    def unpause(self) -> None:
        self.panda.reinitialize()
        self.paused = False


class JointLeader(interface.PandaInterface):
    """
    Teleoperation interface that uses a Panda robot as a haptic input
    device in joint space.
    """

    def __init__(self, params: utils.TeleopParams) -> None:
        super().__init__(params)

    def _pre_teleop(self, container: utils.TeleopContainer) -> None:
        super()._pre_teleop(container)
        container.controller = controllers.AppliedTorque(
            damping=container.params.damping, filter_coeff=container.params.filter_coeff
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(self, container: utils.TeleopContainer) -> utils.JointPositions:
        return utils.JointPositions(container.arm.get_state().q)

    def set_command(self, command: bytes) -> None:
        joint_torques: utils.JointTorques = pickle.loads(command)
        self._set_command(self.panda, joint_torques)

    def _set_command(
        self, container: utils.TeleopContainer, joint_torques: utils.JointTorques
    ) -> None:
        self.fdir(container)
        container.controller.set_control(
            -container.params.gain_joint_torque * np.array(joint_torques.torques)
        )

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()

    def get_sync_command(self) -> bytes:
        return self.get_command()


class CartesianFollower(interface.PandaInterface):
    """
    Teleoperation interface that controls a Panda robot in Cartesian space.
    """

    def _pre_teleop(self, container: utils.TeleopContainer) -> None:
        super()._pre_teleop(container)

        # Compliance parameters
        stiffness = np.zeros((6, 6))
        stiffness[:3, :3] = np.eye(3) * container.params.linear_stiffness
        stiffness[3:, 3:] = np.eye(3) * container.params.angular_stiffness
        container.controller = controllers.CartesianImpedance(
            impedance=stiffness,
            damping_ratio=container.params.damping_ratio,
            nullspace_stiffness=container.params.nullspace_stiffness,
            filter_coeff=container.params.filter_coeff,
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(self, container: utils.TeleopContainer) -> utils.Wrench:
        wrench = container.arm.get_state().O_F_ext_hat_K
        return utils.Wrench(
            force=container.transform_inv.apply(wrench[:3]),
            torque=container.transform_inv.apply(wrench[3:]),
        )

    def set_command(self, command: bytes) -> None:
        displacement: utils.Displacement = pickle.loads(command)
        self._set_command(displacement, self.panda)

    def _set_command(
        self, command: utils.Displacement, container: utils.TeleopContainer
    ) -> None:
        self.fdir(container)
        if container.params.lock_translation:
            command.linear = np.zeros(3)
        if container.params.lock_rotation:
            command.angular = tr.Rotation.identity()
        position_d = container.position_init + container.transform.apply(command.linear)
        orientation_d = (
            container.transform
            * command.angular
            * container.transform_inv
            * container.orientation_init
        )
        container.controller.set_control(position_d, orientation_d.as_quat())

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()


class JointFollower(interface.PandaInterface):
    """
    Teleoperation interface that controls a Panda robot in joint space.
    """

    def _pre_teleop(self, container: utils.TeleopContainer) -> None:
        super()._pre_teleop(container)

        container.controller = controllers.JointPosition(
            stiffness=container.params.stiffness,
            damping=container.params.damping,
            filter_coeff=container.params.filter_coeff,
        )

    def get_command(self) -> bytes:
        return pickle.dumps(self._get_command(self.panda))

    def _get_command(self, container: utils.TeleopContainer) -> utils.JointTorques:
        return utils.JointTorques(container.arm.get_state().tau_ext_hat_filtered)

    def set_command(self, command: bytes) -> None:
        joint_positions: utils.JointPositions = pickle.loads(command)
        self._set_command(joint_positions, self.panda)

    def _set_command(
        self, joint_positions: utils.JointPositions, container: utils.TeleopContainer
    ) -> None:
        self.fdir(container)
        container.controller.set_control(joint_positions.positions)

    def pause(self) -> None:
        self.panda.arm.stop_controller()

    def unpause(self) -> None:
        self.start_teleop()

    def set_sync_command(self, command: bytes) -> None:
        self.move_arm(pickle.loads(command))


def teleop_leader() -> None:
    """
    Panda teleoperation demo.
    This Panda system acts as a network client and teleoperation
    leader that connects to a teleoperation server (the follower).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=13701)
    parser.add_argument(
        "--mode",
        choices=["joint", "cartesian"],
        default="joint",
        help="Specify teleoperation mode",
    )
    args = parser.parse_args()

    robot_host = os.environ.get("PANDA")
    if robot_host is None:
        raise RuntimeError(
            "Please make sure the environment variable "
            + "PANDA is set to the respective robot hostname."
        )
    damping = np.zeros(7)
    leader: interface.PandaInterface
    if args.mode == "joint":
        leader = JointLeader(utils.TeleopParams(robot_host, damping=damping))
    elif args.mode == "cartesian":
        leader = CartesianLeader(utils.TeleopParams(robot_host, damping=damping))

    cli = client.Client(leader, args.host, args.port)
    client.user_interface(cli)
    cli.shutdown()


def teleop_follower() -> None:
    """
    Panda teleoperation demo.
    The Panda system acts as a network server and teleoperation
    follower that accepts connections from teleoperation clients.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=13701)
    parser.add_argument(
        "--mode",
        choices=["joint", "cartesian"],
        default="joint",
        help="Specify teleoperation mode",
    )
    args = parser.parse_args()

    robot_host = os.environ.get("PANDA")
    if robot_host is None:
        raise RuntimeError(
            "Please make sure the environment variable "
            + "PANDA is set to the respective robot hostname."
        )

    follower: interface.PandaInterface
    if args.mode == "joint":
        follower = JointFollower(utils.TeleopParams(robot_host))
    elif args.mode == "cartesian":
        follower = CartesianFollower(utils.TeleopParams(robot_host))
    srv = server.Server(follower, args.port)
    server.user_interface(srv)
    srv.shutdown()
