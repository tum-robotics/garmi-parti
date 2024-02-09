from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from .. import panda
from ..teleoperation import client, containers, interfaces, server

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("panda")


Q_IDLE = containers.JointPositions(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
)
Q_TELEOP = containers.JointPositions(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
)


def leader() -> None:
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
    _leader: interfaces.PandaInterface
    if args.mode == "joint":
        _leader = panda.JointLeader(
            containers.TeleopParams(robot_host, damping=damping)
        )
    elif args.mode == "cartesian":
        _leader = panda.CartesianLeader(
            containers.TeleopParams(robot_host, damping=damping)
        )

    cli = client.Client(_leader, args.host, args.port)
    client.user_interface(cli)
    cli.shutdown()


def follower() -> None:
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

    _follower: interfaces.PandaInterface
    if args.mode == "joint":
        _follower = panda.JointFollower(containers.TeleopParams(robot_host))
    elif args.mode == "cartesian":
        _follower = panda.CartesianFollower(containers.TeleopParams(robot_host))
    srv = server.Server(_follower, args.port)
    server.user_interface(srv)
    srv.shutdown()
