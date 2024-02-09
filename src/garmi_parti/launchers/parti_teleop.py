from __future__ import annotations

import argparse
import logging

from .. import parti
from ..peripherals import gamepad
from ..teleoperation import client, interfaces, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti")


def main() -> None:
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
    leader: interfaces.TwoArmPandaInterface
    if args.mode == "joint":
        leader = parti.JointLeader(left, right)
    elif args.mode == "cartesian":
        leader = parti.CartesianLeader(left, right)

    cli = client.Client(leader, args.host, args.port)
    # gp_publisher = Publisher(args.gamepad_port, 30)
    gamepad_handle = gamepad.GamepadHandle(cli, "localhost", args.gamepad_port)
    logger = interfaces.TwoArmLogger(leader)
    client.user_interface(cli)
    cli.shutdown()
    gamepad_handle.stop()
    # gp_publisher.stop()
    logger.stop()
