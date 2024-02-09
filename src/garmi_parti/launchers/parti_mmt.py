from __future__ import annotations

import argparse
import logging

from ..parti import mmt
from ..peripherals import gamepad
from ..teleoperation import client

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_mmt")


def main() -> None:
    """
    PARTI model-mediated teleoperation demo.
    The PARTI system acts as a network client and teleoperation
    leader that connects to a teleoperation server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=13701)
    parser.add_argument("-gp", "--gamepad-port", type=int, default=13702)
    args = parser.parse_args()

    leader = mmt.Leader()
    try:
        cli = client.Client(leader, args.host, args.port)
    except ConnectionRefusedError:
        leader.post_teleop()
        raise
    gamepad_handle = gamepad.GamepadHandle(cli, "localhost", args.gamepad_port)
    client.user_interface(cli)
    cli.shutdown()
    gamepad_handle.stop()
