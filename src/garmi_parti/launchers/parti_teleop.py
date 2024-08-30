from __future__ import annotations

import argparse
import logging
import time

from .. import parti
from ..peripherals import joystick
from ..teleoperation import client, interfaces, utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti")


class PartiJoystick(joystick.SerialJoysticks):
    """Interface class for the integrated joystick handles of the PARTI system."""

    def __init__(self) -> None:
        self._client: client.Client | None = None
        self.gripper = {"left": True, "right": True}
        self.stop = False
        super().__init__()

    def set_client(self, teleop_client: client.Client) -> None:
        """Sets the teleop network client used to send button commands."""
        self._client = teleop_client

    def handle_changes(
        self, changes: list[tuple[str, str] | tuple[str, str, int]]
    ) -> None:
        if self._client is None:
            return
        for change in changes:
            if len(change) == 2:
                key, device_id = change
                if key == "G":
                    self.gripper[device_id] = not self.gripper[device_id]
                    if self.gripper[device_id]:
                        self._client.open(device_id)
                    else:
                        self._client.close(device_id)
                elif key == "R":
                    self._client.shutdown()
                    self.stop = True
                _logger.info("Button %s pressed on device %s", key, device_id)
            elif len(change) == 3:
                key, device_id, state = change
                _logger.info(
                    "Analog trigger %s crossed threshold on device %s, new state: %s",
                    key,
                    device_id,
                    state,
                )
                if state >= 560:
                    self._client.unpause(device_id)
                else:
                    self._client.pause(device_id)


def main() -> None:
    """
    PARTI teleoperation demo.
    The PARTI system acts as a network client and teleoperation
    leader that connects to a teleoperation server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=13701)
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

    joysticks = PartiJoystick()
    joysticks.start()

    cli = client.Client(leader, args.host, args.port)
    if args.mode == "joint":
        leader.left.arm.stop_controller()
        leader.right.arm.stop_controller()
    cli.pause()
    joysticks.set_client(cli)
    logger = interfaces.TwoArmLogger(leader)

    while not joysticks.stop:
        time.sleep(1)

    joysticks.close()
    cli.shutdown()
    logger.stop()
