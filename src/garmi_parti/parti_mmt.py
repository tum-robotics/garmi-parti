"""
Demos running on the PARTI system.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import threading

import zmq

from .peripherals import gamepad
from .teleoperation import client, interface

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_mmt")


class Leader(interface.Interface):
    """
    Use PARTI or a similar system as a model-mediated teleoperation leader device.
    This module requires the `parti-haptic-sim` module to be running.
    """

    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("ipc:///tmp/parti-haptic-sim")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._receive()
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                self._receive()
            except zmq.error.ContextTerminated:
                break
        self.socket.close()

    def _receive(self) -> None:
        self.joint_positions = pickle.loads(self.socket.recv())

    def pre_teleop(self) -> bool:
        return True

    def start_teleop(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def unpause(self) -> None:
        pass

    def post_teleop(self) -> bool:
        self.context.term()
        return True

    def get_command(self) -> bytes:
        return pickle.dumps(self.joint_positions)

    def set_command(self, command: bytes) -> None:
        pass

    def open(self, end_effector: str = "") -> None:
        pass

    def close(self, end_effector: str = "") -> None:
        pass

    def get_sync_command(self) -> bytes:
        return self.get_command()

    def set_sync_command(self, command: bytes) -> None:
        pass


def teleop() -> None:
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

    leader = Leader()
    try:
        cli = client.Client(leader, args.host, args.port)
    except ConnectionRefusedError:
        leader.post_teleop()
        raise
    gamepad_handle = gamepad.GamepadHandle(cli, "localhost", args.gamepad_port)
    client.user_interface(cli)
    cli.shutdown()
    gamepad_handle.stop()
