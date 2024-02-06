"""
This module integrates gamepads and similar peripheral devices.
"""
from __future__ import annotations

import logging
import pickle
import threading

import zmq

from garmi_parti.teleoperation import client

_logger = logging.getLogger("gamepad")


class GamepadHandle:
    """
    Listens to gamepad events using zmq to trigger actions on
    the provided teleoperation client.
    """

    def __init__(self, teleop_client: client.Client, hostname: str, port: int) -> None:
        self.gripper = {"left": True, "right": True}
        self.client = teleop_client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{hostname}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.running = True
        threading.Thread(target=self._listen).start()

    def _listen(self) -> None:
        while self.running:
            try:
                event = pickle.loads(self.socket.recv())
                button = event["button"]
                side = event["side"]
                event_type = event["event"]
                _logger.info("Received %s button %s event %s", side, button, event_type)

                if event_type == "up" and button == 1:
                    self.gripper[side] = not self.gripper[side]
                    if self.gripper[side]:
                        self.client.open(side)
                    else:
                        self.client.close(side)
                elif event_type == "down" and button == 0:
                    self.client.pause()
                elif event_type == "up" and button == 0:
                    self.client.unpause()
            except zmq.error.ContextTerminated:
                break
        self.socket.close()

    def stop(self) -> None:
        """
        Stop listening and close connection.
        """
        self.context.term()
