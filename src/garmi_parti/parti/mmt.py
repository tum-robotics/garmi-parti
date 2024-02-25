"""
Model-mediated teleoperation interfaces for the Parti robot.
"""

from __future__ import annotations

import pickle
import threading

import zmq

from ..teleoperation import containers, interfaces


class Leader(interfaces.Interface):
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
        self.joint_states: containers.TwoArmJointStates = pickle.loads(
            self.socket.recv()
        )

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
        return pickle.dumps(self.joint_states)

    def set_command(self, command: bytes) -> None:
        pass

    def open(self, end_effector: str = "") -> None:
        pass

    def close(self, end_effector: str = "") -> None:
        pass

    def get_sync_command(self) -> bytes:
        return pickle.dumps(
            containers.TwoArmJointPositions(
                left=self.joint_states.left.q, right=self.joint_states.right.q
            )
        )

    def set_sync_command(self, command: bytes) -> None:
        pass
