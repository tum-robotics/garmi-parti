from __future__ import annotations

import pickle
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pytest
import zmq
from numpy import testing

from garmi_parti import garmi_mmt, parti_mmt
from garmi_parti.teleoperation import utils


class TestGarmi(unittest.TestCase):
    @mock.patch("builtins.input", return_value="")
    def test_garmi_mmt_entrypoints(self, *args):
        del args
        garmi_mmt.teleop()

    @mock.patch("builtins.input", return_value="")
    def test_parti_mmt_entrypoints(self, *args):
        del args
        self.start_pub()
        with pytest.raises(ConnectionRefusedError):  # noqa: PT012
            parti_mmt.teleop()
            time.sleep(5)
        self.stop_pub()

    def test_joint_leader(self):
        self.start_pub()
        leader = parti_mmt.Leader()
        leader.pre_teleop()
        leader.start_teleop()
        leader.get_sync_command()
        leader.set_command(b"")
        joint_positions: utils.TwoArmJointPositions = pickle.loads(leader.get_command())
        self.assert_joint_positions(joint_positions.left)
        self.assert_joint_positions(joint_positions.right)
        leader.pause()
        leader.unpause()
        leader.post_teleop()
        self.stop_pub()

    def assert_joint_positions(self, joint_positions: utils.JointPositions) -> None:
        testing.assert_allclose(joint_positions.positions, np.zeros(7))

    def start_pub(self):
        self.running = True
        self.thread = threading.Thread(target=self.pub)
        self.thread.start()

    def stop_pub(self):
        self.running = False
        self.thread.join()

    def pub(self):
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("ipc:///tmp/parti-haptic-sim")
        while self.running:
            socket.send(
                pickle.dumps(
                    utils.TwoArmJointPositions(
                        left=utils.JointPositions(np.zeros(7)),
                        right=utils.JointPositions(np.zeros(7)),
                    )
                )
            )
            time.sleep(1e-3)
        socket.close()
        context.term()
