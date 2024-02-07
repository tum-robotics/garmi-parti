from __future__ import annotations

import pickle
import threading
import time
import unittest
from unittest import mock

import numpy as np
import zmq

from garmi_parti.peripherals import gamepad
from garmi_parti.teleoperation import client, server, utils

SYNC_CMD = pickle.dumps(utils.JointPositions(np.zeros(7)))


class TestGamepad(unittest.TestCase):
    def test_gamepad(self) -> None:
        leader = mock.Mock()
        leader.get_command.return_value = b"TEST"
        leader.get_sync_command.return_value = SYNC_CMD
        follower = mock.Mock()
        follower.get_command.return_value = b"TEST"
        serv = server.Server(follower, 13701)
        cli = client.Client(leader, "localhost", 13701)
        time.sleep(2)

        self.start_pub()
        gp = gamepad.GamepadHandle(cli, "localhost", 13702)
        time.sleep(1)
        self.stop_pub()

        follower.open.assert_called_with("left")
        follower.close.assert_called_with("left")

        gp.stop()
        cli.shutdown()
        serv.shutdown()

    def start_pub(self):
        self.running = True
        self.thread = threading.Thread(target=self.pub)
        self.thread.start()

    def stop_pub(self):
        self.running = False
        self.thread.join()

    def pub(self):
        context = zmq.Context()
        sock = context.socket(zmq.PUB)
        sock.bind("tcp://0.0.0.0:13702")
        while self.running:
            sock.send(pickle.dumps({"event": "down", "side": "left", "button": 1}))
            sock.send(pickle.dumps({"event": "up", "side": "left", "button": 1}))
        sock.close()
        context.term()
