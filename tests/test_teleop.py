from __future__ import annotations

import pickle
import time
import unittest
from unittest import mock

import numpy as np
import pytest

from garmi_parti.teleoperation import client, server, utils

# Set the seed for reproducibility (optional)
rng = np.random.default_rng(42)

SYNC_CMD = pickle.dumps(utils.JointPositions(np.zeros(7)))


class TestNetworking(unittest.TestCase):
    def test_teleop(self):
        port = rng.integers(low=13701, high=16701)
        leader, follower = self.create_mocks()
        serv = server.Server(follower, port)
        cli = client.Client(leader, "localhost", port)
        time.sleep(2)
        cli.open("left")
        cli.open("right")
        cli.close("left")
        cli.close("right")
        cli.pause()
        cli.unpause()
        cli.shutdown()
        serv.shutdown()
        self.assert_teleop(leader, True)
        self.assert_teleop(follower, False)

    @mock.patch(
        "garmi_parti.teleoperation.server.Server._synchronize", return_value=False
    )
    def test_sync_failed_teleop(self, *args):
        del args
        port = rng.integers(low=13701, high=16701)
        leader, follower = self.create_mocks()
        serv = server.Server(follower, port)
        with pytest.raises(RuntimeError):  # noqa: PT012
            client.Client(leader, "localhost", port)
            time.sleep(2)
        serv.shutdown()

    @mock.patch("garmi_parti.teleoperation.server.Server._connect", return_value=False)
    def test_connect_failed_teleop(self, *args):
        del args
        port = rng.integers(low=13701, high=16701)
        leader, follower = self.create_mocks()
        serv = server.Server(follower, port)
        with pytest.raises(RuntimeError):  # noqa: PT012
            client.Client(leader, "localhost", port)
            time.sleep(2)
        serv.shutdown()

    @mock.patch("builtins.input", return_value="q")
    def test_user_interface(self, *args):
        del args
        cli = mock.Mock()
        client.user_interface(cli)

    def assert_teleop(self, teleop, leader=True):
        teleop.pre_teleop.assert_called_once()
        teleop.start_teleop.assert_called_once()
        teleop.post_teleop.assert_called_once()
        teleop.get_command.assert_called()
        teleop.set_command.assert_called_with(b"TEST")
        teleop.pause.assert_called_once()
        teleop.unpause.assert_called_once()

        if leader:
            assert teleop.get_sync_command.call_count == 2
        else:
            teleop.set_sync_command.assert_called_with(SYNC_CMD)
            assert teleop.set_sync_command.call_count == 2

            teleop.open.assert_any_call("left")
            teleop.open.assert_any_call("right")
            teleop.close.assert_any_call("left")
            teleop.close.assert_any_call("right")

    def test_server_timeout(self):
        port = rng.integers(low=13701, high=16701)
        with self.assertLogs("teleoperation.server", level="ERROR") as cm:
            leader, follower = self.create_mocks()
            serv = server.Server(follower, port)
            cli = client.Client(leader, "localhost", port)
            time.sleep(2)
            cli.running_udp = False
            time.sleep(10)
            cli.shutdown()
            serv.shutdown()
            assert "Client timed out." in cm.output[0]

    def test_client_timeout(self):
        port = rng.integers(low=13701, high=16701)
        with self.assertLogs("teleoperation.client", level="ERROR") as cm:
            leader, follower = self.create_mocks()
            serv = server.Server(follower, port)
            cli = client.Client(leader, "localhost", port)
            time.sleep(2)
            serv.shutdown()
            time.sleep(10)
            cli.shutdown()
            assert "Server timed out." in cm.output[0]

    def create_mocks(self):
        leader = mock.Mock()
        leader.get_command.return_value = b"TEST"
        leader.get_sync_command.return_value = SYNC_CMD
        follower = mock.Mock()
        follower.get_command.return_value = b"TEST"
        return leader, follower
