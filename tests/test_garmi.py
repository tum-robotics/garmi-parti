from __future__ import annotations

import pickle
import unittest
from unittest import mock

import numpy as np
import pytest
from numpy import testing

from garmi_parti import garmi
from garmi_parti.teleoperation import utils

SYNC_CMD = pickle.dumps(
    utils.TwoArmJointPositions(
        left=utils.JointPositions(np.zeros(7)), right=utils.JointPositions(np.zeros(7))
    )
)
ONE_ARM_SYNC_CMD = pickle.dumps(utils.JointPositions(np.zeros(7)))


class TestGarmi(unittest.TestCase):
    def test_cartesian_follower(self):
        left = utils.TeleopParams("left-host")
        right = utils.TeleopParams("right-host")
        follower = garmi.CartesianFollower(left, right, True, True)
        follower.pre_teleop()
        follower.start_teleop()
        cmd = utils.TwoArmDisplacement(utils.Displacement(), utils.Displacement())
        follower.set_command(pickle.dumps(cmd))
        wrench: utils.TwoArmWrench = pickle.loads(follower.get_command())
        self.assert_wrench(wrench.left)
        self.assert_wrench(wrench.right)
        follower.open("left")
        follower.close("left")
        self.assert_gripper(follower.left.gripper, 2)
        follower.open("right")
        follower.close("right")
        self.assert_gripper(follower.right.gripper, 4)
        follower.pause()
        follower.unpause()
        follower.post_teleop()

    def test_joint_follower(self):
        left = utils.TeleopParams("left-host")
        right = utils.TeleopParams("right-host")
        follower = garmi.JointFollower(left, right, True, True)
        follower.pre_teleop()
        follower.start_teleop()

        joint_positions = utils.TwoArmJointPositions(
            left=utils.JointPositions(np.zeros(7)),
            right=utils.JointPositions(np.zeros(7)),
        )
        follower.set_sync_command(SYNC_CMD)
        follower.set_command(pickle.dumps(joint_positions))
        joint_torques: utils.TwoArmJointTorques = pickle.loads(follower.get_command())
        self.assert_torques(joint_torques.left)
        self.assert_torques(joint_torques.right)
        follower.open("left")
        follower.close("left")
        self.assert_gripper(follower.left.gripper, 2)
        follower.open("right")
        follower.close("right")
        self.assert_gripper(follower.right.gripper, 4)
        follower.pause()
        follower.unpause()
        follower.post_teleop()

    def test_one_arm_joint_follower(self):
        left = utils.TeleopParams("left-host")
        right = utils.TeleopParams("right-host")
        follower = garmi.OneArmJointFollower("left", left, right, True, True)
        follower.pre_teleop()
        follower.start_teleop()
        joint_positions = utils.JointPositions(np.zeros(7))
        follower.set_sync_command(ONE_ARM_SYNC_CMD)
        follower.set_command(pickle.dumps(joint_positions))
        joint_torques: utils.JointTorques = pickle.loads(follower.get_command())
        self.assert_torques(joint_torques)
        follower.open("left")
        follower.close("left")
        self.assert_gripper(follower.left.gripper, 2)
        follower.open("right")
        follower.close("right")
        self.assert_gripper(follower.right.gripper, 4)
        follower.pause()
        follower.unpause()
        follower.post_teleop()

    def assert_gripper(self, gripper, count: int) -> None:
        assert gripper.stop.call_count == count
        assert gripper.grasp.call_count == count

    def assert_wrench(self, wrench: utils.Wrench) -> None:
        testing.assert_allclose(wrench.force, np.zeros(3))
        testing.assert_allclose(wrench.torque, np.zeros(3))

    def assert_torques(self, joint_torques: utils.JointTorques) -> None:
        testing.assert_allclose(joint_torques.torques, np.zeros(7))

    @mock.patch("builtins.input", return_value="")
    def test_entrypoints(self, *args):
        del args
        garmi.teleop()

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_entrypoints_raises(self):
        with pytest.raises(RuntimeError) as ctx:
            garmi.teleop()
        assert "PANDA_LEFT and PANDA_RIGHT" in str(ctx.value.args[0])
