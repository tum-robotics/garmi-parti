"""
Haptic interaction demo for the PARTI system.

This demo loads a simulation scene with HIL connection to the
PARTI system that allows the user to manipulate and haptically
interact with the virtual environment.
Alternatively, a teleoperation connection with a two-arm follower
can be established for a model-mediated teleoperation (MMT) scenario.
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
from dm_control import composer
from dm_robotics.moma.tasks import run_loop
from dm_robotics.panda import arm_constants, environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import utils as dmr_panda_utils

from ..parti import haptic_simulation as sim
from ..teleoperation import containers

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_haptic_sim")


XML_PATH = (
    pathlib.Path(pathlib.Path(__file__).parent.parent) / "assets" / "parti_mmt.xml"
)

Q_TELEOP_LEFT = containers.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = containers.JointPositions(
    [-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84]
)

SPEED_FACTOR = 0.2


def main() -> None:
    """
    Simulation for model-mediated teleoperation with the PARTI system.
    The sim connects to the robots to render haptical feedback.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    if args.sim_only:
        left_hostname = None
        right_hostname = None

    arena = composer.Arena(xml_path=XML_PATH)
    left_frame = arena.mjcf_model.find("site", "left")
    right_frame = arena.mjcf_model.find("site", "right")
    plane = arena.mjcf_model.find("body", "plane")
    obj = [arena.mjcf_model.find("joint", jn) for jn in ["x", "y", "theta"]]

    left = params.RobotParams(
        robot_ip=left_hostname,
        name="left",
        has_hand=False,
        joint_positions=Q_TELEOP_LEFT.positions,
        attach_site=left_frame,
        gripper=sim.make_gripper("right"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    right = params.RobotParams(
        robot_ip=right_hostname,
        name="right",
        has_hand=False,
        joint_positions=Q_TELEOP_RIGHT.positions,
        attach_site=right_frame,
        gripper=sim.make_gripper("left"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    robot_params = [left, right]

    env_builder = environment.PandaEnvironment(robot_params, arena, 0.016)
    env_builder.add_extra_effectors([sim.SceneEffector(plane, obj)])

    with env_builder.build_task_environment() as env:
        dmr_panda_utils.full_spec(env)
        agent = sim.TeleopAgent(
            env.task.arena,
            env.action_spec(),
            np.r_[Q_TELEOP_LEFT.positions, Q_TELEOP_RIGHT.positions],
        )
        if not args.testing:
            app = dmr_panda_utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], 100)

    agent.shutdown()
