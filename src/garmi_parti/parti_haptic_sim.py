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
import pickle
import threading

import dm_env
import numpy as np
import panda_py
import zmq
from dm_control import composer, mjcf
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.moma.tasks import run_loop
from dm_robotics.panda import arm_constants, environment, gripper
from dm_robotics.panda import parameters as params
from dm_robotics.panda import utils as dmr_panda_utils
from dm_robotics.transformations import transformations as tr

from .teleoperation import containers

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_haptic_sim")

XML_PATH = pathlib.Path(pathlib.Path(__file__).parent) / "assets" / "parti_mmt.xml"

Q_TELEOP_LEFT = containers.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = containers.JointPositions(
    [-0.04, -1.16, 0.08, -1.57, 0.00, 2.05, 0.84]
)

SPEED_FACTOR = 0.2


class TeleopAgent:
    """Teleoperation agent for MMT.

    This agent uses interprocess communication (IPC) to communicate
    with a local teleoperation node. Run `parti-mmt` on the PARTI
    system and `garmi-mmt` on the GARMI system to run a full MMT scenario.
    """

    def __init__(
        self, arena: composer.Arena, spec: specs.BoundedArray, action: np.ndarray
    ) -> None:
        self._spec = spec
        self._arena: composer.Arena = arena
        self._action = action
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/parti-haptic-sim")

    def shutdown(self) -> None:
        """
        Shutdown, closing any open connections.
        """
        self.socket.close()
        self.context.term()

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Steps the agent.
        Receives a percept from the environment and returns an action.
        """
        joint_positions = containers.TwoArmJointPositions(
            left=containers.JointPositions(timestep.observation["left_joint_pos"]),
            right=containers.JointPositions(timestep.observation["right_joint_pos"]),
        )
        self.socket.send(pickle.dumps(joint_positions))
        return np.zeros(23)


class SceneEffector(effector.Effector):
    """
    Effector used to update the state of tracked objects.
    """

    def __init__(self, plane: mjcf.Element, obj: list[mjcf.Element]) -> None:
        self._plane = plane
        self._object = obj
        self._spec = None

    def close(self) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        del physics
        if self._spec is None:
            self._spec = specs.BoundedArray(
                (7,),
                np.float32,
                np.full((7,), -10, dtype=np.float32),
                np.full((7,), 10, dtype=np.float32),
                "\t".join([f"{self.prefix}_{c}" for c in range(7)]),
            )
        return self._spec

    @property
    def prefix(self) -> str:
        return "scene"

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        del command
        update = True
        for contact in physics.data.contact:
            geom1_name = physics.model.id2name(contact.geom1, "geom")
            geom2_name = physics.model.id2name(contact.geom2, "geom")
            if (
                (geom1_name == "object" or geom2_name == "object")
                and geom1_name != "plane"
                and geom2_name != "plane"
            ):
                update = False
                break
        # only update if object is not in contact with element other than plane
        if update:
            physics.bind(self._object).qpos[:] = [0, 0, 0]
        # update plane as you see fit
        physics.bind(self._plane).mocap_quat[:] = tr.euler_to_quat(
            [np.sin(physics.time()) * 0.1, 0, 0]
        )


def make_gripper(name: str) -> params.GripperParams:
    """Creates a Panda gripper.

    We create the Panda gripper manually here because the connected
    hardware doesn't actually have grippers. If we were to use the
    grippers configured by dm-robotics-panda, the HIL mechanism would
    attempt to connect to nonexistent grippers.
    """
    name = f"{name}_gripper"
    gripper_model = gripper.PandaHand()
    gripper_sensor = gripper.PandaHandSensor(gripper_model, name)
    gripper_effector = gripper.PandaHandEffector(
        params.RobotParams(name=name), gripper_model, gripper_sensor
    )
    return params.GripperParams(gripper_model, gripper_effector)


def move_arms(
    left_hostname: str,
    right_hostname: str,
    q_left: np.ndarray,
    q_right: np.ndarray,
    speed_factor: float,
) -> None:
    """Move robot arms.

    Utility function that moves two robot arms simultaneously
    to the given joint positions.
    """
    panda_left = panda_py.Panda(left_hostname)
    panda_right = panda_py.Panda(right_hostname)
    t_left = threading.Thread(
        target=panda_left.move_to_joint_position, args=(q_left.positions, speed_factor)
    )
    t_right = threading.Thread(
        target=panda_right.move_to_joint_position,
        args=(q_right.positions, speed_factor),
    )
    t_left.start()
    t_right.start()
    t_left.join()
    t_right.join()
    del panda_left, panda_right


def simulate() -> None:
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
        gripper=make_gripper("right"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    right = params.RobotParams(
        robot_ip=right_hostname,
        name="right",
        has_hand=False,
        joint_positions=Q_TELEOP_RIGHT.positions,
        attach_site=right_frame,
        gripper=make_gripper("left"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    robot_params = [left, right]

    env_builder = environment.PandaEnvironment(robot_params, arena, 0.016)
    env_builder.add_extra_effectors([SceneEffector(plane, obj)])

    with env_builder.build_task_environment() as env:
        dmr_panda_utils.full_spec(env)
        agent = TeleopAgent(
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
