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
from dm_robotics.transformations import transformations as moma_tr

from .teleoperation import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_haptic_sim")

XML_PATH = pathlib.Path(pathlib.Path(__file__).parent) / "assets" / "parti_mmt.xml"

Q_IDLE_LEFT = utils.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_IDLE_RIGHT = utils.JointPositions(
    [0, -np.pi / 2, 0, -2 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
)
Q_TELEOP_LEFT = utils.JointPositions([0.02, -1.18, -0.06, -1.47, 0.04, 1.92, 0.75])
Q_TELEOP_RIGHT = utils.JointPositions(
    [
        0.1261991807682472,
        -1.3334758399021756,
        -0.12895294076006777,
        -2.467273292813602,
        -0.26848237937026553,
        2.6813366335795985,
        0.8649797521854519,
    ]
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
        # self.ros = roslibpy.Ros('10.157.174.246', 9090)
        # self.ros.run()
        # roslibpy.Topic(self.ros, '/icg_pose_topic',
        #                'geometry_msgs/PoseStamped').subscribe(self.listen)
        # roslibpy.Topic(self.ros, '/icg_tracker_2/pose',
        #                'geometry_msgs/PoseStamped').subscribe(self.listen)
        self.pos = [0, 0, 0]
        self.quat = [1, 0, 0, 0]

    # def listen(self, msg):
    #   self.pos = [
    #       msg['pose']['position']['x'], msg['pose']['position']['y'],
    #       msg['pose']['position']['z']
    #   ]
    #   self.quat = [
    #       msg['pose']['orientation']['w'], msg['pose']['orientation']['x'],
    #       msg['pose']['orientation']['y'], msg['pose']['orientation']['z']
    #   ]

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
        joint_positions = utils.TwoArmJointPositions(
            left=utils.JointPositions(timestep.observation["left_joint_pos"]),
            right=utils.JointPositions(timestep.observation["right_joint_pos"]),
        )
        self.socket.send(pickle.dumps(joint_positions))
        return np.zeros(16)
        # action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # action[:14] = self._action
        # pos = [0, 0, 0]
        # quat = [1, 0, 0, 0]
        # action[-7:] = np.r_[self.pos, self.quat]
        # # print(self._arena.mjcf_model.find('body', 'mmt'))
        # return action


class MocapEffector(effector.Effector):
    """
    Effector used to update the state of the tracked object.
    """

    def __init__(self, body: mjcf.Element) -> None:
        self._spec = None
        self._body = body
        self._pos = [0, -0.205, 0.24]
        self._quat = moma_tr.euler_to_quat([0, np.pi / 2, np.pi / 2], "XYZ")

    def close(self) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        del physics
        if self._spec is None:
            # self._spec = specs.DiscreteArray(2, name=f'{self.prefix}_grasp')
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
        return "mocap"

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        physics_body = physics.bind(self._body)
        physics_body.mocap_pos[:] = self._pos + moma_tr.quat_rotate(
            self._quat, command[:3]
        )
        physics_body.mocap_quat[:] = moma_tr.quat_mul(self._quat, command[3:])


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

    if not args.sim_only:
        left_hostname, right_hostname = utils.get_robot_hostnames()
        move_arms(
            left_hostname, right_hostname, Q_TELEOP_LEFT, Q_TELEOP_RIGHT, SPEED_FACTOR
        )
    else:
        left_hostname = None
        right_hostname = None

    left_pos = [0, 0.205, 0.24]
    left_eul = moma_tr.quat_to_euler(
        moma_tr.euler_to_quat([-np.pi / 2, np.pi / 2, 0], "XYZ"), "XYZ"
    )
    right_pos = [0, -0.205, 0.24]
    right_eul = moma_tr.quat_to_euler(
        moma_tr.euler_to_quat([np.pi / 2, np.pi / 2, 0], "XYZ"), "XYZ"
    )

    arena = composer.Arena(xml_path=XML_PATH)

    left = params.RobotParams(
        robot_ip=left_hostname,
        name="left",
        has_hand=False,
        joint_positions=Q_IDLE_LEFT.positions,
        pose=np.concatenate([left_pos, left_eul]),
        gripper=make_gripper("right"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    right = params.RobotParams(
        robot_ip=right_hostname,
        name="right",
        has_hand=False,
        joint_positions=Q_IDLE_RIGHT.positions,
        pose=np.concatenate([right_pos, right_eul]),
        gripper=make_gripper("left"),
        actuation=arm_constants.Actuation.HAPTIC,
    )
    robot_params = [left, right]

    # def add_hole(arena: composer.Arena) -> None:
    #   hole = mjcf.from_path(HOLE_XML_PATH)
    #   hole.find('body', 'hole').pos = [.9, .2, .2]
    #   hole.find('body', 'hole').euler = [0, 0, 0]
    #   arena.mjcf_model.attach(hole)

    # def add_peg(robots: typing.Sequence[robot.Robot]) -> None:

    #   def hand():
    #     elem = mjcf.from_path(gripper_constants.XML_PATH)
    #     return elem

    #   robots[0].gripper.tool_center_point.attach(hand())
    #   robots[1].gripper.tool_center_point.attach(hand())
    #   peg = mjcf.from_path(PEG_XML_PATH)
    #   peg.find('body', 'peg').pos = [0, 0, 0.165]
    #   peg.find('body', 'peg').euler = [180, 0, 0]
    #   robots[0].gripper.tool_center_point.attach(peg)

    # def adjust_gripper(
    #     robots: typing.Sequence[robot.Robot],
    #     arena: composer.Arena) -> entity_initializer.base_initializer.Initializer:
    #   del robots

    #   class AdjustGripper(entity_initializer.base_initializer.Initializer):

    #     def __call__(self, physics: mjcf.Physics,
    #                  random_state: np.random.RandomState) -> bool:
    #       physics.bind(
    #           arena.mjcf_model.find(
    #               'actuator',
    #               'left/left_hand/panda_hand/panda_hand_actuator')).ctrl = .55
    #       return True

    #   return AdjustGripper()

    # def add_extra_effectors(robots, arena: composer.Arena):
    #   return [MocapEffector(arena.mjcf_model.find('body', 'mmt'))]

    # build_params = params.BuilderExtensions(
    #     build_robots=add_peg,
    #     build_arena=add_hole,
    #     build_entity_initializer=adjust_gripper,
    #     build_extra_effectors=add_extra_effectors)
    # panda_env_builder = env_builder.PandaEnvironmentBuilder(
    #     robot_params, env_params, build_params)

    env_builder = environment.PandaEnvironment(robot_params, arena, 0.016)

    with env_builder.build_task_environment() as env:
        dmr_panda_utils.full_spec(env)
        agent = TeleopAgent(
            env.task.arena,
            env.action_spec(),
            np.r_[Q_IDLE_LEFT.positions, Q_IDLE_RIGHT.positions],
        )
        if not args.testing:
            app = dmr_panda_utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], 100)

    agent.shutdown()

    # if not args.sim_only:
    #     move_arms(
    #         left_hostname, right_hostname, Q_IDLE_LEFT, Q_IDLE_RIGHT, SPEED_FACTOR
    #     )
