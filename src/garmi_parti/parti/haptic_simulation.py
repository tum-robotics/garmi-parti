"""
Haptic simulation module for the Parti robot.
"""

from __future__ import annotations

import pickle
import threading

import dm_env
import numpy as np
import panda_py
import zmq
from dm_control import composer, mjcf
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.panda import gripper
from dm_robotics.panda import parameters as params
from dm_robotics.transformations import transformations as tr

from ..teleoperation import containers


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
        self._comm(timestep)
        action = np.zeros(23)

        # This controls the position of the object (x, y, theta)
        action[-3:] = [0, 0, 0]
        # This controls the orientation of the plane (w, x, y, z)
        action[-7:-3] = tr.euler_to_quat(
            [np.sin(timestep.observation["time"][0]) * 0.1, 0, 0]
        )
        return action

    def _comm(self, timestep: dm_env.TimeStep) -> None:
        joint_positions = containers.TwoArmJointPositions(
            left=containers.JointPositions(timestep.observation["left_joint_pos"]),
            right=containers.JointPositions(timestep.observation["right_joint_pos"]),
        )
        joint_velocities = containers.TwoArmJointVelocities(
            left=containers.JointVelocities(timestep.observation["left_joint_vel"]),
            right=containers.JointVelocities(timestep.observation["right_joint_vel"]),
        )
        self.socket.send(pickle.dumps((joint_positions, joint_velocities)))


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
                "\t".join(
                    [
                        f"{self.prefix}_{n}"
                        for n in [
                            "plane_w",
                            "plane_x",
                            "plane_y",
                            "plane_z",
                            "object_x",
                            "object_y",
                            "object_theta",
                        ]
                    ]
                ),
            )
        return self._spec

    @property
    def prefix(self) -> str:
        return "scene"

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
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
            physics.bind(self._object).qpos[:] = command[4:]
        # always update plane
        physics.bind(self._plane).mocap_quat[:] = command[:4]


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
