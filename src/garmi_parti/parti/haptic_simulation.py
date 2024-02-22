"""
Haptic simulation module for the Parti robot.
"""

from __future__ import annotations

import pickle
import threading

import dm_env
import numpy as np
import panda_py

# from geometry_msgs.msg import Pose, PoseStamped
# from pyquaternion import Quaternion
import roslibpy

# import rospy
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
        self,
        arena: composer.Arena,
        spec: specs.BoundedArray,
        action: np.ndarray,
        use_ros: bool,
        ros_hostname: str = "localhost",
        rosbridge_port: int = 9090,
    ) -> None:
        self._spec = spec
        self._arena: composer.Arena = arena
        self._action = action
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("ipc:///tmp/parti-haptic-sim")

        self._camera_pos = np.array([0.21323, -0.363705, 0.215217])
        self._camera_quat = np.array([0.507675, -0.86154, 0.000303674, -0.00397619])

        self._object_qpos = np.array([-0.06, 0, 0])
        self._plane_declination = 0

        if use_ros:
            client = roslibpy.Ros(host=ros_hostname, port=rosbridge_port)
            client.run()
            listener = roslibpy.Topic(
                client, "/icg_tracker_1/pose", "geometry_msgs/PoseStamped"
            )
            listener.subscribe(self._object_callback)
            listener_2 = roslibpy.Topic(
                client, "/detect_plane/pose", "geometry_msgs/Pose"
            )
            listener_2.subscribe(self._plane_callback)

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
        action = np.zeros(20)
        action[7] = 1
        action[15] = 1

        # # This controls the position of the object (x, y, theta)
        action[-3:] = self._object_qpos
        # This controls the declination of the plane
        action[-4] = self._plane_declination
        return action

    def _plane_callback(self, message: dict) -> None:
        local_normal = [
            message["position"]["x"],
            message["position"]["y"],
            message["position"]["z"],
        ]
        global_normal = tr.quat_rotate(self._camera_quat, local_normal)
        # TODO this is probably in right arm base frame for two-arm system
        # this assumes camera frame to be in world frame
        local_normal = tr.quat_between_vectors([0, 0, 1], global_normal)
        self._plane_declination = tr.quat_to_euler(local_normal)[0]

    def _object_callback(self, message):
        # right arm base frame in world frame
        T_0_right0 = tr.pos_quat_to_hmat([0, -0.24, 0.5], [0.5, 0.5, 0.5, 0.5])
        # object in right arm base frame
        T_right0_object = tr.pos_quat_to_hmat(
            [
                message["pose"]["position"]["x"],
                message["pose"]["position"]["y"],
                message["pose"]["position"]["z"],
            ],
            [
                message["pose"]["orientation"]["w"],
                message["pose"]["orientation"]["x"],
                message["pose"]["orientation"]["y"],
                message["pose"]["orientation"]["z"],
            ],
        )
        # plane frame in world frame
        T_0_plane = tr.pos_quat_to_hmat([1, 0, 0.3], [1, 0, 0, 0])
        # world frame in plane frame
        T_plane_0 = tr.hmat_inv(T_0_plane)

        # object in plane frame
        poseuler = tr.hmat_to_poseuler(T_plane_0 @ T_0_right0 @ T_right0_object, "XYZ")
        self._object_qpos = np.array([poseuler[0], poseuler[1], poseuler[5]])

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
                (4,),
                np.float32,
                np.full((4,), -10, dtype=np.float32),
                np.full((4,), 10, dtype=np.float32),
                "\t".join(
                    [
                        f"{self.prefix}_{n}"
                        for n in [
                            "plane",
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
        allowed_collision = ["plane", "side_a", "side_b", "side_c", "side_d"]
        for contact in physics.data.contact:
            geom1_name = physics.model.id2name(contact.geom1, "geom")
            geom2_name = physics.model.id2name(contact.geom2, "geom")
            if (
                (
                    geom1_name in ["object_1", "object_2"]
                    or geom2_name in ["object_1", "object_2"]
                )
                and geom1_name not in allowed_collision
                and geom2_name not in allowed_collision
            ):
                update = False
                break
        # only update scene if object is not in contact with element other than plane
        if update:
            # update object
            physics.bind(self._object).qpos[:] = command[1:]
            physics.bind(self._object).qvel[:] = np.zeros(3)
            # update plane
            physics.bind(self._plane).qpos[:] = command[0]
            physics.bind(self._plane).qvel[:] = 0


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
