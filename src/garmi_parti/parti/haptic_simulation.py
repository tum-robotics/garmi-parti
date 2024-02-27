"""
Haptic simulation module for the Parti robot.
"""

from __future__ import annotations

import pathlib
import pickle
import threading

import dm_env
import numpy as np
import panda_py
import roslibpy
import spatialmath
import zmq
from dm_control import composer, mjcf
from dm_env import specs
from dm_robotics.moma import effector
from dm_robotics.panda import gripper
from dm_robotics.panda import parameters as params
from dm_robotics.transformations import transformations as tr

from ..teleoperation import containers

# right arm base frame in world frame
T_0_right0 = tr.pos_quat_to_hmat([0, -0.24, 0.5], [0.5, 0.5, 0.5, 0.5])

# plane frame in world frame
T_0_plane = tr.pos_quat_to_hmat([0.845, 0.012, 0.34], [1, 0, 0, 0])

# world frame in plane frame
T_plane_0 = tr.hmat_inv(T_0_plane)

DEADTIME = 1.0


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

        self._object_qpos = np.array([-0.06, 0, 0])
        self._object_qpos_offset = np.array([0.028, 0, 0])
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
        action = np.zeros(19)

        # # This controls the position of the object (x, y, theta)
        action[-3:] = self._object_qpos + self._object_qpos_offset
        # This controls the declination of the plane
        action[-4] = self._plane_declination
        return action

    def _plane_callback(self, message: dict) -> None:
        normal = [
            message["position"]["x"],
            message["position"]["y"],
            message["position"]["z"],
        ]
        global_normal = T_0_right0[:3, :3] @ normal
        orientation = tr.quat_between_vectors([0, 0, 1], global_normal[:3])

        self._plane_declination = tr.quat_to_euler(orientation)[0]

    def _object_callback(self, message: dict) -> None:
        # object in right arm base frame
        T_right0_object = tr.pos_quat_to_hmat(  # pylint: disable=invalid-name
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
        # we move the COM
        COM = tr.hmat_inv(
            tr.pos_quat_to_hmat([2 * 0.075, 2 * -0.04, 2 * 0.015], [1, 0, 0, 0])
        )
        # object in plane frame
        T_plane_object = COM @ T_plane_0 @ T_0_right0 @ T_right0_object  # pylint: disable=invalid-name

        theta = spatialmath.SO3(T_plane_object[:3, :3]).rpy()
        theta = theta[2]
        self._object_qpos = np.array(
            [T_plane_object[0, 3], T_plane_object[1, 3], theta]
        )

    def _comm(self, timestep: dm_env.TimeStep) -> None:
        joint_states = containers.TwoArmJointStates(
            left=containers.JointStates(
                q=containers.JointPositions(timestep.observation["left_joint_pos"]),
                dq=containers.JointVelocities(timestep.observation["left_joint_vel"]),
            ),
            right=containers.JointStates(
                q=containers.JointPositions(timestep.observation["right_joint_pos"]),
                dq=containers.JointVelocities(timestep.observation["right_joint_vel"]),
            ),
        )
        self.socket.send(pickle.dumps(joint_states))


class SceneEffector(effector.Effector):
    """
    Effector used to update the state of tracked objects.
    """

    def __init__(self, plane: mjcf.Element, obj: list[mjcf.Element]) -> None:
        self._plane = plane
        self._object = obj
        self._spec = None
        self._deadtime = 0

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
        if physics.time() - self._deadtime < DEADTIME:
            return
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
                self._deadtime = physics.time()
                break
        # only update scene if object is not in contact with element other than plane
        if update:
            # update object
            physics.bind(self._object).qpos[:] = command[1:]
            physics.bind(self._object).qvel[:] = np.zeros(3)
            # update plane
            physics.bind(self._plane).qpos[:] = command[0]
            physics.bind(self._plane).qvel[:] = 0


ENDEFFECTOR_XML_PATH = (
    pathlib.Path(__file__).parent / ".." / "assets" / "endeffector.xml"
)


class EndEffector(gripper.DummyHand):
    def _build(self, name: str = "endeffector") -> None:
        self._mjcf_root = mjcf.from_path(ENDEFFECTOR_XML_PATH)
        self._mjcf_root.model = name
        self._tool_center_point = self._mjcf_root.find("site", "TCP")


def make_endeffector(name: str) -> params.GripperParams:
    """Creates the endeffector."""
    name = f"{name}_gripper"
    gripper_model = EndEffector()
    gripper_effector = None
    return params.GripperParams(gripper_model, gripper_effector)


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
