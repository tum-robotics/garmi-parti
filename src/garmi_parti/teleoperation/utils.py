"""
Utility module.
"""

from __future__ import annotations

import collections
import os
import time

from . import containers


class TeleopTimeoutError(RuntimeError):
    """
    Teleoperation network timeout error.
    """


class Timer:
    """
    Utility class to track timeouts.
    """

    def __init__(self, timestep: float, timeout: float, buffer_size: int = 100):
        self.timestep = timestep
        self.timeout = timeout
        self.last_tick_time: float | None = None
        self.buffer: collections.deque[float] = collections.deque(maxlen=buffer_size)
        self.last_sleep_time: float = 0

    def tick(self) -> None:
        """
        Tick the internal clock.
        This function will keep track of the time between calls
        and block if necessary to keep the required timestep interval.
        """
        current_time = time.perf_counter()

        if self.last_tick_time is not None:
            time_delta = current_time - self.last_tick_time
            self.buffer.append(time_delta - self.last_sleep_time)
            average_time_delta = (
                sum(self.buffer) / len(self.buffer) if self.buffer else 0.0
            )
            sleep_time = max(0, self.timestep - average_time_delta)
            self.last_sleep_time = sleep_time
            time.sleep(sleep_time)

        self.last_tick_time = current_time

    def check_timeout(self) -> None:
        """
        Check whether a timeout occurred since the last call to `tick`.
        """
        current_time = time.perf_counter()

        if self.last_tick_time is not None:
            time_since_last_tick = current_time - self.last_tick_time
            if time_since_last_tick > self.timeout:
                raise TeleopTimeoutError()


def compute_displacement(
    container: containers.TeleopContainer,
) -> containers.Displacement:
    """
    Compute the displacement between the teleoperator's
    initial and its current pose.

    Args:
      container: The teleoperator container.

    Returns:
      Displacement: Container holding the relative pose.
    """
    return containers.Displacement(
        linear=container.transform.apply(
            container.arm.get_position() - container.position_init
        ),
        angular=container.transform
        * container.get_rotation()
        * container.orientation_init_inv
        * container.transform_inv,
    )


def get_robot_hostnames(required: bool = True) -> tuple[str, str]:
    """
    Retrieve the left and right robot ips (hostnames) from
    the environment variables `PANDA_LEFT` and `PANDA_RIGHT`.
    """
    left, right = os.environ.get("PANDA_LEFT"), os.environ.get("PANDA_RIGHT")
    if required and (left is None or right is None):
        raise RuntimeError(
            "Please make sure the environment variables "
            + "PANDA_LEFT and PANDA_RIGHT are set to the respective robot hostnames."
        )
    return left, right  # type: ignore[return-value]
