"""
Teleoperation module including networking code for the client side.
"""
from __future__ import annotations

import contextlib
import logging
import socket
import threading
from xmlrpc import client

from . import interface, utils

_logger = logging.getLogger("teleoperation.client")

UDP_TIMEOUT = 2.0
UDP_TIMESTEP = 1e-3

socket.setdefaulttimeout(5.0)


class Client:
    def __init__(self, teleoperator: interface.Interface, host: str, port: int) -> None:
        self.rpc = client.ServerProxy(
            f"http://{host}:{port}", allow_none=True, use_builtin_types=True
        )
        self.host = host
        self.port = port
        self.teleoperator = teleoperator
        self.start()

    def run_udp(self) -> None:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.settimeout(1e-3)  # 1ms
        _logger.info("Sending UDP data to %s", (self.host, self.port))

        t = utils.Timer(UDP_TIMESTEP, UDP_TIMEOUT)
        t.tick()
        while self.running_udp:
            try:
                t.check_timeout()
                udp.sendto(self.teleoperator.get_command(), (self.host, self.port))
                self.teleoperator.set_command(udp.recv(1024))
                t.tick()
            except socket.timeout:
                pass
            except utils.TeleopTimeoutError:
                _logger.error("Server timed out.")
                break

    def pause(self) -> None:
        self.rpc.pause()
        self.teleoperator.pause()

    def unpause(self) -> None:
        if not self.rpc.synchronize(self.teleoperator.get_sync_command()):
            msg = "Synchronization failed"
            raise RuntimeError(msg)
        self.rpc.unpause()
        self.teleoperator.unpause()

    def open(self, end_effector: str = "") -> None:
        self.rpc.open(end_effector)

    def close(self, end_effector: str = "") -> None:
        self.rpc.close(end_effector)

    def shutdown(self) -> None:
        _logger.info("Shutting down teleoperation client")
        with contextlib.suppress(TimeoutError):
            self.rpc.stop()
        self.running_udp = False
        self.udp_thread.join()
        self.teleoperator.post_teleop()

    def start(self) -> None:
        _logger.info("Connecting to %s", self.host)

        if not self.rpc.connect():
            msg = "Connection failed"
            raise RuntimeError(msg)
        self.teleoperator.pre_teleop()
        if not self.rpc.synchronize(self.teleoperator.get_sync_command()):
            msg = "Synchronization failed"
            raise RuntimeError(msg)

        _logger.info("Ready for teleoperation.")
        self.rpc.start()
        self.teleoperator.start_teleop()

        self.running_udp = True
        self.udp_thread = threading.Thread(target=self.run_udp)
        self.udp_thread.start()

    def reset(self) -> None:
        self.shutdown()
        self.start()


def user_interface(cli: Client) -> None:
    """
    Waits for the user to type q and press enter to quit.
    The user can also pause teleop by pressing enter.
    """
    try:
        print("Press enter to quit. Type r and press enter to reset.\n")  # noqa:T201
        while True:
            inp = input()
            if inp == "r":
                cli.reset()
            else:
                break
    except KeyboardInterrupt:
        pass
