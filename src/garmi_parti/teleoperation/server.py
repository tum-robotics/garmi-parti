"""
Networking teleoperation module for the server side.
"""

from __future__ import annotations

import contextlib
import logging
import socketserver
import threading
import time
from xmlrpc import server

from . import interfaces, utils

_logger = logging.getLogger("teleoperation.server")

UDP_TIMEOUT = 2.0
UDP_TIMESTEP = 0


class _UDPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server: _UDPServer
        self.server.t.tick()
        data, sock = self.request
        self.server.teleoperator.set_command(data)
        sock.sendto(self.server.teleoperator.get_command(), self.client_address)


class _UDPServer(socketserver.UDPServer):
    def __init__(
        self, server_address: tuple[str, int], teleoperator: interfaces.Interface
    ) -> None:
        self.teleoperator = teleoperator
        self.t = utils.Timer(UDP_TIMESTEP, UDP_TIMEOUT)
        self.t.tick()
        super().__init__(server_address, _UDPHandler, True)


class Server:
    """
    Teleoperation network server for a teleoperated robot that
    accepts connection from teleoperation clients.
    """

    def __init__(
        self, teleoperator: interfaces.Interface, port: int, udp_timeout: float = 1.0
    ) -> None:
        _logger.info("Starting teleoperation server")
        self.teleoperator = teleoperator
        self.port = port
        self.udp_timeout = udp_timeout
        self.udp_thread: threading.Thread | None = None
        self.udp: _UDPServer | None = None

        self.rpc = server.SimpleXMLRPCServer(
            ("0.0.0.0", port), allow_none=True, use_builtin_types=True
        )
        self.rpc.register_function(self._connect, "connect")
        self.rpc.register_function(self._synchronize, "synchronize")
        self.rpc.register_function(self._start, "start")
        self.rpc.register_function(self.pause)
        self.rpc.register_function(self.unpause)
        self.rpc.register_function(self.open)
        self.rpc.register_function(self.close)
        self.rpc.register_function(self._stop, "stop")

        self.rpc_thread = threading.Thread(target=self.rpc.serve_forever)
        self.rpc_thread.start()
        self.running = True
        self.check_timeout_thread = threading.Thread(target=self._check_timeout)
        self.check_timeout_thread.start()

    def _check_timeout(self) -> None:
        while self.running:
            if self.udp is not None:
                try:
                    self.udp.t.check_timeout()
                except utils.TeleopTimeoutError:
                    _logger.error("Client timed out.")
                    self._stop()
            time.sleep(1)

    def _connect(self) -> bool:
        self._stop()
        _logger.info("New teleoperation connection")
        return bool(self.teleoperator.pre_teleop())

    def _synchronize(self, command: bytes = b"", end_effector: str = "") -> bool:
        _logger.info("Synchronizing teleoperators")
        self.teleoperator.set_sync_command(command, end_effector)
        return True

    def _start(self) -> None:
        self.teleoperator.start_teleop()
        self.udp = _UDPServer(("0.0.0.0", self.port), self.teleoperator)
        _logger.info("Listening for UDP data on %s", ("0.0.0.0", self.port))
        self.udp_thread = threading.Thread(target=self.udp.serve_forever)
        self.udp_thread.start()

    def _stop(self) -> None:
        if self.udp is not None:
            _logger.info("Stopping UDP")
            self.udp.shutdown()
            self.udp.socket.close()
            self.udp = None
            if self.udp_thread is not None:
                self.udp_thread.join()
            self.teleoperator.post_teleop()

    def pause(self, end_effector: str = "") -> None:
        """
        Pause teleoperation service.
        """
        self.teleoperator.pause(end_effector)

    def unpause(self, end_effector: str = "") -> None:
        """
        Unpause teleoperation service.
        """
        self.teleoperator.unpause(end_effector)

    def open(self, end_effector: str = "") -> None:
        """
        Open an end-effector on the teleoperation interface.
        """
        self.teleoperator.open(end_effector)

    def close(self, end_effector: str = "") -> None:
        """
        Close an end-effector on the teleoperation interface.
        """
        self.teleoperator.close(end_effector)

    def shutdown(self) -> None:
        """
        Shutdown the server and end any running teleoperation.
        """
        self._stop()
        _logger.info("Shutting down teleoperation server")
        self.rpc.shutdown()
        self.rpc.socket.close()
        self.rpc_thread.join()
        self.running = False
        self.check_timeout_thread.join()


def user_interface(srv: Server) -> None:
    """
    Waits for the user to press enter or ctrl+c.
    """
    del srv
    with contextlib.suppress(KeyboardInterrupt):
        input("Press enter to quit\n")
