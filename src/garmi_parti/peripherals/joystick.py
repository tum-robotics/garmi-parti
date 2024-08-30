from __future__ import annotations

import logging
import threading
import time

import serial
import serial.tools.list_ports

_logger = logging.getLogger("joystick")


class SerialJoysticks:
    """Serial interface for custom joystick."""

    def __init__(self) -> None:
        self.threads: list[threading.Thread] = []
        self.stop_threads = threading.Event()
        self.previous_states: dict[tuple[str, str], str] = {}

    def _parse_line(self, line: str, device_id: str) -> None:
        values = line.split(",")
        changes: list[tuple[str, str] | tuple[str, str, int]] = []
        for value in values:
            key, state = value.split("=")
            if key in ["R", "G", "B", "Y", "T"]:
                previous_state = self.previous_states.get((device_id, key), None)
                if (
                    previous_state is not None
                    and previous_state == "1"
                    and state == "0"
                ):
                    changes.append((key, device_id))
                self.previous_states[(device_id, key)] = state
            elif key == "A":
                try:
                    state_int = int(state)
                    previous_state = self.previous_states.get((device_id, key), None)
                    if previous_state is not None:
                        previous_state_int = int(previous_state)
                        if previous_state_int < 560 != state_int < 560:
                            changes.append((key, device_id, state_int))
                    self.previous_states[(device_id, key)] = state
                except ValueError:
                    _logger.error(
                        "Invalid integer value for A on %s: %s", device_id, state
                    )

        if changes:
            self.handle_changes(changes)

    def handle_changes(
        self, changes: list[tuple[str, str] | tuple[str, str, int]]
    ) -> None:
        """Called periodically with a list of button state changes.
        Override this function to attach functionality to the buttons."""
        # User-defined callback to handle changes
        for change in changes:
            if len(change) == 2:
                key, device_id = change
                _logger.info("Button %s pressed on device %s", key, device_id)
            elif len(change) == 3:
                key, device_id, state = change
                _logger.info(
                    "Analog trigger %s crossed threshold on device %s, new state: %d",
                    key,
                    device_id,
                    state,
                )

    def find_ports(self) -> list[str]:
        """Returns a list of potential joystick com ports."""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports if "ttyUSB" in port.device]

    def read_id(self, port: str | None, lines_to_ignore: int = 3) -> str | None:
        """Tries to read the joystick id from the given com port."""
        try:
            with serial.Serial(port, 9600, timeout=1) as com:
                com.readlines(lines_to_ignore)
                line = com.readline().decode("utf-8", errors="ignore").strip()
                if "ID=left" in line:
                    return "left"
                if "ID=right" in line:
                    return "right"
        except serial.SerialException as e:
            logging.error("Serial exception on %s: %s", port, e)
        except UnicodeDecodeError as e:
            logging.error("Unicode exception while reading ID from %s: %s", port, e)
        return None

    def start(self) -> None:
        """Finds all connected joystick devices and reads from them.
        Changes in the button states are processed in :py:func:`handle_changes`."""
        # User-defined callback to handle changes""
        com_ports = self.find_ports()
        for port in com_ports:
            device_id = self.read_id(port)
            if device_id in ["left", "right"]:
                thread = threading.Thread(target=self._run, args=(port, device_id))
                thread.start()
                self.threads.append(thread)
                logging.info("Started reading from %s on %s", device_id, port)

    def _run(self, port: str, device_id: str) -> None:
        try:
            with serial.Serial(port, 9600, timeout=0.1) as com:
                while not self.stop_threads.is_set():
                    try:
                        line = com.readline().decode("utf-8", errors="ignore").strip()
                        if line:
                            self._parse_line(line, device_id)
                    except UnicodeDecodeError as e:
                        logging.error("Decode error on %s: %s", port, e)
                    time.sleep(0.01)  # Small sleep to reduce CPU usage
        except serial.SerialException as e:
            logging.error("Serial exception on %s: %s", port, e)

    def close(self) -> None:
        """Stop threads reading from com ports."""
        self.stop_threads.set()
        for thread in self.threads:
            thread.join()
        logging.info("Closed all ports and stopped threads")


if __name__ == "__main__":
    reader = SerialJoysticks()
    reader.start()
    try:
        while True:
            time.sleep(1)  # Main thread sleep to reduce CPU usage
    except KeyboardInterrupt:
        reader.close()
