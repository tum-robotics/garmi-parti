from __future__ import annotations

import argparse
import logging

from ..parti import mmt
from ..teleoperation import client

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("parti_mmt")


def main() -> None:
    """
    PARTI model-mediated teleoperation demo.
    The PARTI system acts as a network client and teleoperation
    leader that connects to a teleoperation server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=13701)
    args = parser.parse_args()

    leader = mmt.Leader()
    try:
        cli = client.Client(leader, args.host, args.port)
    except ConnectionRefusedError:
        leader.post_teleop()
        raise
    client.user_interface(cli)
    cli.shutdown()
