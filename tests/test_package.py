from __future__ import annotations

import importlib.metadata

import garmi_parti as m


def test_version():
    assert importlib.metadata.version("garmi_parti") == m.__version__
