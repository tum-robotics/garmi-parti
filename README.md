# garmi-parti

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link] [![codecov][cov-badge]][cov-link]

Teleoperation framework for the GARMI and PARTI systems as well as the Franka robot in general. Includes support for Model-Mediated Teleoperation (MMT) with multi-body physics simulation. 

## Install

To install run

```
pip install garmi-parti
```

or if you're working with the code in a local clone of the repository

```
pip install -v -e .[dev]
```

## Getting Started
This package installs executables for various teleoperation configurations (two-arm, single-arm, MMT, etc.). Check out the [documentation](https://garmi-parti.readthedocs.io/en/latest/getting_started.html) for details on how to get started running the various teleoperation configurations.

## Requirements

The robots are controlled using
[panda-py](https://github.com/JeanElsner/panda-py), which is automatically
installed from pypi as part of the requirements. However, if you use an older
firmware or the FR3, you will need to manually install the correct version.

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/tum-robotics/garmi-parti/workflows/CI/badge.svg
[actions-link]:             https://github.com/tum-robotics/garmi-parti/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/garmi-parti
[conda-link]:               https://github.com/conda-forge/garmi-parti-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/tum-robotics/garmi-parti/discussions
[pypi-link]:                https://pypi.org/project/garmi-parti/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/garmi-parti
[pypi-version]:             https://img.shields.io/pypi/v/garmi-parti
[rtd-badge]:                https://readthedocs.org/projects/garmi-parti/badge/?version=latest
[rtd-link]:                 https://garmi-parti.readthedocs.io/en/latest/?badge=latest
[cov-badge]:                https://img.shields.io/codecov/c/gh/tum-robotics/garmi-gui
[cov-link]:                 https://app.codecov.io/gh/tum-robotics/garmi-gui

<!-- prettier-ignore-end -->
