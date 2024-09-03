Getting Started
---------------

Executables
===========

As an end-user you will probably mostly interact with the executable scripts
shipped with this package. If you're using the Docker version or installed the
package using ``pip``, the scripts will be installed to the environment's path
and can be called from the terminal.

The following table lists the currently available executables.

+-----------------------+--------------------------------------------------------+
| Executable            | Entry-point                                            |
+=======================+========================================================+
| parti-teleop          | :py:func:`garmi_parti.launchers.parti_teleop.main`     |
+-----------------------+--------------------------------------------------------+
| garmi-teleop          | :py:func:`garmi_parti.launchers.garmi_teleop.main`     |
+-----------------------+--------------------------------------------------------+
| panda-teleop-leader   | :py:func:`garmi_parti.launchers.panda_teleop.leader`   |
+-----------------------+--------------------------------------------------------+
| panda-teleop-follower | :py:func:`garmi_parti.launchers.panda_teleop.follower` |
+-----------------------+--------------------------------------------------------+
| parti-haptic-sim      | :py:func:`garmi_parti.launchers.parti_haptic_sim.main` |
+-----------------------+--------------------------------------------------------+
| parti-mmt             | :py:func:`garmi_parti.launchers.parti_mmt.main`        |
+-----------------------+--------------------------------------------------------+
| garmi-mmt             | :py:func:`garmi_parti.launchers.garmi_mmt.main`        |
+-----------------------+--------------------------------------------------------+

To find out more about a specific executable, check out the corresponding
entry-point or call :code:`<executable> --help` on the terminal.
