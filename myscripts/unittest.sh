# Ensure the workspace environment is sourced (so `rai` is importable).
source setup_shell.sh

# ROS2 launch_testing pytest plugins can prevent normal unit test collection.
# Disable third-party plugin autoload and run the test suite explicitly.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && python -m pytest tests/ -v

