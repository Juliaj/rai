pyenv local 3.12.3
source setup_shell.sh
ros2 launch examples/manipulation-demo.launch.py game_launcher:=demo_assets/manipulation/RAIManipulationDemo/RAIManipulationDemo.GameLauncher
