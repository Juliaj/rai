<!--- --8<-- [start:sec1] -->

# RAI Perception

## Overview

**RAI Perception** is a ROS2 package that integrates GroundingDINO and Segment Anything models for open-set object detection and segmentation. It provides:

- **Agents**: `GroundingDinoAgent` and `GroundedSamAgent` exposed as ROS2 services
- **Tools for LLM agents**: 
  - `GetDetectionTool` - detects objects from camera topics
  - `GetDistanceToObjectsTool` - estimates distance to detected objects using depth data
  - `GetSegmentationTool` and `GetGrabbingPointTool` for manipulation tasks


## Installation

The `rai-perception` package can be installed directly via pip:

```bash
pip install rai-perception
```

## Development Setup

To develop or contribute to `rai_perception`, follow these steps to set up your workspace:

### Prerequisites

Ensure your ROS2 workspace has an `src` folder containing both the `rai_perception` and `rai_interfaces` packages.

### Install Dependencies

Install required ROS2 dependencies:

```bash
rosdep install --from-paths src --ignore-src -r
```

In the base directory of your RAI workspace, install Python dependencies:

```bash
poetry install --with perception
```

### Build the Package

Source your ROS2 installation:

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
```

Build the workspace:

```bash
colcon build --symlink-install
```

Source the workspace environment:

```bash
source setup_shell.sh
```

### Run the Agents

Launch the `GroundedSamAgent` and `GroundingDinoAgent`, change the directory path based on your setup.

```bash
python rai_perception/scripts/run_perception_agents.py
```

<!--- --8<-- [end:sec1] -->

This script creates two ROS 2 Nodes: `grounding_dino` and `grounded_sam` using [ROS2Connector](../../../docs/API_documentation/connectors/ROS_2_Connectors.md).
The services exposed by agents can be triggered as regular ROS2 services:

-   `grounding_dino_classify`: `rai_interfaces/srv/RAIGroundingDino`
-   `grounded_sam_segment`: `rai_interfaces/srv/RAIGroundedSam`

> [!TIP]
>
> If you wish to start rai_perception agents as part of ros2 launch, an example launch
> file can be found in `rai/src/rai_bringup/launch/openset.launch.py`

> [!NOTE]
> The model weights will be downloaded to `~/.cache/rai/vision/` directory.

## RAI Tools

`rai_perception` package contains tools that can be used by [RAI LLM agents](../../../docs/tutorials/walkthrough.md)
enhance their perception capabilities. For more information on RAI Tools see
[Tool use and development](../../../docs/tutorials/tools.md) tutorial.

<!--- --8<-- [start:sec3] -->

### `GetDetectionTool`

This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt. A tool for detecting specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result. 

<!--- --8<-- [end:sec3] -->

> [!TIP]
>
> you can try example below with [rosbotxl demo](../../../docs/demos/rosbot_xl.md) binary.
> The binary exposes `/camera/camera/color/image_raw` and `/camera/camera/depth/image_raw` topics.

<!--- --8<-- [start:sec4] -->

**Example call**

```python
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    x = GetDetectionTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        object_names=["chair", "human", "plushie", "box", "ball"],
    )
```

**Example output**

```
I have detected the following items in the picture - chair, human
```

### `GetDistanceToObjectsTool`

This tool calls the grounding dino service to use the model to see if the message from the provided camera topic contains objects from a comma separated prompt. Then it utilises messages from depth camera to create an estimation of distance to a detected object. A tool for calculating distance to specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."


**Example call**

```python
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    connector.node.declare_parameter("conversion_ratio", 1.0) # scale parameter for the depth map
    x = GetDistanceToObjectsTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/depth/image_rect_raw",
        object_names=["chair", "human", "plushie", "box", "ball"],
    )

```

**Example output**

```
I have detected the following items in the picture human: 3.77m away
```


## Interact with agents with a simple ROS2 client

You can also use a simple ROS provided with the package as `rai_perception/talker.py` to interact with the agents. Following are the steps to run this client.  

First, make sure the ROS2 packages in your workspace are built and sourced. Then run following from one terminal, adjust the directory path based on your setup

```
python src/rai_extensions/rai_perception/scripts/run_perception_agents.py

```
Finally run following in another termainal:

```
ros2 run rai_perception talker --ros-args -p image_path:=src/rai_extensions/rai_perception/images/sample.jpg
```

If everything was set up properly you should see a couple of detections with classes `dinosaur`, `dragon`, and `lizard`.

<!--- --8<-- [end:sec4] -->



