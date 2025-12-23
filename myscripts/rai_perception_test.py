import time
from rai_perception.tools import GetDetectionTool
from rai.communication.ros2 import ROS2Connector, ROS2Context

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")

    # Wait for topic discovery to complete
    print("Waiting for topic discovery...")
    time.sleep(3)

    x = GetDetectionTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        object_names=["bed", "bed pillow", "table lamp", "plant", "desk"],
    )
    print(x)
