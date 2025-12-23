from rai_perception.tools import GetDistanceToObjectsTool
from rai.communication.ros2 import ROS2Connector, ROS2Context
import time

with ROS2Context():
    connector=ROS2Connector(node_name="test_node")
    connector.node.declare_parameter("conversion_ratio", 1.0)  # scale parameter for the depth map

    # Wait for topic discovery to complete
    print("Waiting for topic discovery...")
    time.sleep(3)

    x = GetDistanceToObjectsTool(connector=connector)._run(
        camera_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/depth/image_rect_raw",
        object_names=["desk"],
    )

    print(x)
