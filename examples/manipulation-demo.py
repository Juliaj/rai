# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language goveself.rning permissions and
# limitations under the License.


import logging
from typing import List

import rclpy
import rclpy.qos
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain.core import create_conversational_agent
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    ResetArmTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_open_set_vision.tools import GetGrabbingPointTool

from rai_whoami.models import EmbodimentInfo

logger = logging.getLogger(__name__)


def create_agent():
 
    rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")

    required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
    required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    wait_for_ros2_services(connector, required_services)

    wait_for_ros2_topics(connector, required_topics)

    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    tools: List[BaseTool] = [
        GetObjectPositionsTool(
            connector=connector,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
            get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
        ),
        MoveObjectFromToTool(connector=connector, manipulator_frame="panda_link0"),
        ResetArmTool(connector=connector, manipulator_frame="panda_link0"),
        GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5"),
    ]

    llm = get_llm_model(model_type="complex_model", streaming=True)
    
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )
    
    # Create enhanced system prompt to force object detection
    enhanced_prompt = f"""
{embodiment_info.to_langchain()}

IMPORTANT WORKFLOW FOR MANIPULATION TASKS:
1. ALWAYS start by using get_object_positions to detect objects in the scene
2. Use the detected positions to plan your manipulation
3. Only use move_object_from_to with actual detected object positions
4. Never use hardcoded coordinates without first detecting objects

OBJECT FILTERING GUIDELINES:
- When detecting objects, be aware that large surfaces like tables may be incorrectly identified as "cube"
- Filter out objects that are too large to be manipulatable cubes (e.g., tables, walls, floors)
- Focus on smaller, discrete objects that can actually be picked up and moved
- If you detect a very large "cube" that covers most of the scene, it's likely a table or surface and should be ignored
- Look for smaller, distinct objects that are appropriate for manipulation tasks

Available tools:
- get_object_positions: Use this FIRST to detect objects
- move_object_from_to: Use this to move objects between detected positions
- reset_arm: Use this if the arm gets stuck
- get_ros2_image: Use this to see the current camera view

For "swap any two cubes":
1. Call get_object_positions with "cube" to find cube positions
2. Filter out any large objects that are likely tables or surfaces
3. Select only smaller, manipulatable cubes for the task
4. Use the filtered positions in move_object_from_to
5. Do NOT use arbitrary coordinates like (0,1,0) or (1,0,0)
6. If only one small cube is found, inform the user that you need at least two manipulatable cubes

EXAMPLE FILTERING:
- If you detect: "cube: 0.48" covering most of the scene → This is likely a table, ignore it
- If you detect: "cube: 0.56" as a small object → This is a manipulatable cube, use it
- If you detect: "object: 0.36" as a small object → This might be a cube, consider it for manipulation
"""
    
    logger.info("🔧 [create_agent] Creating conversational agent...")
    agent = create_conversational_agent(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
    )
    logger.info("✅ [create_agent] Agent created successfully")
    return agent

import logging
import sys
def setup_debug_logging():
    """Setup comprehensive debug logging"""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('manipulation_debug.log')
        ]
    )
    
    # Set specific loggers to DEBUG level
    debug_loggers = [
        'rai_open_set_vision.tools.segmentation_tools',
        'rai.tools.ros2.manipulation.custom',
        'rai.communication.ros2',
        'rai.agents.langchain',
        'examples.manipulation_demo'
    ]
    
    for logger_name in debug_loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    # Enable ROS2 debug logging
    logging.getLogger('rclpy').setLevel(logging.INFO)
    
    print("🔧 Debug logging enabled!")
    print("📝 Logs will be saved to 'manipulation_debug.log'")
    print("=" * 60)

def main():
    logger.info("🚀 [main] Starting manipulation demo...")
    
    agent = create_agent()
    messages: List[BaseMessage] = []

    setup_debug_logging()
    logger.info("✅ [main] Debug logging setup complete")

    while True:
        try:
            prompt = input("Enter a prompt: ")
            logger.info(f"📝 [main] User prompt: '{prompt}'")
            
            messages.append(HumanMessage(content=prompt))
            logger.info("🔧 [main] Invoking agent...")
            
            output = agent.invoke({"messages": messages})
            logger.info("✅ [main] Agent response received")
            
            output["messages"][-1].pretty_print()
            
        except KeyboardInterrupt:
            logger.info("🛑 [main] User interrupted")
            break
        except Exception as e:
            logger.error(f"❌ [main] Error in main loop: {e}", exc_info=True)
            break


if __name__ == "__main__":
    main()
