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
# See the License for the specific language governing permissions and
# limitations under the License.


import warnings
from pathlib import Path

import rclpy
from rai.agents import BaseAgent
from rai.communication.ros2 import ROS2Connector
from rclpy.parameter import Parameter

from rai_perception.services.detection_service import DetectionService

GDINO_NODE_NAME = "grounding_dino"
GDINO_SERVICE_NAME = "grounding_dino_classify"


class GroundingDinoAgent(BaseAgent):
    """Deprecated: Use DetectionService from rai_perception.services instead.

    This class is deprecated and will be removed in a future version.

    Architecture Note:
    - This was incorrectly named an "agent" - it's actually a ROS2 service node wrapper.
    - Real RAI agents (rai.agents.BaseAgent) are high-level abstractions that orchestrate
      behavior and use services/tools, not ROS2 service nodes themselves.
    - DetectionService is the correct abstraction for ROS2 detection service nodes.
    - If you need a real RAI agent that uses detection, create an agent that uses
      DetectionService as a tool/service, don't inherit from this class.

    This is a thin compatibility wrapper that delegates to DetectionService.
    """

    def __init__(
        self,
        weights_root_path: str | Path = Path.home() / Path(".cache/rai"),
        ros2_name: str = GDINO_NODE_NAME,
    ):
        warnings.warn(
            "GroundingDinoAgent is deprecated. Use DetectionService from "
            "rai_perception.services instead. DetectionService supports model-agnostic "
            "detection via ROS2 parameters.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

        # Create ROS2 connector to set parameters before service initialization
        self.ros2_connector = ROS2Connector(ros2_name, executor_type="single_threaded")

        # Set ROS2 parameters for the service
        self.ros2_connector.node.set_parameters(
            [
                Parameter(
                    "model_name",
                    rclpy.parameter.Parameter.Type.STRING,
                    "grounding_dino",
                ),
                Parameter(
                    "service_name",
                    rclpy.parameter.Parameter.Type.STRING,
                    GDINO_SERVICE_NAME,
                ),
            ]
        )

        # Create service instance with shared connector (it will read the parameters we just set)
        self._service = DetectionService(
            weights_root_path, ros2_name, ros2_connector=self.ros2_connector
        )
        self.logger = self._service.logger

    def run(self):
        """Delegate to the service."""
        self._service.run()

    def stop(self):
        """Delegate to the service."""
        self._service.stop()
