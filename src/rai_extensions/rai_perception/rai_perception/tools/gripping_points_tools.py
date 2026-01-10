# Copyright (C) 2025 Robotec.AI
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

import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.timeout import RaiTimeoutError, timeout

from rai_perception.components.gripping_points import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)
from rai_perception.components.topic_utils import (
    _suggest_topic_match,
    discover_camera_topics,
)

# Parameter prefix for ROS2 configuration
GRIPPING_POINTS_TOOL_PARAM_PREFIX = "perception.gripping_points"

logger = logging.getLogger(__name__)


class GetObjectGrippingPointsToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The name of the object to get the gripping point of e.g. 'box', 'apple', 'screwdriver'",
    )


class GetObjectGrippingPointsTool(BaseROS2Tool):
    name: str = "get_object_gripping_points"
    description: str = "Get gripping points for specified object/objects. Returns 3D coordinates where a robot gripper can grasp the object."

    # Configuration for PCL components
    segmentation_config: PointCloudFromSegmentationConfig = Field(
        default_factory=PointCloudFromSegmentationConfig,
        description="Configuration for point cloud segmentation from camera images",
    )
    estimator_config: GrippingPointEstimatorConfig = Field(
        default_factory=GrippingPointEstimatorConfig,
        description="Configuration for gripping point estimation strategies",
    )
    filter_config: PointCloudFilterConfig = Field(
        default_factory=PointCloudFilterConfig,
        description="Configuration for point cloud filtering and outlier removal",
    )

    # Auto-initialized in model_post_init from ROS2 parameters
    target_frame: Optional[str] = Field(
        default=None, description="Target coordinate frame for gripping points"
    )
    source_frame: Optional[str] = Field(
        default=None, description="Source coordinate frame of camera data"
    )
    camera_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera RGB images"
    )
    depth_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera depth images"
    )
    camera_info_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera calibration info"
    )
    timeout_sec: Optional[float] = Field(
        default=None, description="Timeout in seconds for gripping point detection"
    )
    conversion_ratio: Optional[float] = Field(
        default=0.001, description="Conversion ratio from depth units to meters"
    )

    # Components initialized in model_post_init
    gripping_point_estimator: Optional[GrippingPointEstimator] = Field(
        default=None, exclude=True
    )
    point_cloud_filter: Optional[PointCloudFilter] = Field(default=None, exclude=True)
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = Field(
        default=None, exclude=True
    )

    args_schema: Type[GetObjectGrippingPointsToolInput] = (
        GetObjectGrippingPointsToolInput
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool with ROS2 parameters and components."""
        self._load_parameters()
        self._initialize_components()

    def _load_parameters(self) -> None:
        """Load configuration from ROS2 parameters with auto-declaration and defaults."""
        node = self.connector.node
        param_prefix = GRIPPING_POINTS_TOOL_PARAM_PREFIX

        # Default values for parameters (deployment-specific, but common defaults)
        defaults = {
            "target_frame": "base_link",
            "source_frame": "camera_link",
            "camera_topic": "/camera/rgb/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "camera_info_topic": "/camera/rgb/camera_info",
            "timeout_sec": 10.0,
            "conversion_ratio": 0.001,
        }

        # Auto-declare parameters with defaults if not already set
        param_values = {}
        for param_key, default_value in defaults.items():
            param_name = f"{param_prefix}.{param_key}"
            if not node.has_parameter(param_name):
                node.declare_parameter(param_name, default_value)
                param_values[param_key] = default_value
                logger.info(
                    f"Auto-declared parameter '{param_name}' with default: {default_value}"
                )
            else:
                param_value = node.get_parameter(param_name).value
                param_values[param_key] = param_value
                if param_value != default_value:
                    logger.info(
                        f"Using overridden parameter '{param_name}': {param_value} (default: {default_value})"
                    )
                else:
                    logger.debug(
                        f"Using parameter '{param_name}': {param_value} (default)"
                    )

        # Load parameters
        self.target_frame = param_values["target_frame"]
        self.source_frame = param_values["source_frame"]
        self.camera_topic = param_values["camera_topic"]
        self.depth_topic = param_values["depth_topic"]
        self.camera_info_topic = param_values["camera_info_topic"]
        self.timeout_sec = param_values["timeout_sec"]
        self.conversion_ratio = param_values["conversion_ratio"]

        # Log summary of all parameters for observability
        logger.info(
            f"GetObjectGrippingPointsTool initialized with parameters:\n"
            f"  target_frame: {self.target_frame}\n"
            f"  source_frame: {self.source_frame}\n"
            f"  camera_topic: {self.camera_topic}\n"
            f"  depth_topic: {self.depth_topic}\n"
            f"  camera_info_topic: {self.camera_info_topic}\n"
            f"  timeout_sec: {self.timeout_sec}\n"
            f"  conversion_ratio: {self.conversion_ratio}"
        )

        # Early validation: check if topics exist and provide suggestions
        self._validate_topics_early()

    def _validate_topics_early(self) -> None:
        """Validate that required topics exist and provide suggestions if missing."""
        try:
            all_topics = [
                topic[0] for topic in self.connector.get_topics_names_and_types()
            ]
        except Exception:
            # If we can't query topics (e.g., ROS2 not fully initialized), skip validation
            logger.debug("Could not query topics for early validation, skipping")
            return

        required_topics = [
            self.camera_topic,
            self.depth_topic,
            self.camera_info_topic,
        ]
        missing = [t for t in required_topics if t not in all_topics]

        if missing:
            discovered = discover_camera_topics(self.connector)

            logger.warning(
                f"GetObjectGrippingPointsTool: Some required topics are not currently available:\n"
                f"  Missing topics: {missing}\n"
                f"  Note: Topics may not be available yet (this is an early check).\n"
                f"  If topics remain unavailable after waiting, check for topic name mismatches.\n"
                f"  To remap to your robot/simulation topics, set ROS2 parameters before tool initialization:\n"
                f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic', '/your/robot/camera/topic')\n"
                f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic', '/your/robot/depth/topic')\n"
                f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic', '/your/robot/camera_info/topic')"
            )

            # Provide suggestions for missing topics
            suggestions = []
            for missing_topic in missing:
                topic_lower = missing_topic.lower()
                if "depth" in topic_lower:
                    candidates = discovered["depth_topics"]
                    topic_type = "depth"
                elif "info" in topic_lower or "camera_info" in topic_lower:
                    candidates = discovered["camera_info_topics"]
                    topic_type = "camera_info"
                else:
                    candidates = discovered["image_topics"]
                    topic_type = "camera"

                if candidates:
                    # Find best match
                    suggestion = _suggest_topic_match(
                        missing_topic, candidates, topic_type
                    )
                    if suggestion:
                        suggestions.append(
                            f"    '{missing_topic}' -> try '{suggestion}'"
                        )
                    else:
                        suggestions.append(
                            f"    '{missing_topic}' -> available {topic_type} topics: {candidates[:3]}"
                        )

            if suggestions:
                logger.warning(
                    "  Available topics on your system:\n" + "\n".join(suggestions)
                )
        else:
            logger.debug("All required topics are available")

    def _initialize_components(self) -> None:
        """Initialize PCL components with loaded parameters."""
        self.point_cloud_from_segmentation = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            conversion_ratio=self.conversion_ratio,
            config=self.segmentation_config,
        )
        self.gripping_point_estimator = GrippingPointEstimator(
            config=self.estimator_config
        )
        self.point_cloud_filter = PointCloudFilter(config=self.filter_config)

    @property
    def detection_service_name(self) -> str:
        """Get the detection service name used by this tool."""
        return self.point_cloud_from_segmentation._get_detection_service_name()

    @property
    def segmentation_service_name(self) -> str:
        """Get the segmentation service name used by this tool."""
        return self.point_cloud_from_segmentation._get_segmentation_service_name()

    def get_config(self) -> Dict[str, Any]:
        """Get current ROS2 parameter configuration for observability.

        Returns:
            Dictionary mapping parameter names to their current values.
            Includes all deployment-specific parameters and service names.
        """
        return {
            "target_frame": self.target_frame,
            "source_frame": self.source_frame,
            "camera_topic": self.camera_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "timeout_sec": self.timeout_sec,
            "conversion_ratio": self.conversion_ratio,
            "detection_service_name": self.detection_service_name,
            "segmentation_service_name": self.segmentation_service_name,
        }

    def _run(self, object_name: str) -> str:
        @timeout(
            self.timeout_sec,
            f"Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds",
        )
        def _run_with_timeout():
            pcl = self.point_cloud_from_segmentation.run(object_name)
            if len(pcl) == 0:
                return f"No {object_name}s detected."

            pcl_filtered = self.point_cloud_filter.run(pcl)
            if len(pcl_filtered) == 0:
                return f"No {object_name}s detected after applying filtering"

            gripping_points = self.gripping_point_estimator.run(pcl_filtered)

            message = ""
            if len(gripping_points) == 0:
                message += f"No gripping point found for the object {object_name}\n"
            elif len(gripping_points) == 1:
                message += f"The gripping point of the object {object_name} is {gripping_points[0]}\n"
            else:
                message += (
                    f"Multiple gripping points found for the object {object_name}\n"
                )

            for i, gp in enumerate(gripping_points):
                message += (
                    f"The gripping point of the object {i + 1} {object_name} is {gp}\n"
                )

            return message

        try:
            return _run_with_timeout()
        except RaiTimeoutError as e:
            self.connector.node.get_logger().warning(f"Timeout: {e}")
            return f"Timeout: Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds"
        except Exception:
            raise
