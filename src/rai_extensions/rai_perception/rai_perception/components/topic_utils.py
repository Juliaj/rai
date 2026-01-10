# Copyright (C) 2025 Julia Jia
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
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector

logger = logging.getLogger(__name__)


def discover_camera_topics(connector: ROS2Connector) -> Dict[str, List[str]]:
    """Discover available camera-related topics in the ROS2 system.

    Searches for topics matching common camera naming patterns and categorizes them.

    Args:
        connector: ROS2 connector to query topics from

    Returns:
        Dictionary with keys:
        - "image_topics": List of image topics (sensor_msgs/Image)
        - "depth_topics": List of depth topics (sensor_msgs/Image)
        - "camera_info_topics": List of camera info topics (sensor_msgs/CameraInfo)
        - "all_topics": List of all available topics
    """
    try:
        all_topics = connector.get_topics_names_and_types()
    except Exception as e:
        logger.warning(f"Failed to query topics: {e}")
        return {
            "image_topics": [],
            "depth_topics": [],
            "camera_info_topics": [],
            "all_topics": [],
        }

    image_topics = []
    depth_topics = []
    camera_info_topics = []
    all_topic_names = []

    for topic_name, topic_types in all_topics:
        all_topic_names.append(topic_name)
        topic_types_str = " ".join(topic_types).lower()

        # Check for image topics
        if (
            "sensor_msgs/msg/image" in topic_types_str
            or "sensor_msgs/Image" in topic_types_str
        ):
            topic_lower = topic_name.lower()
            if "depth" in topic_lower or "depth" in topic_name:
                depth_topics.append(topic_name)
            else:
                image_topics.append(topic_name)

        # Check for camera info topics
        if (
            "sensor_msgs/msg/camerainfo" in topic_types_str
            or "sensor_msgs/CameraInfo" in topic_types_str
        ):
            camera_info_topics.append(topic_name)

    return {
        "image_topics": sorted(image_topics),
        "depth_topics": sorted(depth_topics),
        "camera_info_topics": sorted(camera_info_topics),
        "all_topics": sorted(all_topic_names),
    }


def _suggest_topic_match(
    expected: str, available: List[str], topic_type: str = "topic"
) -> Optional[str]:
    """Suggest a matching topic from available topics based on similarity.

    Args:
        expected: Expected topic name
        available: List of available topic names
        topic_type: Type of topic for logging ("image", "depth", "camera_info")

    Returns:
        Suggested topic name if a close match is found, None otherwise
    """
    if not available:
        return None

    expected_lower = expected.lower()
    expected_parts = set(re.split(r"[/_]", expected_lower))

    best_match = None
    best_score = 0

    for candidate in available:
        candidate_lower = candidate.lower()
        candidate_parts = set(re.split(r"[/_]", candidate_lower))

        # Calculate similarity based on common parts
        common_parts = expected_parts & candidate_parts
        score = len(common_parts) / max(len(expected_parts), len(candidate_parts))

        # Boost score if topic type keywords match
        if topic_type == "image" and (
            "color" in candidate_lower or "rgb" in candidate_lower
        ):
            score += 0.2
        elif topic_type == "depth" and "depth" in candidate_lower:
            score += 0.2
        elif topic_type == "camera_info" and (
            "info" in candidate_lower or "camera_info" in candidate_lower
        ):
            score += 0.2

        if score > best_score:
            best_score = score
            best_match = candidate

    # Only suggest if similarity is reasonable (>= 0.3)
    if best_score >= 0.3:
        return best_match
    return None


def _validate_topics_with_suggestions(
    connector: ROS2Connector, required_topics: List[str]
) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """Validate topics exist and provide suggestions for missing ones.

    Args:
        connector: ROS2 connector
        required_topics: List of required topic names

    Returns:
        Tuple of (missing_topics, suggestions_dict)
        where suggestions_dict maps missing topic -> suggested alternative
    """
    try:
        available_topics = [
            topic[0] for topic in connector.get_topics_names_and_types()
        ]
    except Exception:
        available_topics = []

    missing = [t for t in required_topics if t not in available_topics]
    suggestions = {}

    if missing:
        discovered = discover_camera_topics(connector)

        for missing_topic in missing:
            topic_lower = missing_topic.lower()
            if "depth" in topic_lower:
                suggestions[missing_topic] = _suggest_topic_match(
                    missing_topic, discovered["depth_topics"], "depth"
                )
            elif "info" in topic_lower or "camera_info" in topic_lower:
                suggestions[missing_topic] = _suggest_topic_match(
                    missing_topic, discovered["camera_info_topics"], "camera_info"
                )
            else:
                suggestions[missing_topic] = _suggest_topic_match(
                    missing_topic, discovered["image_topics"], "image"
                )

    return missing, suggestions


def wait_for_perception_dependencies(
    connector: ROS2Connector, tools: List[BaseTool]
) -> None:
    """Wait for ROS2 services and topics required by perception tools.

    Automatically extracts service names and topics from perception tools
    in the tools list and waits for them to be available. Provides helpful
    error messages with suggestions if topics/services are missing.

    Args:
        connector: ROS2 connector to use for waiting
        tools: List of tools that may include perception tools

    Raises:
        RuntimeError: If required perception tools are not found in tools list
        TimeoutError: If topics/services don't become available, with suggestions
    """
    # Lazy import to avoid circular dependency
    from rai_perception.tools.gripping_points_tools import GetObjectGrippingPointsTool

    # Extract service names from perception tools
    detection_service = None
    segmentation_service = None

    for tool in tools:
        if isinstance(tool, GetObjectGrippingPointsTool):
            detection_service = tool.detection_service_name
            segmentation_service = tool.segmentation_service_name
            break
        elif hasattr(tool, "service_name"):
            # For tools that only use detection service
            detection_service = tool.service_name
            break

    if detection_service is None or segmentation_service is None:
        raise RuntimeError(
            "Required perception tools not found in tools list. "
            "GetObjectGrippingPointsTool or tools with service_name property required."
        )

    required_services = [detection_service, segmentation_service]

    # Extract topics from perception tools
    required_topics = None
    for tool in tools:
        if isinstance(tool, GetObjectGrippingPointsTool):
            config = tool.get_config()
            required_topics = [
                config["camera_topic"],
                config["depth_topic"],
                config["camera_info_topic"],
            ]
            break

    if required_topics is None:
        raise RuntimeError(
            "GetObjectGrippingPointsTool not found in tools list. "
            "Cannot determine required topics."
        )

    # Early validation: check if topics exist now (tool already logs warnings)
    missing_topics, topic_suggestions = _validate_topics_with_suggestions(
        connector, required_topics
    )

    # Wait for services
    try:
        wait_for_ros2_services(connector, required_services)
    except TimeoutError as e:
        available_services = [s[0] for s in connector.get_services_names_and_types()]
        raise TimeoutError(
            f"{str(e)}\n"
            f"Available services: {sorted(available_services)[:10]}\n"
            f"Expected: {required_services}\n"
            f"Tip: Set ROS2 parameters '/detection_tool/service_name' and "
            f"'/segmentation_tool/service_name' to match your service names."
        ) from e

    # Wait for topics with enhanced error message
    try:
        wait_for_ros2_topics(connector, required_topics)
    except TimeoutError as e:
        # Recompute missing topics and suggestions at timeout
        missing_at_timeout, suggestions_at_timeout = _validate_topics_with_suggestions(
            connector, required_topics
        )
        discovered = discover_camera_topics(connector)
        suggestion_msg = ""
        for missing in missing_at_timeout:
            suggestion = suggestions_at_timeout.get(missing)
            if suggestion:
                suggestion_msg += f"\n  - '{missing}' -> try '{suggestion}'"

        raise TimeoutError(
            f"{str(e)}\n"
            f"Available image topics: {discovered['image_topics'][:5]}\n"
            f"Available depth topics: {discovered['depth_topics'][:5]}\n"
            f"Available camera_info topics: {discovered['camera_info_topics'][:5]}\n"
            f"Suggestions:{suggestion_msg}\n"
            f"Tip: Override topic parameters before tool initialization:\n"
            f"  node.declare_parameter('perception.gripping_points.camera_topic', '/your/topic')"
        ) from e
