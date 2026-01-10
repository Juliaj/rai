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
# See the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

import pytest
from rai_perception.agents.grounded_sam import GroundedSamAgent

from rai_interfaces.srv import RAIGroundedSam
from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
from tests.rai_perception.conftest import (
    create_valid_weights_file,
)
from tests.rai_perception.test_helpers import (
    create_segmentation_request,
    create_test_detection2d,
    get_default_segmentation_weights_path,
    get_segmentation_weights_path,
    patch_segmentation_agent_dependencies,
    patch_segmentation_agent_dependencies_default_path,
)
from tests.rai_perception.test_mocks import EmptySegmenter, MockGDSegmenter


class TestGroundedSamAgent:
    """Test cases for GroundedSamAgent.

    Note: All tests patch ROS2Connector to prevent hanging. BaseVisionAgent.__init__
    creates a real ROS2Connector which requires ROS2 to be initialized, so we patch
    it to use a mock instead for unit testing.
    """

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test GroundedSamAgent initialization."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            assert agent._service._segmenter is not None

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_init_default_path(self, mock_connector):
        """Test GroundedSamAgent initialization with default path."""
        weights_path = get_default_segmentation_weights_path()
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        create_valid_weights_file(weights_path)

        with patch_segmentation_agent_dependencies_default_path(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(ros2_name="test")

            assert agent._service._segmenter is not None

            cleanup_agent(agent)
            weights_path.unlink()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            with patch.object(
                agent.ros2_connector, "create_service"
            ) as mock_create_service:
                agent.run()

                mock_create_service.assert_called_once()

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_segment_callback(self, tmp_path, mock_connector):
        """Test segment callback processes request correctly."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, MockGDSegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            # Create mock request
            detection1 = create_test_detection2d(30.0, 30.0, 40.0, 40.0)
            detection2 = create_test_detection2d(75.0, 75.0, 30.0, 30.0)
            request = create_segmentation_request([detection1, detection2])
            response = RAIGroundedSam.Response()

            # Call callback via service
            result = agent._service._segment_callback(request, response)

            # Verify response contains masks
            assert len(result.masks) == 2
            assert result is response

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_segment_callback_empty_detections(self, tmp_path, mock_connector):
        """Test segment callback with empty detections."""
        weights_path = get_segmentation_weights_path(tmp_path)

        with patch_segmentation_agent_dependencies(
            mock_connector, EmptySegmenter, weights_path
        ):
            agent = GroundedSamAgent(weights_root_path=str(tmp_path), ros2_name="test")

            request = create_segmentation_request()
            response = RAIGroundedSam.Response()
            result = agent._service._segment_callback(request, response)

            assert len(result.masks) == 0

            cleanup_agent(agent)
