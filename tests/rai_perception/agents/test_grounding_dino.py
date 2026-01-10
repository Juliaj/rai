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
from rai_perception.agents.grounding_dino import GroundingDinoAgent

from tests.rai_perception.agents.test_base_vision_agent import cleanup_agent
from tests.rai_perception.conftest import (
    create_valid_weights_file,
    setup_mock_clock,
)
from tests.rai_perception.test_helpers import (
    create_detection_request,
    get_default_detection_weights_path,
    get_detection_weights_path,
    patch_detection_agent_dependencies,
    patch_detection_agent_dependencies_default_path,
)
from tests.rai_perception.test_mocks import EmptyBoxer, MockGDBoxer


class TestGroundingDinoAgent:
    """Test cases for GroundingDinoAgent.

    Note: All tests patch ROS2Connector to prevent hanging. BaseVisionAgent.__init__
    creates a real ROS2Connector which requires ROS2 to be initialized, so we patch
    it to use a mock instead for unit testing.
    """

    @pytest.mark.timeout(10)
    def test_init(self, tmp_path, mock_connector):
        """Test GroundingDinoAgent initialization."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            assert agent._service._boxer is not None

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_init_default_path(self, mock_connector):
        """Test GroundingDinoAgent initialization with default path."""
        weights_path = get_default_detection_weights_path()
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        create_valid_weights_file(weights_path)

        with patch_detection_agent_dependencies_default_path(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(ros2_name="test")

            assert agent._service._boxer is not None

            cleanup_agent(agent)
            weights_path.unlink()

    @pytest.mark.timeout(10)
    def test_run_creates_service(self, tmp_path, mock_connector):
        """Test that run() creates the ROS2 service."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            with patch.object(
                agent.ros2_connector, "create_service"
            ) as mock_create_service:
                agent.run()

                mock_create_service.assert_called_once()

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_classify_callback(self, tmp_path, mock_connector):
        """Test classify callback processes request correctly."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, MockGDBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            # Create mock request
            from rai_interfaces.srv import RAIGroundingDino

            request = create_detection_request("dinosaur, dragon")
            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            # Call callback via service
            result = agent._service._classify_callback(request, response)

            # Verify response
            assert len(result.detections.detections) == 2
            assert result.detections.detection_classes == ["dinosaur", "dragon"]
            assert result is response

            cleanup_agent(agent)

    @pytest.mark.timeout(10)
    def test_classify_callback_empty_boxes(self, tmp_path, mock_connector):
        """Test classify callback with no detections."""
        weights_path = get_detection_weights_path(tmp_path)

        with patch_detection_agent_dependencies(
            mock_connector, EmptyBoxer, weights_path
        ):
            agent = GroundingDinoAgent(
                weights_root_path=str(tmp_path), ros2_name="test"
            )

            from rai_interfaces.srv import RAIGroundingDino

            request = create_detection_request("dinosaur")
            response = RAIGroundingDino.Response()

            setup_mock_clock(agent)

            result = agent._service._classify_callback(request, response)

            assert len(result.detections.detections) == 0
            assert result.detections.detection_classes == ["dinosaur"]

            cleanup_agent(agent)
