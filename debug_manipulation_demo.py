#!/usr/bin/env python3
"""
Debug script for manipulation demo issues.
This script helps diagnose problems with object detection and grabbing in the manipulation demo.
"""

import logging
import rclpy
import numpy as np
from typing import List, Optional
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import Image, CameraInfo
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.api import convert_ros_img_to_ndarray
from rai_open_set_vision.tools import GetGrabbingPointTool, GetDetectionTool
from rai_interfaces.srv import RAIGroundingDino, RAIGroundedSam
from rclpy.task import Future
from rai.communication.ros2.ros_async import get_future_result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManipulationDemoDebugger:
    def __init__(self):
        self.connector = None
        self.node = None
        
    def initialize(self):
        """Initialize ROS2 connection and wait for required services/topics"""
        logger.info("Initializing ROS2 connection...")
        rclpy.init()
        self.connector = ROS2Connector(executor_type="single_threaded")
        self.node = self.connector.node
        
        # Declare conversion ratio parameter
        self.node.declare_parameter("conversion_ratio", 1.0)
        
        # Wait for required services
        required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
        logger.info("Waiting for required services...")
        wait_for_ros2_services(self.connector, required_services)
        
        # Wait for required topics
        required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
        logger.info("Waiting for required topics...")
        wait_for_ros2_topics(self.connector, required_topics)
        
        logger.info("✅ All services and topics are available")
        
    def test_camera_topics(self):
        """Test if camera topics are publishing data"""
        logger.info("Testing camera topics...")
        
        topics_to_test = [
            ("/color_image5", Image),
            ("/depth_image5", Image), 
            ("/color_camera_info5", CameraInfo)
        ]
        
        for topic, msg_type in topics_to_test:
            try:
                msg = self.connector.receive_message(topic, timeout_sec=5.0).payload
                if isinstance(msg, msg_type):
                    logger.info(f"✅ {topic}: Received {msg_type.__name__}")
                    if hasattr(msg, 'width') and hasattr(msg, 'height'):
                        logger.info(f"   Image size: {msg.width}x{msg.height}")
                else:
                    logger.error(f"❌ {topic}: Expected {msg_type.__name__}, got {type(msg).__name__}")
            except Exception as e:
                logger.error(f"❌ {topic}: Failed to receive message - {e}")
                
    def test_grounding_dino_service(self):
        """Test Grounding DINO object detection service"""
        logger.info("Testing Grounding DINO service...")
        
        try:
            # Get a camera image
            camera_msg = self.connector.receive_message("/color_image5", timeout_sec=5.0).payload
            if not isinstance(camera_msg, Image):
                logger.error("❌ Failed to get camera image for testing")
                return
                
            # Create client
            client = self.node.create_client(RAIGroundingDino, "/grounding_dino_classify")
            if not client.wait_for_service(timeout_sec=5.0):
                logger.error("❌ Grounding DINO service not available")
                return
                
            # Create request
            request = RAIGroundingDino.Request()
            request.source_img = camera_msg
            request.classes = "cube"
            request.box_threshold = 0.35
            request.text_threshold = 0.45
            
            # Call service
            future = client.call_async(request)
            response = get_future_result(future, timeout_sec=10.0)
            
            if response is not None:
                detections = response.detections.detections
                logger.info(f"✅ Grounding DINO: Found {len(detections)} cube(s)")
                for i, detection in enumerate(detections):
                    bbox = detection.bbox
                    logger.info(f"   Detection {i+1}: bbox=({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height})")
            else:
                logger.error("❌ Grounding DINO service call failed")
                
        except Exception as e:
            logger.error(f"❌ Grounding DINO test failed: {e}")
            
    def test_grounded_sam_service(self):
        """Test Grounded SAM segmentation service"""
        logger.info("Testing Grounded SAM service...")
        
        try:
            # Get a camera image
            camera_msg = self.connector.receive_message("/color_image5", timeout_sec=5.0).payload
            if not isinstance(camera_msg, Image):
                logger.error("❌ Failed to get camera image for testing")
                return
                
            # First get detections from Grounding DINO
            dino_client = self.node.create_client(RAIGroundingDino, "/grounding_dino_classify")
            if not dino_client.wait_for_service(timeout_sec=5.0):
                logger.error("❌ Grounding DINO service not available")
                return
                
            dino_request = RAIGroundingDino.Request()
            dino_request.source_img = camera_msg
            dino_request.classes = "cube"
            dino_request.box_threshold = 0.35
            dino_request.text_threshold = 0.45
            
            dino_future = dino_client.call_async(dino_request)
            dino_response = get_future_result(dino_future, timeout_sec=10.0)
            
            if dino_response is None or len(dino_response.detections.detections) == 0:
                logger.warning("⚠️ No detections from Grounding DINO, skipping SAM test")
                return
                
            # Now test Grounded SAM
            sam_client = self.node.create_client(RAIGroundedSam, "/grounded_sam_segment")
            if not sam_client.wait_for_service(timeout_sec=5.0):
                logger.error("❌ Grounded SAM service not available")
                return
                
            sam_request = RAIGroundedSam.Request()
            sam_request.source_img = camera_msg
            sam_request.detections = dino_response.detections
            
            sam_future = sam_client.call_async(sam_request)
            sam_response = get_future_result(sam_future, timeout_sec=10.0)
            
            if sam_response is not None:
                masks = sam_response.masks
                logger.info(f"✅ Grounded SAM: Generated {len(masks)} mask(s)")
                for i, mask in enumerate(masks):
                    mask_array = convert_ros_img_to_ndarray(mask)
                    logger.info(f"   Mask {i+1}: shape={mask_array.shape}, unique_values={np.unique(mask_array)}")
            else:
                logger.error("❌ Grounded SAM service call failed")
                
        except Exception as e:
            logger.error(f"❌ Grounded SAM test failed: {e}")
            
    def test_get_grabbing_point_tool(self):
        """Test the GetGrabbingPointTool"""
        logger.info("Testing GetGrabbingPointTool...")
        
        try:
            tool = GetGrabbingPointTool(connector=self.connector)
            
            # Test with "cube" object
            results = tool._run(
                camera_topic="/color_image5",
                depth_topic="/depth_image5", 
                camera_info_topic="/color_camera_info5",
                object_name="cube"
            )
            
            if results:
                logger.info(f"✅ GetGrabbingPointTool: Found {len(results)} grabbing point(s)")
                for i, (centroid, rotation) in enumerate(results):
                    logger.info(f"   Point {i+1}: centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), rotation={rotation:.1f}°")
            else:
                logger.warning("⚠️ GetGrabbingPointTool: No grabbing points found")
                
        except Exception as e:
            logger.error(f"❌ GetGrabbingPointTool test failed: {e}")
            
    def test_get_object_positions_tool(self):
        """Test the GetObjectPositionsTool"""
        logger.info("Testing GetObjectPositionsTool...")
        
        try:
            from rai.tools.ros2.manipulation import GetObjectPositionsTool
            
            tool = GetObjectPositionsTool(
                connector=self.connector,
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
                get_grabbing_point_tool=GetGrabbingPointTool(connector=self.connector)
            )
            
            # Test with "cube" object
            result = tool._run("cube")
            logger.info(f"✅ GetObjectPositionsTool result: {result}")
            
        except Exception as e:
            logger.error(f"❌ GetObjectPositionsTool test failed: {e}")
            
    def test_manipulator_service(self):
        """Test the manipulator move service"""
        logger.info("Testing manipulator move service...")
        
        try:
            from rai_interfaces.srv import ManipulatorMoveTo
            from geometry_msgs.msg import PoseStamped, Point, Quaternion
            
            client = self.node.create_client(ManipulatorMoveTo, "/manipulator_move_to")
            if not client.wait_for_service(timeout_sec=5.0):
                logger.error("❌ Manipulator move service not available")
                return
                
            # Create a simple test pose (home position)
            request = ManipulatorMoveTo.Request()
            request.target_pose = PoseStamped()
            request.target_pose.header.frame_id = "panda_link0"
            request.target_pose.pose = Pose(
                position=Point(x=0.31, y=0.0, z=0.59),
                orientation=Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0)
            )
            request.initial_gripper_state = True  # open
            request.final_gripper_state = True   # keep open
            
            future = client.call_async(request)
            response = get_future_result(future, timeout_sec=20.0)
            
            if response is not None:
                if response.success:
                    logger.info("✅ Manipulator move service: Success")
                else:
                    logger.warning("⚠️ Manipulator move service: Failed")
            else:
                logger.error("❌ Manipulator move service: No response")
                
        except Exception as e:
            logger.error(f"❌ Manipulator service test failed: {e}")
            
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        logger.info("🚀 Starting comprehensive manipulation demo debugging...")
        
        try:
            self.initialize()
            
            # Run tests in logical order
            self.test_camera_topics()
            self.test_grounding_dino_service()
            self.test_grounded_sam_service()
            self.test_get_grabbing_point_tool()
            self.test_get_object_positions_tool()
            self.test_manipulator_service()
            
            logger.info("🎉 Comprehensive debugging completed!")
            
        except Exception as e:
            logger.error(f"❌ Debugging failed: {e}")
        finally:
            if self.connector:
                rclpy.shutdown()

def main():
    debugger = ManipulationDemoDebugger()
    debugger.run_comprehensive_test()

if __name__ == "__main__":
    main() 