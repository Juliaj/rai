#!/usr/bin/env python3
"""
Quick diagnostic script for manipulation demo issues.
Run this first to identify the most common problems.
"""

import rclpy
import time
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics

def quick_diagnostic():
    """Run quick diagnostic checks"""
    print("🔍 Quick Diagnostic for Manipulation Demo")
    print("=" * 50)
    
    try:
        # Initialize ROS2
        rclpy.init()
        connector = ROS2Connector(executor_type="single_threaded")
        node = connector.node
        node.declare_parameter("conversion_ratio", 1.0)
        
        print("✅ ROS2 initialized successfully")
        
        # Check 1: Required services
        print("\n1. Checking required services...")
        required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
        try:
            wait_for_ros2_services(connector, required_services, timeout_sec=10.0)
            print("✅ All required services are available")
        except Exception as e:
            print(f"❌ Service check failed: {e}")
            print("   Make sure the vision agents are running:")
            print("   python src/rai_extensions/rai_open_set_vision/scripts/run_vision_agents.py")
            return
        
        # Check 2: Required topics
        print("\n2. Checking required topics...")
        required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
        try:
            wait_for_ros2_topics(connector, required_topics, timeout_sec=10.0)
            print("✅ All required topics are publishing")
        except Exception as e:
            print(f"❌ Topic check failed: {e}")
            print("   Make sure the simulation is running and camera topics are active")
            return
        
        # Check 3: Camera data quality
        print("\n3. Checking camera data quality...")
        try:
            # Check color image
            color_msg = connector.receive_message("/color_image5", timeout_sec=5.0).payload
            print(f"✅ Color image: {color_msg.width}x{color_msg.height}")
            
            # Check depth image
            depth_msg = connector.receive_message("/depth_image5", timeout_sec=5.0).payload
            print(f"✅ Depth image: {depth_msg.width}x{depth_msg.height}")
            
            # Check camera info
            camera_info = connector.receive_message("/color_camera_info5", timeout_sec=5.0).payload
            print(f"✅ Camera info: fx={camera_info.k[0]:.1f}, fy={camera_info.k[4]:.1f}")
            
        except Exception as e:
            print(f"❌ Camera data check failed: {e}")
            return
        
        # Check 4: Object detection test
        print("\n4. Testing object detection...")
        try:
            from rai_interfaces.srv import RAIGroundingDino
            from rai.communication.ros2.ros_async import get_future_result
            
            # Get camera image
            camera_msg = connector.receive_message("/color_image5", timeout_sec=5.0).payload
            
            # Test Grounding DINO
            client = node.create_client(RAIGroundingDino, "/grounding_dino_classify")
            request = RAIGroundingDino.Request()
            request.source_img = camera_msg
            request.classes = "cube"
            request.box_threshold = 0.25  # Lower threshold for testing
            request.text_threshold = 0.35
            
            future = client.call_async(request)
            response = get_future_result(future, timeout_sec=10.0)
            
            if response and response.detections.detections:
                print(f"✅ Object detection: Found {len(response.detections.detections)} cube(s)")
                for i, detection in enumerate(response.detections.detections):
                    bbox = detection.bbox
                    print(f"   Cube {i+1}: bbox=({bbox.x:.1f}, {bbox.y:.1f}, {bbox.width:.1f}, {bbox.height:.1f})")
            else:
                print("⚠️ Object detection: No cubes detected")
                print("   Try different object names: 'block', 'object', 'box'")
                print("   Or lower the detection thresholds")
        
        except Exception as e:
            print(f"❌ Object detection test failed: {e}")
        
        # Check 5: Manipulator service
        print("\n5. Checking manipulator service...")
        try:
            from rai_interfaces.srv import ManipulatorMoveTo
            from geometry_msgs.msg import PoseStamped, Point, Quaternion
            
            client = node.create_client(ManipulatorMoveTo, "/manipulator_move_to")
            if client.wait_for_service(timeout_sec=5.0):
                print("✅ Manipulator service is available")
            else:
                print("❌ Manipulator service not available")
                print("   Make sure the robotic_manipulation node is running")
        
        except Exception as e:
            print(f"❌ Manipulator service check failed: {e}")
        
        # Check 6: TF transforms
        print("\n6. Checking coordinate transforms...")
        try:
            transform = connector.get_transform(
                target_frame="panda_link0", 
                source_frame="RGBDCamera5",
                timeout_sec=5.0
            )
            if transform:
                print("✅ Coordinate transform available")
                print(f"   Transform: {transform}")
            else:
                print("❌ Coordinate transform not available")
                print("   Check if TF is being published")
        
        except Exception as e:
            print(f"❌ Transform check failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 Quick diagnostic completed!")
        print("\nNext steps:")
        print("1. If any checks failed, address those issues first")
        print("2. Run the full debug script: python debug_manipulation_demo.py")
        print("3. Check the debugging strategy: MANIPULATION_DEBUG_STRATEGY.md")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    quick_diagnostic() 