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

from typing import Any, List, Optional, Sequence, Type

import cv2
import numpy as np
import rclpy
import sensor_msgs.msg
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2.api import (
    convert_ros_img_to_base64,
    convert_ros_img_to_ndarray,
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.ros_async import get_future_result
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino
from rai_open_set_vision import GDINO_SERVICE_NAME

# --------------------- Inputs ---------------------


class GetSegmentationInput(BaseModel):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


class GetGrabbingPointInput(BaseModel):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    depth_topic: str = Field(
        ...,
        description="Ros2 topic for the depth image containing data to run distance calculations on",
    )
    camera_info_topic: str = Field(
        ...,
        description="Ros2 topic for the camera info to get the camera intrinsic from",
    )
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


# --------------------- Tools ---------------------
class GetSegmentationTool:
    connector: ROS2Connector = Field(..., exclude=True)

    name: str = ""
    description: str = ""

    box_threshold: float = Field(default=0.25, description="Box threshold for GDINO (lower = more detections)")
    text_threshold: float = Field(default=0.35, description="Text threshold for GDINO (lower = more matches)")

    args_schema: Type[GetSegmentationInput] = GetSegmentationInput

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        return get_future_result(future)

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        return get_future_result(future)

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                f"service {GDINO_SERVICE_NAME} not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        future = cli.call_async(req)
        return future

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ):
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service grounded_sam_segment not available, waiting again..."
            )
        req = RAIGroundedSam.Request()
        req.detections = data.detections
        req.source_img = camera_img_message
        future = cli.call_async(req)

        return future

    def _run(
        self,
        camera_topic: str,
        object_name: str,
    ):
        camera_img_msg = self._get_image_message(camera_topic)

        future = self._call_gdino_node(camera_img_msg, object_name)
        logger = self.connector.node.get_logger()
        try:
            conversion_ratio = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parameter conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        resolved = None
        while rclpy.ok():
            resolved = self._get_gdino_response(future)
            if resolved is not None:
                break

        assert resolved is not None
        future = self._call_gsam_node(camera_img_msg, resolved)

        ret = []
        while rclpy.ok():
            resolved = self._get_gsam_response(future)
            if resolved is not None:
                for img_msg in resolved.masks:
                    ret.append(convert_ros_img_to_base64(img_msg))
                break
        return "", {"segmentations": ret}


def depth_to_point_cloud(
    depth_image: np.ndarray, fx: float, fy: float, cx: float, cy: float
):
    height, width = depth_image.shape

    # Create grid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate 3D coordinates
    z = depth_image
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy

    # Stack the coordinates
    points = np.stack((x, y, z), axis=-1)

    # Reshape to a list of points
    points = points.reshape(-1, 3)

    # Remove points with zero depth
    points = points[points[:, 2] > 0]

    return points


class GetGrabbingPointTool(BaseTool):
    connector: ROS2Connector = Field(..., exclude=True)

    name: str = "GetGrabbingPointTool"
    description: str = "Get the grabbing point of an object"
    pcd: List[Any] = []

    args_schema: Type[GetGrabbingPointInput] = GetGrabbingPointInput
    box_threshold: float = Field(default=0.25, description="Box threshold for GDINO (lower = more detections)")
    text_threshold: float = Field(default=0.35, description="Text threshold for GDINO (lower = more matches)")

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        logger = self.connector.node.get_logger()
        logger.info("🔍 [GDINO] Waiting for response...")
        response = get_future_result(future, timeout_sec=30.0)  # 30 second timeout
        
        if response is not None:
            detections = response.detections.detections
            logger.info(f"🔍 [GDINO] Response received: {len(detections)} detection(s)")
            for i, detection in enumerate(detections):
                bbox = detection.bbox
                # logger.info(f"🔍 [GDINO] Detection {i+1}: bbox=({bbox.center.x:.1f}, {bbox.center.y:.1f}, {bbox.size_x:.1f}, {bbox.size_y:.1f})")
        else:
            # Check future status to distinguish between timeout and failure
            if future.done():
                if future.exception() is not None:
                    logger.error(f"🔍 [GDINO] Service call failed with exception: {future.exception()}")
                else:
                    logger.error("🔍 [GDINO] Service call completed but returned None")
            else:
                logger.error("🔍 [GDINO] Service call timed out after 30 seconds")
                # Save the image for debugging when timeout occurs
                self._save_debug_image()
        
        return response

    def _save_debug_image(self):
        """Save the camera image for debugging when GDINO times out"""
        if hasattr(self, '_debug_camera_image') and hasattr(self, '_debug_object_name'):
            try:
                import cv2
                import numpy as np
                import os
                from datetime import datetime
                
                logger = self.connector.node.get_logger()
                
                # Convert ROS image to OpenCV format
                img_array = convert_ros_img_to_ndarray(self._debug_camera_image)
                
                # Create debug directory if it doesn't exist
                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                
                # Generate filename with timestamp and object name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{debug_dir}/gdino_timeout_{self._debug_object_name}_{timestamp}.jpg"
                
                # Save image
                cv2.imwrite(filename, img_array)
                logger.info(f"🔍 [GDINO] Debug image saved: {filename}")
                logger.info(f"🔍 [GDINO] Image details: {img_array.shape}, dtype={img_array.dtype}, range=[{img_array.min()}, {img_array.max()}]")
                
                # Also save depth image if available
                if hasattr(self, '_debug_depth_image'):
                    try:
                        depth_array = convert_ros_img_to_ndarray(self._debug_depth_image)
                        
                        # Normalize depth image for visualization
                        if depth_array.dtype == np.uint16:
                            # Convert 16-bit depth to 8-bit for saving
                            depth_normalized = (depth_array.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
                        elif depth_array.dtype == np.float32:
                            # Normalize float depth to 8-bit
                            depth_min, depth_max = depth_array.min(), depth_array.max()
                            if depth_max > depth_min:
                                depth_normalized = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                            else:
                                depth_normalized = depth_array.astype(np.uint8)
                        else:
                            depth_normalized = depth_array
                        
                        depth_filename = f"{debug_dir}/gdino_timeout_depth_{self._debug_object_name}_{timestamp}.png"
                        cv2.imwrite(depth_filename, depth_normalized)
                        logger.info(f"🔍 [GDINO] Debug depth image saved: {depth_filename}")
                        logger.info(f"🔍 [GDINO] Depth image details: {depth_array.shape}, dtype={depth_array.dtype}, range=[{depth_array.min()}, {depth_array.max()}]")
                        logger.info(f"🔍 [GDINO] Normalized depth range: [{depth_normalized.min()}, {depth_normalized.max()}]")
                    except Exception as e:
                        logger.warning(f"🔍 [GDINO] Could not save depth image: {e}")
                        
            except Exception as e:
                logger.error(f"🔍 [GDINO] Failed to save debug image: {e}")
        else:
            logger.warning("🔍 [GDINO] No debug image available to save")

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        logger = self.connector.node.get_logger()
        logger.info("🎯 [GSAM] Waiting for response...")
        response = get_future_result(future, timeout_sec=30.0)  # 30 second timeout
        
        if response is not None:
            masks = response.masks
            logger.info(f"🎯 [GSAM] Response received: {len(masks)} mask(s)")
            for i, mask in enumerate(masks):
                logger.info(f"🎯 [GSAM] Mask {i+1}: encoding={mask.encoding}, width={mask.width}, height={mask.height}")
        else:
            # Check future status to distinguish between timeout and failure
            if future.done():
                if future.exception() is not None:
                    logger.error(f"🎯 [GSAM] Service call failed with exception: {future.exception()}")
                else:
                    logger.error("🎯 [GSAM] Service call completed but returned None")
            else:
                logger.error("🎯 [GSAM] Service call timed out after 30 seconds")
        
        return response

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        logger = self.connector.node.get_logger()
        logger.info(f"🔍 [GDINO] Starting object detection for '{object_name}'")
        logger.info(f"🔍 [GDINO] Image size: {camera_img_message.width}x{camera_img_message.height}")
        logger.info(f"🔍 [GDINO] Image encoding: {camera_img_message.encoding}")
        logger.info(f"🔍 [GDINO] Image step: {camera_img_message.step}")
        logger.info(f"🔍 [GDINO] Image data size: {len(camera_img_message.data)} bytes")
        logger.info(f"🔍 [GDINO] Thresholds: box={self.box_threshold}, text={self.text_threshold}")
        
        # Check if image data looks reasonable
        if len(camera_img_message.data) == 0:
            logger.error("🔍 [GDINO] ERROR: Image data is empty!")
            raise Exception("Empty image data")
        
        expected_size = camera_img_message.width * camera_img_message.height
        if camera_img_message.encoding in ['rgb8', 'bgr8']:
            expected_size *= 3
        elif camera_img_message.encoding in ['rgba8', 'bgra8']:
            expected_size *= 4
        
        if len(camera_img_message.data) < expected_size * 0.5:  # Allow some compression
            logger.warning(f"🔍 [GDINO] WARNING: Image data size seems small. Expected ~{expected_size}, got {len(camera_img_message.data)}")
        
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        wait_count = 0
        while not cli.wait_for_service(timeout_sec=1.0):
            wait_count += 1
            logger.warning(f"🔍 [GDINO] Service not available, waiting... (attempt {wait_count})")
            if wait_count > 10:
                logger.error("🔍 [GDINO] Service wait timeout exceeded!")
                raise Exception("Grounding DINO service not available after 10 attempts")
        
        logger.info("🔍 [GDINO] Service available, creating request...")
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        logger.info(f"🔍 [GDINO] Sending request with classes='{object_name}', box_threshold={self.box_threshold}, text_threshold={self.text_threshold}")
        future = cli.call_async(req)
        logger.info("🔍 [GDINO] Request sent, waiting for response...")
        
        # Store the image for potential debugging
        self._debug_camera_image = camera_img_message
        self._debug_object_name = object_name
        
        return future

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ):
        logger = self.connector.node.get_logger()
        logger.info(f"🎯 [GSAM] Starting segmentation with {len(data.detections.detections)} detection(s)")
        
        # Log detection details
        for i, detection in enumerate(data.detections.detections):
            bbox = detection.bbox
            # logger.info(f"🎯 [GSAM] Detection {i+1}: bbox=({bbox.center.x:.1f}, {bbox.center.y:.1f}, {bbox.size_x:.1f}, {bbox.size_y:.1f})")
        
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        wait_count = 0
        while not cli.wait_for_service(timeout_sec=1.0):
            wait_count += 1
            logger.warning(f"🎯 [GSAM] Service not available, waiting... (attempt {wait_count})")
            if wait_count > 10:
                logger.error("🎯 [GSAM] Service wait timeout exceeded!")
                raise Exception("Grounded SAM service not available after 10 attempts")
        
        logger.info("🎯 [GSAM] Service available, creating request...")
        req = RAIGroundedSam.Request()
        req.detections = data.detections
        req.source_img = camera_img_message
        
        logger.info(f"🎯 [GSAM] Sending segmentation request for {len(data.detections.detections)} detection(s)")
        future = cli.call_async(req)
        logger.info("🎯 [GSAM] Request sent, waiting for response...")
        return future

    def _get_camera_info_message(self, topic: str) -> sensor_msgs.msg.CameraInfo:
        for _ in range(3):
            msg = self.connector.receive_message(topic, timeout_sec=3.0).payload
            if isinstance(msg, sensor_msgs.msg.CameraInfo):
                return msg
            self.connector.node.get_logger().warn(
                "Received wrong message type. Retrying..."
            )

        raise Exception("Failed to receive correct CameraInfo message after 3 attempts")

    def _get_intrinsic_from_camera_info(self, camera_info: sensor_msgs.msg.CameraInfo):
        """Extract camera intrinsic parameters from the CameraInfo message."""

        fx = camera_info.k[0]  # Focal length in x-axis
        fy = camera_info.k[4]  # Focal length in y-axis
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y

        return fx, fy, cx, cy

    def _process_mask(
        self,
        mask_msg: sensor_msgs.msg.Image,
        depth_msg: sensor_msgs.msg.Image,
        intrinsic: Sequence[float],
        depth_to_meters_ratio: float,
    ):
        logger = self.connector.node.get_logger()
        logger.info(f"🔧 [ProcessMask] Starting mask processing...")
        
        # Convert mask to numpy array
        mask = convert_ros_img_to_ndarray(mask_msg)
        binary_mask = np.where(mask == 255, 1, 0)
        logger.info(f"🔧 [ProcessMask] Mask shape: {mask.shape}, binary mask sum: {np.sum(binary_mask)}")
        
        # Convert depth to numpy array
        depth = convert_ros_img_to_ndarray(depth_msg)
        logger.info(f"🔧 [ProcessMask] Depth shape: {depth.shape}, depth range: [{np.min(depth)}, {np.max(depth)}]")
        
        # Apply mask to depth
        masked_depth_image = np.zeros_like(depth, dtype=np.float32)
        masked_depth_image[binary_mask == 1] = depth[binary_mask == 1]
        masked_depth_image = masked_depth_image * depth_to_meters_ratio
        
        valid_depth_pixels = np.sum(masked_depth_image > 0)
        logger.info(f"🔧 [ProcessMask] Valid depth pixels after masking: {valid_depth_pixels}")
        
        if valid_depth_pixels == 0:
            logger.error("🔧 [ProcessMask] No valid depth pixels found in masked region!")
            raise Exception("No valid depth pixels in masked region")

        # Generate point cloud
        logger.info("🔧 [ProcessMask] Generating point cloud...")
        pcd = depth_to_point_cloud(
            masked_depth_image, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        )
        logger.info(f"🔧 [ProcessMask] Point cloud generated: {len(pcd)} points")

        # TODO: Filter out outliers
        points = pcd
        
        if len(points) == 0:
            logger.error("🔧 [ProcessMask] No points in point cloud!")
            raise Exception("Empty point cloud generated")

        # Calculate grasping parameters
        logger.info("🔧 [ProcessMask] Calculating grasping parameters...")
        grasp_z = points[:, 2].max()
        near_grasp_z_points = points[points[:, 2] > grasp_z - 0.008]
        logger.info(f"🔧 [ProcessMask] Grasp Z: {grasp_z:.3f}, near grasp points: {len(near_grasp_z_points)}")
        
        if len(near_grasp_z_points) == 0:
            logger.error("🔧 [ProcessMask] No near grasp points found!")
            raise Exception("No near grasp points found")
        
        xy_points = near_grasp_z_points[:, :2]
        xy_points = xy_points.astype(np.float32)
        _, dimensions, theta = cv2.minAreaRect(xy_points)
        
        logger.info(f"🔧 [ProcessMask] Min area rect: dimensions=({dimensions[0]:.3f}, {dimensions[1]:.3f}), theta={theta:.1f}°")

        gripper_rotation = theta
        # NOTE  - estimated dimentsion from the RGBDCamera5 not very precise, what may cause not desired rotation
        if dimensions[0] > dimensions[1]:
            gripper_rotation -= 90
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180
        
        logger.info(f"🔧 [ProcessMask] Final gripper rotation: {gripper_rotation:.1f}°")

        # Calculate full 3D centroid for OBJECT
        centroid = np.mean(points, axis=0)
        logger.info(f"🔧 [ProcessMask] Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        
        logger.info(f"✅ [ProcessMask] Successfully processed mask. Centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), rotation={gripper_rotation:.1f}°")
        return centroid, gripper_rotation

    def _run(
        self,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        object_name: str,
    ):
        logger = self.connector.node.get_logger()
        logger.info(f"🚀 [GetGrabbingPointTool] Starting for object: '{object_name}'")
        logger.info(f"🚀 [GetGrabbingPointTool] Topics: camera={camera_topic}, depth={depth_topic}, info={camera_info_topic}")
        
        # Get camera messages
        logger.info("📷 [GetGrabbingPointTool] Receiving camera messages...")
        camera_img_msg = self.connector.receive_message(camera_topic).payload
        depth_msg = self.connector.receive_message(depth_topic).payload
        camera_info = self._get_camera_info_message(camera_info_topic)
        
        logger.info(f"📷 [GetGrabbingPointTool] Camera image: {camera_img_msg.width}x{camera_img_msg.height}, encoding={camera_img_msg.encoding}, data_size={len(camera_img_msg.data)} bytes")
        logger.info(f"📷 [GetGrabbingPointTool] Depth image: {depth_msg.width}x{depth_msg.height}, encoding={depth_msg.encoding}, data_size={len(depth_msg.data)} bytes")
        
        # Check depth image quality
        if len(depth_msg.data) == 0:
            logger.error("📷 [GetGrabbingPointTool] ERROR: Depth image data is empty!")
            raise Exception("Empty depth image data")
        
        # Log depth image statistics
        try:
            import numpy as np
            from rai.communication.ros2.api import convert_ros_img_to_ndarray
            depth_array = convert_ros_img_to_ndarray(depth_msg)
            logger.info(f"📷 [GetGrabbingPointTool] Depth image stats: min={np.min(depth_array)}, max={np.max(depth_array)}, mean={np.mean(depth_array):.2f}")
            logger.info(f"📷 [GetGrabbingPointTool] Depth image non-zero pixels: {np.sum(depth_array > 0)}/{depth_array.size}")
            
            # Store depth image for debugging
            self._debug_depth_image = depth_msg
        except Exception as e:
            logger.warning(f"📷 [GetGrabbingPointTool] Could not analyze depth image: {e}")

        # Get camera intrinsics
        intrinsic = self._get_intrinsic_from_camera_info(camera_info)
        logger.info(f"📷 [GetGrabbingPointTool] Camera intrinsics: fx={intrinsic[0]:.1f}, fy={intrinsic[1]:.1f}, cx={intrinsic[2]:.1f}, cy={intrinsic[3]:.1f}")

        # Call Grounding DINO
        logger.info("🔍 [GetGrabbingPointTool] Calling Grounding DINO...")
        future = self._call_gdino_node(camera_img_msg, object_name)
        
        # Get conversion ratio
        try:
            conversion_ratio = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parameter conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        
        logger.info(f"📏 [GetGrabbingPointTool] Using conversion_ratio: {conversion_ratio}")

        # Get DINO response
        resolved = self._get_gdino_response(future)
        
        if resolved is None or len(resolved.detections.detections) == 0:
            logger.warning(f"⚠️ [GetGrabbingPointTool] No detections found for '{object_name}'")
            return []

        # Call Grounded SAM
        logger.info("🎯 [GetGrabbingPointTool] Calling Grounded SAM...")
        future = self._call_gsam_node(camera_img_msg, resolved)

        # Get SAM response
        resolved = self._get_gsam_response(future)
        
        if resolved is None or len(resolved.masks) == 0:
            logger.warning(f"⚠️ [GetGrabbingPointTool] No masks generated for '{object_name}'")
            return []

        # Process masks
        logger.info(f"🔧 [GetGrabbingPointTool] Processing {len(resolved.masks)} mask(s)...")
        rets = []
        for i, mask_msg in enumerate(resolved.masks):
            logger.info(f"🔧 [GetGrabbingPointTool] Processing mask {i+1}/{len(resolved.masks)}")
            try:
                result = self._process_mask(
                    mask_msg,
                    depth_msg,
                    intrinsic,
                    depth_to_meters_ratio=conversion_ratio,
                )
                rets.append(result)
                logger.info(f"✅ [GetGrabbingPointTool] Mask {i+1} processed successfully")
            except Exception as e:
                logger.error(f"❌ [GetGrabbingPointTool] Failed to process mask {i+1}: {e}")
                continue

        logger.info(f"🎉 [GetGrabbingPointTool] Completed. Generated {len(rets)} grabbing point(s)")
        return rets
