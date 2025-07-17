# Manipulation Demo Debugging Strategy

## Problem Description
The manipulation demo arm moves in simulation but fails to grab any cubes or swap them when given the prompt "swap any two cubes".

## Root Cause Analysis

Based on the code analysis, the issue likely stems from one or more of these areas:

### 1. Object Detection Issues
- **Grounding DINO** may not be detecting cubes properly
- **Grounded SAM** may not be generating accurate segmentation masks
- **Threshold values** might be too high/low for the current scene

### 2. Depth and Coordinate Issues
- **Depth camera calibration** or conversion ratio problems
- **Coordinate frame transformations** between camera and manipulator frames
- **Point cloud processing** in grabbing point calculation

### 3. Grabbing Point Calculation Issues
- **Centroid calculation** may be inaccurate
- **Gripper rotation** calculation problems
- **Mask processing** issues in the `_process_mask` function

### 4. Manipulation Tool Issues
- **MoveObjectFromToTool** timing or coordinate problems
- **Gripper state management** issues
- **Service timeouts** or communication problems

## Debugging Strategy

### Phase 1: Basic System Health Check
1. **Run the debug script** to verify all components are working:
   ```bash
   python debug_manipulation_demo.py
   ```

2. **Check ROS2 topics and services**:
   ```bash
   # Check if topics are publishing
   ros2 topic list
   ros2 topic echo /color_image5 --once
   ros2 topic echo /depth_image5 --once
   
   # Check if services are available
   ros2 service list | grep -E "(grounding|grounded|manipulator)"
   ```

### Phase 2: Object Detection Debugging

#### 2.1 Test Grounding DINO Detection
```bash
# Test with different object names
ros2 service call /grounding_dino_classify rai_interfaces/srv/RAIGroundingDino "{source_img: <image>, classes: 'cube', box_threshold: 0.25, text_threshold: 0.35}"
ros2 service call /grounding_dino_classify rai_interfaces/srv/RAIGroundingDino "{source_img: <image>, classes: 'block', box_threshold: 0.25, text_threshold: 0.35}"
ros2 service call /grounding_dino_classify rai_interfaces/srv/RAIGroundingDino "{source_img: <image>, classes: 'object', box_threshold: 0.25, text_threshold: 0.35}"
```

#### 2.2 Test Grounded SAM Segmentation
- Verify that Grounded SAM receives detections from Grounding DINO
- Check if segmentation masks are being generated
- Validate mask quality and coverage

#### 2.3 Visual Inspection
- Use the `GetROS2ImageConfiguredTool` to capture and save images
- Manually inspect if cubes are visible in the camera view
- Check if camera positioning is optimal for object detection

### Phase 3: Coordinate and Depth Debugging

#### 3.1 Check Camera Calibration
```bash
# Verify camera info parameters
ros2 topic echo /color_camera_info5 --once
```

#### 3.2 Test Depth Conversion
- Verify `conversion_ratio` parameter is correct (default: 1.0)
- Check if depth values are reasonable (should be in meters)
- Validate point cloud generation from depth data

#### 3.3 Test Coordinate Transformations
- Verify TF transforms between `RGBDCamera5` and `panda_link0` frames
- Check if grabbing points are in reasonable coordinates
- Validate that points are within the manipulator's workspace

### Phase 4: Grabbing Point Debugging

#### 4.1 Test GetGrabbingPointTool Directly
```python
from rai_open_set_vision.tools import GetGrabbingPointTool
from rai.communication.ros2.connectors import ROS2Connector

connector = ROS2Connector()
tool = GetGrabbingPointTool(connector=connector)
results = tool._run(
    camera_topic="/color_image5",
    depth_topic="/depth_image5",
    camera_info_topic="/color_camera_info5", 
    object_name="cube"
)
print(f"Grabbing points: {results}")
```

#### 4.2 Analyze Grabbing Point Calculation
- Check if centroids are calculated correctly
- Verify gripper rotation angles are reasonable
- Ensure point cloud filtering is working properly

### Phase 5: Manipulation Tool Debugging

#### 5.1 Test MoveObjectFromToTool
```python
from rai.tools.ros2.manipulation import MoveObjectFromToTool

tool = MoveObjectFromToTool(connector=connector, manipulator_frame="panda_link0")
result = tool._run(x=0.3, y=0.1, z=0.1, x1=0.3, y1=-0.1, z1=0.1)
print(f"Move result: {result}")
```

#### 5.2 Check Service Communication
```bash
# Test manipulator service directly
ros2 service call /manipulator_move_to rai_interfaces/srv/ManipulatorMoveTo "{target_pose: {header: {frame_id: 'panda_link0'}, pose: {position: {x: 0.31, y: 0.0, z: 0.59}, orientation: {x: 0.9238795325112867, y: -0.3826834323650898, z: 0.0, w: 0.0}}}, initial_gripper_state: true, final_gripper_state: true}}"
```

### Phase 6: Integration Testing

#### 6.1 Test Complete Pipeline
1. Get object positions using `GetObjectPositionsTool`
2. Verify coordinates are reasonable
3. Test single object movement
4. Test object swapping

#### 6.2 Monitor Tool Execution
- Add logging to track tool execution flow
- Monitor service call success/failure
- Check for timeout issues

## Common Issues and Solutions

### Issue 1: No Objects Detected
**Symptoms**: Grounding DINO returns no detections
**Solutions**:
- Lower `box_threshold` and `text_threshold` values
- Try different object names ("cube", "block", "object")
- Check camera positioning and lighting
- Verify objects are visible in camera view

### Issue 2: Incorrect Grabbing Points
**Symptoms**: Grabbing points are outside workspace or at wrong height
**Solutions**:
- Check depth camera calibration
- Verify `conversion_ratio` parameter
- Validate coordinate frame transformations
- Check point cloud processing in `_process_mask`

### Issue 3: Manipulator Service Failures
**Symptoms**: Service calls timeout or fail
**Solutions**:
- Increase service timeout values
- Check if manipulator is in a valid state
- Verify target poses are reachable
- Check for collision detection issues

### Issue 4: Gripper Not Closing
**Symptoms**: Arm moves but doesn't grab objects
**Solutions**:
- Verify gripper state management in tools
- Check if gripper commands are being sent
- Validate gripper hardware/simulation state

## Debugging Tools

### 1. Debug Script
Use the provided `debug_manipulation_demo.py` script for comprehensive testing.

### 2. ROS2 Command Line Tools
```bash
# Monitor topics
ros2 topic echo /color_image5 --once
ros2 topic echo /depth_image5 --once

# Check services
ros2 service list
ros2 service type /grounding_dino_classify

# Monitor TF transforms
ros2 run tf2_tools view_frames
```

### 3. RViz for Visualization
```bash
# Launch RViz to visualize camera data and transforms
ros2 run rviz2 rviz2
```

## Expected Results

After successful debugging, you should see:
1. ✅ Grounding DINO detects cubes with reasonable bounding boxes
2. ✅ Grounded SAM generates accurate segmentation masks
3. ✅ GetGrabbingPointTool returns valid 3D coordinates
4. ✅ GetObjectPositionsTool returns positions in manipulator frame
5. ✅ Manipulator service responds successfully to movement commands
6. ✅ Gripper opens/closes as expected during manipulation

## Next Steps

1. Run the debug script and analyze results
2. Focus on the first failing component in the pipeline
3. Implement fixes based on the debugging findings
4. Test the complete pipeline again
5. Document any configuration changes needed

## Additional Resources

- [RAI OpenSet Vision Documentation](../docs/extensions/openset.md)
- [Manipulation Demo Documentation](../docs/demos/manipulation.md)
- [ROS2 Tools Documentation](../docs/API_documentation/langchain_integration/ROS_2_tools.md) 