import h5py
import numpy as np
from PIL import Image

def check_depth_in_h5(file_path):
    """Check if H5 file contains depth image data."""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📁 File: {file_path}")
            print(f"🔍 Keys in H5 file:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  📊 {name}: shape={obj.shape}, dtype={obj.dtype}")
                    # Check if it looks like depth data
                    if len(obj.shape) == 2 and obj.dtype in ['uint16', 'int16', 'float32', 'float64']:
                        print(f"    ✅ Potential depth image: {obj.shape[0]}x{obj.shape[1]}")
                        # Show sample values
                        sample = obj[0:5, 0:5] if obj.size > 25 else obj[:]
                        print(f"    📏 Sample values: {sample}")
                elif isinstance(obj, h5py.Group):
                    print(f"  📁 {name}/ (group)")
            
            f.visititems(print_structure)
            
            # Look for common depth data keys
            depth_keys = ['depth', 'image', 'data', 'depth_image', 'z']
            found_depth = False
            
            for key in depth_keys:
                if key in f:
                    data = f[key]
                    if len(data.shape) == 2:
                        print(f"\n🎯 Found potential depth data in '{key}':")
                        print(f"   Shape: {data.shape}")
                        print(f"   Data type: {data.dtype}")
                        # Fix: Convert to numpy array first
                        data_array = np.array(data)
                        print(f"   Value range: {data_array.min()} to {data_array.max()}")
                        print(f"   Non-zero pixels: {np.count_nonzero(data_array)}")
                        found_depth = True
            
            if not found_depth:
                print("\n❌ No obvious depth image found")
                print("Available keys:", list(f.keys()))
                
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")\

def save_depth_image(file_path):
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key == 'depth':
                data = f[key]
                # Convert HDF5 Dataset to numpy array
                depth_array = np.array(data)
                depth_scaled = (depth_array * 255.0 / depth_array.max()).astype(np.uint8)
                Image.fromarray(depth_scaled).save('depth_visualization.png')
                print("✅ Depth image saved as 'depth_visualization.png'")
                break
            else:
                print(f"❌ No depth image found in {file_path}")

def find_camera_data(file_path):
    with h5py.File(file_path, 'r') as f:
        print('🔍 Searching for camera intrinsics in H5 file...')
        
        def search_recursive(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Look for camera-related data
                if any(keyword in name.lower() for keyword in ['camera', 'intrinsic', 'k', 'fx', 'fy', 'cx', 'cy', 'matrix']):
                    print(f'📷 Found camera data: {name}')
                    print(f'   Shape: {obj.shape}, dtype: {obj.dtype}')
                    if obj.size < 20:  # Small arrays might be camera params
                        print(f'   Values: {np.array(obj)}')
                # Look for 3x3 matrices (camera matrix)
                elif obj.shape == (3, 3):
                    print(f'📐 Found 3x3 matrix: {name}')
                    print(f'   Values: {np.array(obj)}')
            elif isinstance(obj, h5py.Group):
                print(f'📁 Group: {name}/')
        
        f.visititems(search_recursive)
        
        # Also check top-level attributes
        print('\\n🏷️  Top-level attributes:')
        for attr_name in f.attrs:
            print(f'   {attr_name}: {f.attrs[attr_name]}')

def check_dimensions():
    # Check actual dimensions
    with h5py.File('tests/resources/detection_data/cracker_box/np1_42_depth.h5', 'r') as f:
        print('Depth shape:', np.array(f['depth']).shape)
        print('Depth keys:', list(f.keys()))

    # Check mask
    try:
        mask = Image.open('tests/resources/detection_data/cracker_box/np1_42_mask.pbm')
        print('Mask size:', mask.size)  # (width, height)
        print('Mask mode:', mask.mode)
    except Exception as e:
        print('Mask error:', e)

    # Check RGB
    try:
        rgb = Image.open('tests/resources/detection_data/cracker_box/np1_42_color.jpg')
        print('RGB size:', rgb.size)  # (width, height)
    except Exception as e:
        print('RGB error:', e)


def check_pose():
    try:
        with h5py.File('tests/resources/detection_data/cracker_box/np1_42_pose.h5', 'r') as f:
            print('Pose file keys:', list(f.keys()))
            for key in f.keys():
                data = np.array(f[key])
                print(f'{key}: shape={data.shape}, dtype={data.dtype}')
                print(f'{key} data:', data)
                print()
    except Exception as e:
        print('Error:', e)

# Check the file
file_path = "tests/resources/detection_datasets/cracker_box/np1_42_depth.h5"
# check_depth_in_h5(file_path)
# save_depth_image(file_path)
# calibration_file_path = "tests/resources/detection_datasets/cracker_box/calibration.h5"
# find_camera_data(calibration_file_path)
# check_dimensions()
# check_pose()
# Let's check all the data in the pose file more carefully
# with h5py.File('tests/resources/detection_data/cracker_box/np1_42_pose.h5', 'r') as f:
#     print('All keys in pose file:', list(f.keys()))
    
#     # Check the transformation matrix more carefully
#     H_matrix = np.array(f['H_table_from_reference_camera'])
#     print('\\nH_table_from_reference_camera:')
#     print('Full 4x4 matrix:')
#     print(H_matrix)
#     print('\\nTranslation (last column, first 3 rows):', H_matrix[:3, 3])
#     print('Rotation matrix (top-left 3x3):', H_matrix[:3, :3])
    
#     # Check board_frame_offset
#     if 'board_frame_offset' in f:
#         offset = np.array(f['board_frame_offset'])
#         print('\\nboard_frame_offset:', offset)
    
#     # The name suggests this might be TABLE pose, not object pose
#     print('\\nMatrix name suggests this is TABLE pose relative to camera, not OBJECT pose')

# Table pose (from H_table_from_reference_camera)
# table_translation = np.array([0.02201708, 0.08063162, 0.97343685])
# table_rotation = np.array([[ 0.57488349, -0.81805406, -0.01722022],
#                           [-0.81822417, -0.57485804, -0.0068881 ],
#                           [-0.00426434,  0.01804986, -0.99982799]])

# # Object offset relative to table
# board_frame_offset = np.array([2.67954484e-01, 8.82416179e-02, 6.08015247e-08])

# # Transform object offset to camera frame
# object_offset_in_camera_frame = table_rotation @ board_frame_offset
# print('Object offset in camera frame:', object_offset_in_camera_frame)

# # Calculate object position in camera frame
# object_position = table_translation + object_offset_in_camera_frame
# print('Object position in camera frame:', object_position)
# print('Object position (formatted):', [float(x) for x in object_position])

# Check raw depth values
with h5py.File('tests/resources/detection_data/cracker_box/np1_42_depth.h5', 'r') as f:
    depth_raw = np.array(f['depth'])
    
print('Raw depth statistics:')
print(f'Min: {depth_raw.min()}, Max: {depth_raw.max()}')
print(f'Mean: {depth_raw.mean():.1f}, Median: {np.median(depth_raw):.1f}')
print(f'Non-zero values: {np.count_nonzero(depth_raw)}')

# Check if values look like millimeters (expect ~1000 for 1 meter)
non_zero_depths = depth_raw[depth_raw > 0]
print(f'Non-zero depth range: {non_zero_depths.min()} to {non_zero_depths.max()}')
print(f'If these are mm, object would be at {non_zero_depths.mean()/1000:.2f} meters average')