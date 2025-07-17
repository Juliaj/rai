#!/usr/bin/env python3
"""
Script to test saved debug images with GDINO service to verify if cubes are detectable.
This helps determine if the issue is with the service or the image content.
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

def find_debug_images():
    """Find all saved debug images"""
    debug_dir = "debug_images"
    if not os.path.exists(debug_dir):
        print(f"❌ Debug directory '{debug_dir}' not found!")
        return []
    
    # Find all saved images
    color_images = glob.glob(f"{debug_dir}/gdino_timeout_cube_*.jpg")
    depth_images = glob.glob(f"{debug_dir}/gdino_timeout_depth_cube_*.png")
    
    print(f"📁 Found {len(color_images)} color images and {len(depth_images)} depth images")
    return color_images, depth_images

def test_image_with_gdino(image_path):
    """Test a single image with GDINO service using talker.py"""
    print(f"\n🔍 Testing image: {image_path}")
    
    # Get the talker.py path
    talker_path = "src/rai_extensions/rai_open_set_vision/rai_open_set_vision/examples/talker.py"
    
    if not os.path.exists(talker_path):
        print(f"❌ Talker script not found at {talker_path}")
        return False
    
    try:
        # Run the talker script with our debug image
        cmd = [
            "python3", talker_path,
            "--ros-args", 
            "-p", f"image_path:={os.path.abspath(image_path)}"
        ]
        
        print(f"🚀 Running: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        print(f"📤 STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"📤 STDERR:\n{result.stderr}")
        
        # Check if detection was successful
        if "detections" in result.stdout and "detection" in result.stdout.lower():
            print("✅ GDINO detected objects in the image!")
            return True
        else:
            print("❌ GDINO found no objects in the image")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def test_with_different_classes(image_path):
    """Test the same image with different object classes"""
    print(f"\n🎯 Testing different object classes for: {image_path}")
    
    # Common object classes to test
    test_classes = [
        "cube",
        "box", 
        "object",
        "block",
        "red cube",
        "blue cube", 
        "green cube",
        "yellow cube",
        "wooden cube",
        "plastic cube",
        "toy",
        "item"
    ]
    
    talker_path = "src/rai_extensions/rai_open_set_vision/rai_open_set_vision/examples/talker.py"
    
    for class_name in test_classes:
        print(f"\n🔍 Testing class: '{class_name}'")
        
        try:
            # Create a temporary modified talker script
            with open(talker_path, 'r') as f:
                talker_content = f.read()
            
            # Modify the classes in the script
            modified_content = talker_content.replace(
                'self.req.classes = "dragon , lizard , dinosaur"',
                f'self.req.classes = "{class_name}"'
            )
            
            # Write to temporary file
            temp_talker = f"temp_talker_{class_name.replace(' ', '_')}.py"
            with open(temp_talker, 'w') as f:
                f.write(modified_content)
            
            # Run the modified script
            cmd = [
                "python3", temp_talker,
                "--ros-args", 
                "-p", f"image_path:={os.path.abspath(image_path)}"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check for detections
            if "detections" in result.stdout and "detection" in result.stdout.lower():
                print(f"✅ Found '{class_name}' in image!")
                # Clean up temp file
                os.remove(temp_talker)
                return class_name
            else:
                print(f"❌ No '{class_name}' found")
            
            # Clean up temp file
            os.remove(temp_talker)
            
        except Exception as e:
            print(f"❌ Error testing '{class_name}': {e}")
            if os.path.exists(temp_talker):
                os.remove(temp_talker)
    
    return None

def main():
    print("🔍 Debug Image Testing Script")
    print("=" * 50)
    
    # Find debug images
    color_images, depth_images = find_debug_images()
    
    if not color_images:
        print("❌ No debug images found. Run the manipulation demo first to generate debug images.")
        return
    
    print(f"\n📸 Testing {len(color_images)} color images...")
    
    successful_detections = 0
    working_classes = []
    
    for i, image_path in enumerate(color_images, 1):
        print(f"\n{'='*60}")
        print(f"Image {i}/{len(color_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Test with default classes first
        if test_image_with_gdino(image_path):
            successful_detections += 1
        else:
            # If default test fails, try different object classes
            working_class = test_with_different_classes(image_path)
            if working_class:
                working_classes.append(working_class)
                successful_detections += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images tested: {len(color_images)}")
    print(f"Successful detections: {successful_detections}")
    print(f"Detection rate: {successful_detections/len(color_images)*100:.1f}%")
    
    if working_classes:
        print(f"Working object classes: {list(set(working_classes))}")
    
    if successful_detections == 0:
        print("\n❌ No objects detected in any debug images!")
        print("This suggests either:")
        print("1. The images don't contain visible cubes")
        print("2. The GDINO service is not working properly")
        print("3. The image quality is too poor for detection")
        print("4. The cubes are not in the expected format/color")
    elif successful_detections < len(color_images):
        print(f"\n⚠️ Only {successful_detections}/{len(color_images)} images had detectable objects")
        print("This suggests inconsistent image quality or object visibility")
    else:
        print("\n✅ All images had detectable objects!")
        print("The issue might be with the service timing out during the demo")

if __name__ == "__main__":
    main() 