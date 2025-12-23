# Testing README in Docker with ROS2 Jazzy

This guide provides step-by-step instructions to test the installation and usage instructions from the README in a clean Docker environment.

## Prerequisites

- Docker installed on your system
- Internet connection (for downloading packages and models)

## Step-by-Step Testing Guide

### Step 1: Start Docker Container with ROS2 Jazzy

```bash
docker run -it --rm --gpus all --name rai-perception-test osrf/ros:jazzy-desktop-full /bin/bash
```

This starts an interactive container with ROS2 Jazzy pre-installed and GPU support enabled.

> [!NOTE]
> If you don't have GPU support or get an error, you can remove `--gpus all` (the package will fall back to CPU).

### Step 2: Update Package Lists

```bash
apt-get update
```

### Step 3: Install Basic Tools

```bash
apt-get install -y python3-pip wget curl
```

> [!NOTE]
> `wget` is required for downloading model weights. The agents will download weights automatically on first use.

### Step 4: Source ROS2 Jazzy

```bash
source /opt/ros/jazzy/setup.bash
```

Verify ROS2 is working:
```bash
ros2 --help
```

### Step 5: Install ROS2 Dependencies

**Update package lists:**
```bash
apt-get update
```

**Check if vision_msgs is available (usually included with desktop-full):**
```bash
source /opt/ros/jazzy/setup.bash
ros2 pkg list | grep vision_msgs
```

If not found, install it:
```bash
apt-get install -y ros-jazzy-vision-msgs
```

**Install rai_interfaces as a debian package:**
```bash
apt-get install -y ros-jazzy-rai-interfaces
```

### Step 6: Verify rai_interfaces is accessible

```bash
source /opt/ros/jazzy/setup.bash
python3 -c "from rai_interfaces.srv import RAIGroundingDino, RAIGroundedSam; print('✓ rai_interfaces accessible')"
```

### Step 7: Install rai-perception via pip

```bash
apt install python3.12-venv
python3 -m venv test-venv
source test-venv/bin/activate

pip install rai-perception
```

If testing from test-PyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rai-perception

pip install --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rai-perception
```

Verify installation:
```bash
python3 -c "import rai_perception; print('rai-perception installed successfully')"
```

### Step 8: Test Quick Start - Launch Perception Agents

In the container, run:
```bash
source /opt/ros/jazzy/setup.bash
python3 -m rai_perception.scripts.run_perception_agents
```

Expected behavior:
- Agents should start without errors
- You should see logs indicating `grounding_dino` and `grounded_sam` nodes are running
- Services should be available: `grounding_dino_classify` and `grounded_sam_segment`

### Step 9: Test Service Availability (in a new terminal)

Open a new terminal and attach to the running container:
```bash
docker exec -it rai-perception-test /bin/bash
```

Then check services:
```bash
source /opt/ros/jazzy/setup.bash
ros2 service list | grep -E "(grounding_dino|grounded_sam)"
```

Expected output should include:
- `/grounding_dino_classify`
- `/grounded_sam_segment`

### Step 10: Test Example Client (Optional)

If you have a test image available, you can test the talker example:

```bash
# In the second terminal (with agents still running in first)
source /opt/ros/jazzy/setup.bash

mkdir -p /tmp/rai_perception_images
wget https://raw.githubusercontent.com/RobotecAI/rai/main/src/rai_extensions/rai_perception/images/sample.jpg -O /tmp/rai_perception_images/sample.jpg

python -m rai_perception.examples.talker --ros-args -p image_path:=/tmp/rai_perception_images/sample.jpg
```



Expected behavior:
- Client connects to services
- Detection and segmentation complete
- `masks.png` file created in current directory

### Step 11: Verify Weights Download

Check that weights were downloaded:
```bash
ls -lh ~/.cache/rai/
```

Expected: Model weight files should be present.

## Troubleshooting

### Issue: `ros-jazzy-rai-interfaces` not found

**Solution:** Ensure `apt-get update` has been run to refresh the package cache. The package should be available as a debian package. If it's still not found after updating, check that the ROS2 repositories are properly configured.

### Issue: Import errors for `rai_interfaces`

**Solution:** Ensure ROS2 is sourced (rai_interfaces should be available after apt install):
```bash
source /opt/ros/jazzy/setup.bash
python3 -c "from rai_interfaces.srv import RAIGroundingDino; print('OK')"
```

### Issue: CUDA/GPU not available

**Solution:** The package should fall back to CPU. If you need GPU support, ensure:
1. NVIDIA drivers are installed on the host
2. `nvidia-container-toolkit` is installed: `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker`
3. Use `--gpus all` flag when starting the container (see Step 1)

## Quick Test Script

You can also create a test script to automate the basic checks:

```bash
#!/bin/bash
set -e

echo "Testing rai-perception installation..."

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Check Python import
python3 -c "import rai_perception; print('✓ rai-perception import successful')"

# Check ROS2 interfaces
python3 -c "from rai_interfaces.srv import RAIGroundingDino, RAIGroundedSam; print('✓ rai_interfaces import successful')"

# Check vision_msgs
python3 -c "from vision_msgs.msg import BoundingBox2D, Detection2D; print('✓ vision_msgs import successful')"

# Check agents can be imported
python3 -c "from rai_perception.agents import GroundingDinoAgent, GroundedSamAgent; print('✓ Agents import successful')"

echo "All basic checks passed!"
```

Save as `test_installation.sh`, make executable, and run:
```bash
chmod +x test_installation.sh
./test_installation.sh
```

