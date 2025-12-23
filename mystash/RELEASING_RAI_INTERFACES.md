# Why `rai_interfaces` isn't available as a Debian package

`rai_interfaces` is listed on the [ROS2 index](https://index.ros.org/p/rai_interfaces/#jazzy) but **hasn't been officially released** into the ROS 2 Jazzy distribution's package repositories. This means:

- ✅ It's recognized as a ROS2 package
- ✅ It can be built from source
- ❌ It's **not** available via `apt install ros-jazzy-rai-interfaces`

## Why build from source is needed

ROS2 packages need to go through an official **release process** to become debian packages available via `apt`. The package must be:
1. Released into the ROS 2 build farm
2. Built and tested by the ROS infrastructure
3. Added to the official ROS 2 distribution repositories

## How to make it a Debian package

### Option 1: Build a local Debian package (for personal use)

This creates a `.deb` file you can install locally:

```bash
# Install build tools
sudo apt install python3-bloom python3-rosdep fakeroot debhelper dh-python

# Navigate to rai_interfaces
cd ~/rai_interfaces_ws/src/rai_interfaces

# Generate debian package files
bloom-generate rosdebian --os-name ubuntu --os-version jammy --ros-distro jazzy

# Build the debian package
fakeroot debian/rules binary

# Install the resulting .deb file
sudo dpkg -i ../ros-jazzy-rai-interfaces_*.deb
```

### Option 2: Release to ROS 2 build farm (for public distribution)

To make it available via `apt install ros-jazzy-rai-interfaces` for everyone:

1. **Create a release repository:**
   - Fork or create a `rai_interfaces-release` repository
   - This will contain the debian packaging files

2. **Use Bloom to release:**
   ```bash
   bloom-release --rosdistro jazzy --track jazzy rai_interfaces
   ```

3. **Submit to ROS 2 distribution:**
   - Create a pull request to the [rosdistro](https://github.com/ros/rosdistro) repository
   - Add your package to the appropriate distribution file
   - Wait for ROS build farm to build and test

4. **After approval:**
   - The package will be built by ROS build farm
   - It will become available via `apt install ros-jazzy-rai-interfaces`

## Current workaround

Until `rai_interfaces` is officially released, users must:
1. Clone the repository
2. Build it from source
3. Source the workspace

This is why the README instructions include building from source.

## References

- [ROS 2 Releasing a Package Tutorial](https://docs.ros.org/en/foxy/Tutorials/Releasing-a-ROS-2-Package.html)
- [Building a Custom Debian Package](https://docs.ros.org/en/foxy/How-To-Guides/Building-a-Custom-Debian-Package.html)
- [Bloom Documentation](https://bloom.readthedocs.io/)

