import pytest
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(params=[
    # ("rai_core", "rai"),
    # ("rai_whoami", "rai_whoami"),
    # ("rai_s2s", "rai_s2s"),
    # ("rai_sim", "rai_sim"),
    # ("rai_bench", "rai_bench"),
    ("rai_extensions/rai_perception", "rai_perception"),
])
def package_info(request):
    """Fixture to provide package paths and import names."""
    package_path, import_name = request.param
    package_root = Path(__file__).parent.parent.parent / "src" / package_path
    return package_root, import_name


def run_command(cmd, cwd=None):
    """Run shell command and return result."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def test_package_build_and_install(package_info):
    """Test that package builds and installs correctly."""
    package_root, import_name = package_info
    
    print(f"Testing package: {package_root.name} ({import_name})")
    # Clean previous builds
    for path in ["dist", "build", "*.egg-info"]:
        for item in package_root.glob(path):
            if item.is_dir():
                shutil.rmtree(item)
    
    # Build package
    returncode, stdout, stderr = run_command("python -m build", cwd=package_root)
    assert returncode == 0, f"Build failed for {package_root.name}: {stderr}"
    
    # Check dist files exist
    dist_dir = package_root / "dist"
    assert dist_dir.exists(), f"dist/ directory not found for {package_root.name}"
    
    whl_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    assert whl_files, f"No .whl file found for {package_root.name}"
    assert tar_files, f"No .tar.gz file found for {package_root.name}"
    
    # Test installation in virtual environment
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"
        
        # Create venv
        returncode, _, stderr = run_command(f"python -m venv {venv_path}")
        assert returncode == 0, f"venv creation failed: {stderr}"
        
        # Get pip path
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = venv_path / "Scripts" / "pip.exe"  # Windows
        
        # Install package
        whl_path = whl_files[0].absolute()
        returncode, _, stderr = run_command(f"{pip_path} install {whl_path}")
        assert returncode == 0, f"Installation failed for {import_name}: {stderr}"
        
        # Verify import
        python_path = venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = venv_path / "Scripts" / "python.exe"  # Windows
        
        returncode, stdout, stderr = run_command(
            f"{python_path} -c 'import {import_name}; print(\"{import_name} imported successfully\")'")
        
        assert returncode == 0, f"Import failed for {import_name}: {stderr}"
        assert f"{import_name} imported successfully" in stdout


# make sure test-pypi is added to ~/.pypirc
# update version before publishing
@pytest.mark.manual
def test_package_publish(package_info):
    """Test that package publishes to Test PyPI correctly."""
    package_root, import_name = package_info
    package_path = package_root.name
    
    print(f"Testing package: {package_root.name} ({import_name})")
    
    # Publish package
    dist_path = package_root / "dist"
    publish_cmd = f"python -m twine upload {dist_path}/{package_path}* -r test-pypi --verbose"
    print(f"Publishing package: {publish_cmd}")
    returncode, stdout, stderr = run_command(publish_cmd, cwd=package_root)
    assert returncode == 0, f"Publish failed for {import_name}: {stderr}"