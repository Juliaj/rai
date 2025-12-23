# Debug script for CUDA unknown error 

import torch
import subprocess

def check_gpu_processes():
    """Get GPU processes using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                pid, name = line.split(', ')
                processes.append((int(pid), name))
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Failed to get GPU processes: {e}")
        return []

def get_gpu_info():
    """Get GPU info using torch and nvidia-smi"""
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_device)
        
        # Get actual memory usage from nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        memory_line = result.stdout.strip().split('\n')[current_device]
        total_mb, used_mb, free_mb = map(int, memory_line.split(', '))
        
        return {
            'name': props.name,
            'device_count': device_count,
            'current_device': current_device,
            'memory_total': total_mb / 1024,
            'memory_used': used_mb / 1024,
            'memory_free': free_mb / 1024,
            'pytorch_allocated': torch.cuda.memory_allocated() / 1024**3,
            'pytorch_reserved': torch.cuda.memory_reserved() / 1024**3
        }
    except Exception as e:
        return {'error': f'Failed to get GPU info: {e}'}

def test_cuda_context():
    """Test CUDA context health"""
    try:
        torch.cuda.init()
        device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        # Test basic operation
        x = torch.ones(1).cuda()
        print(f"CUDA context healthy on device {device}")
        return True
    except Exception as e:
        print(f"CUDA context corrupted: {e}")
        return False

def check_nvidia_driver_health():
    """Check for NVIDIA driver corruption"""
    checks = {}
    
    # 1. Check nvidia-smi basic functionality
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True, timeout=10)
        checks['nvidia_smi'] = 'OK'
    except Exception as e:
        checks['nvidia_smi'] = f'FAILED: {e}'
    
    # 2. Check GPU temperature (driver communication)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        temp = result.stdout.strip()
        checks['gpu_temperature'] = f'{temp}°C'
    except Exception as e:
        checks['gpu_temperature'] = f'FAILED: {e}'
    
    # 3. Check persistent mode
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=persistence_mode', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        persistence = result.stdout.strip()
        checks['persistence_mode'] = persistence
    except Exception as e:
        checks['persistence_mode'] = f'FAILED: {e}'
    
    # 4. Check for Xid errors in dmesg (requires sudo)
    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
        xid_errors = [line for line in result.stdout.split('\n') if 'NVRM: Xid' in line]
        checks['xid_errors'] = len(xid_errors)
        if xid_errors:
            checks['latest_xid'] = xid_errors[-1]
    except Exception as e:
        checks['xid_errors'] = f'Cannot check: {e}'
    
    return checks

if __name__ == "__main__":
    print("-----------GPU Info--------------------------------")
    print(get_gpu_info())
    print("-----------GPU Processes---------------------------")
    print(check_gpu_processes())
    print("-----------CUDA Context Test-----------------------")
    print(f"CUDA context healthy: {test_cuda_context()}")
    print("-----------NVIDIA Driver Health---------------------")
    health = check_nvidia_driver_health()
    for key, value in health.items():
        print(f"{key}: {value}")