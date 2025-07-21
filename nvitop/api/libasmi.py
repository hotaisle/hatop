import atexit as _atexit
import amdsmi as _asmi
from amdsmi import AmdSmiInitFlags
from amdsmi import AmdSmiException
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import TypeAlias as _TypeAlias
import ctypes as _ctypes
import logging
import threading as _threading

ASMIError: type[_asmi.AmdSmiException] = _asmi.AmdSmiException
ASMIDeviceHandle: _TypeAlias = _ctypes.c_void_p
"""
cp -r /opt/rocm/share/amd_smi ~/amd_smi
python3 -m pip install --user ~/amd_smi

"""

# Global state for initialization
_initialized = False
_init_lock = _threading.Lock()
_device_handles: list[ASMIDeviceHandle] | None = None

def _lazy_init() -> None:
    """Lazily initialize the AMD SMI context.

    Raises:
        AmdSmiException: If initialization fails
    """
    global _initialized, _device_handles
    
    with _init_lock:
        if _initialized:
            return
        
        try:
            _asmi.amdsmi_init()
            _device_handles = _asmi.amdsmi_get_processor_handles()
            _initialized = True
            _atexit.register(_shutdown)
        except AmdSmiException as e:
            _initialized = False
            _device_handles = None
            raise

def _shutdown() -> None:
    """Shutdown AMD SMI if initialized."""
    global _initialized, _device_handles
    
    with _init_lock:
        if _initialized:
            try:
                _asmi.amdsmi_shut_down()
            except AmdSmiException:
                pass  # Ignore errors during shutdown
            finally:
                _initialized = False
                _device_handles = None


def asmi_available() -> bool:
    try:
        return device_count() > 0
    except:
        return False


def device_count() -> int:
    try:
        _lazy_init()
        return len(_device_handles) if _device_handles else 0
    except AmdSmiException:
        return 0


def get_uuid(index: int) -> str | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        uuid = _asmi.amdsmi_get_gpu_device_uuid(handle)
        return uuid
    except (IndexError, AmdSmiException):
        # UUID may not be available
        return None
    

def get_rocm_version() -> str:
    """Get the installed ROCm version."""
    try:
        _lazy_init()
        # Use the built-in amdsmi function to get ROCm version
        result = _asmi.amdsmi_get_rocm_version()
        # The function returns a tuple (success, version_string)
        if isinstance(result, tuple) and len(result) >= 2 and result[0]:
            return result[1]
        else:
            # Fallback in case the format is different
            return str(result)
    except (AmdSmiException, AttributeError, KeyError):
        return "Unknown"


def get_driver_version() -> str:
    try:
        _lazy_init()
        if not _device_handles or len(_device_handles) == 0:
            return "No AMD GPU Found"
        handle = _device_handles[0]
        driver_info = _asmi.amdsmi_get_gpu_driver_info(handle)
        
        # The AMD driver returns the full Linux kernel version string
        # Extract just the kernel version number
        driver_version = driver_info['driver_version']
        
        # Look for pattern like "6.8.0-64-generic" in the string
        import re
        match = re.search(r'(\d+\.\d+\.\d+-\d+-\w+)', driver_version)
        if match:
            return f"Linux {match.group(1)}"
        
        # If no match, try to extract just the version number
        match = re.search(r'(\d+\.\d+\.\d+)', driver_version)
        if match:
            return f"Linux {match.group(1)}"
            
        # Fallback: return a shortened version
        if len(driver_version) > 30:
            return "Linux " + driver_version[:24] + "..."
        
        return driver_version
    except (IndexError, AmdSmiException):
        # Driver version query failed
        return "ERROR"


def get_device_name(index: int) -> str | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        asic_info = _asmi.amdsmi_get_gpu_asic_info(handle)
        # print(asic_info['market_name'])
        # print(hex(asic_info['vendor_id']))
        # print(hex(asic_info['device_id']))
        # print(hex(asic_info['rev_id']))
        # print(asic_info['asic_serial'])
        return asic_info['market_name']
    except (IndexError, AmdSmiException):
        # Device name may not be available
        return None


def get_bdf(index: int) -> str | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        bdf: str = _asmi.amdsmi_get_gpu_device_bdf(handle)
        return bdf
    except (IndexError, AmdSmiException):
        # BDF may not be available
        return None


def get_memory_info(index: int) -> tuple[int, int] | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        used = _asmi.amdsmi_get_gpu_memory_usage(handle, _asmi.AmdSmiMemoryType.VRAM)
        total = _asmi.amdsmi_get_gpu_memory_total(handle, _asmi.AmdSmiMemoryType.VRAM)
        return used, total
    except (IndexError, AmdSmiException):
        # Memory info may not be available
        return None


def get_utilization_rates(index: int) -> (int, int):  # type: ignore
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return -1, -1
        handle = _device_handles[index]
        engine_usage = _asmi.amdsmi_get_gpu_activity(handle)
        return int(engine_usage['gfx_activity']), int(engine_usage['umc_activity'])
    except (IndexError, AmdSmiException):
        # Utilization rates may not be available
        return -1, -1


def get_fan_speed(index: int) -> int | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        cur_speed = _asmi.amdsmi_get_gpu_fan_speed(handle, 0)
        max_speed = _asmi.amdsmi_get_gpu_fan_speed_max(handle, 0)
        # Return percentage (0-100)
        if max_speed > 0:
            return int((cur_speed / max_speed) * 100)
        return cur_speed
    except (IndexError, AmdSmiException) as e:
        # Some AMD GPUs (like MI300X) don't have fan speed sensors
        # This is expected behavior, not an error
        return None


def get_temperature(index: int) -> int | None:
    """
    return temp in degrees Celsius
    """
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        temp_metric = _asmi.amdsmi_get_temp_metric(handle, _asmi.AmdSmiTemperatureType.EDGE,
                                                   _asmi.AmdSmiTemperatureMetric.CURRENT)
        return temp_metric
    except (IndexError, AmdSmiException) as e:
        # Some AMD GPUs may not support temperature reading
        # This is expected behavior, not an error
        return None

def get_power_usage(index: int) -> int | None:
    """
    Returns power usage in milliwatts
    """
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        power_info = _asmi.amdsmi_get_power_info(handle)
        
        # First try average_socket_power
        avg_power = power_info.get('average_socket_power')
        if isinstance(avg_power, (int, float)) and avg_power != 'N/A':
            return int(avg_power * 1000)  # Convert watts to milliwatts
        
        # Fall back to current_socket_power if available
        current_power = power_info.get('current_socket_power')
        if isinstance(current_power, (int, float)) and current_power != 'N/A':
            return int(current_power * 1000)  # Convert watts to milliwatts
        
        # Return None if no valid power data
        return None
    except (IndexError, AmdSmiException) as e:
        # Power usage may not be available on all AMD GPUs
        return None

def get_power_cap(index: int) -> int | None:
    """
    Return Power Capability in milliwatts
    """
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        power_info = _asmi.amdsmi_get_power_cap_info(handle)
        
        # power_cap is already in milliwatts according to AMD SMI docs
        power_cap = power_info.get('power_cap')
        
        # Validate it's a reasonable value
        if isinstance(power_cap, (int, float)) and power_cap > 0:
            # AMD MI300X returns power_cap in microwatts (750000000 = 750W)
            # Check if the value is unreasonably high (> 10MW in milliwatts)
            if power_cap > 10000000:  # > 10MW in milliwatts
                # Convert from microwatts to milliwatts
                return int(power_cap / 1000)
            else:
                # Already in milliwatts
                return int(power_cap)
        
        return None
    except (IndexError, AmdSmiException) as e:
        # Power cap may not be available on all AMD GPUs
        return None

def get_perf_level(index: int) -> str | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        perf_level = _asmi.amdsmi_get_gpu_perf_level(handle)
        return str(perf_level).replace("AMDSMI_DEV_PERF_LEVEL_", "").upper()[:3]
    except (IndexError, AmdSmiException):
        # Performance level may not be available
        return None

def get_uncorrectable_ecc(index: int) -> int | None:
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        ecc_count = _asmi.amdsmi_get_gpu_total_ecc_count(handle)
        return ecc_count['uncorrectable_count']
    except (IndexError, AmdSmiException):
        # ECC count may not be available
        return None


def get_processes(index: int) -> list[tuple[int, int]] | None:
    """
    Get list of processes using the GPU.
    
    Returns:
        List of tuples (pid, gpu_memory_bytes) or None if unavailable
    """
    try:
        _lazy_init()
        if not _device_handles or index >= len(_device_handles):
            return None
        handle = _device_handles[index]
        
        processes = []
        
        # Get process list - returns a list of dictionaries
        process_list = _asmi.amdsmi_get_gpu_process_list(handle)
        
        for proc_info in process_list:
            try:
                # Extract PID
                pid = proc_info.get('pid', 0)
                
                # Extract VRAM memory usage
                gpu_memory = 0
                memory_usage = proc_info.get('memory_usage', {})
                if isinstance(memory_usage, dict):
                    # VRAM memory is what we want for GPU memory
                    gpu_memory = memory_usage.get('vram_mem', 0)
                
                # Alternative: check if mem field exists (total memory)
                if gpu_memory == 0:
                    gpu_memory = proc_info.get('mem', 0)
                
                if pid > 0:
                    processes.append((pid, gpu_memory))
            except Exception as e:
                # Process may have exited or we may not have permissions
                logging.debug(f"Failed to get info for process: {e}")
                continue
                
        return processes
    except (IndexError, AmdSmiException) as e:
        # Process info may not be available
        logging.debug(f"Failed to get processes for GPU {index}: {e}")
        return []