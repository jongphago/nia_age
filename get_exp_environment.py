import os
import shutil
import sys

import GPUtil
import cpuinfo
import psutil


def print_to_file(file_path, func):
    original_stdout = sys.stdout
    with open(file_path, 'w') as fid:
        sys.stdout = fid
        func()
        sys.stdout = original_stdout


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def save_cpu_info():
    info = cpuinfo.get_cpu_info()
    for key in info.keys():
        print('{} : {}'.format(key, info[key]))


def save_memory_info():
    # Memory Information
    print("=" * 40, "Memory Information", "=" * 40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("=" * 20, "SWAP", "=" * 20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")


def save_disk_info():
    # Disk Information
    print("=" * 40, "Disk Information", "=" * 40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")


def save_gpu_info():
    GPUtil.showUtilization(all=True)


def save_os_info():
    print(os.uname())


def save_env(output_root):
    output_dir = output_root + '1_env_logs/'
    os.makedirs(output_dir, exist_ok=True)
    print_to_file(output_dir + '1)cpu.txt', save_cpu_info)
    print_to_file(output_dir + '2)gpu.txt', save_gpu_info)
    print_to_file(output_dir + '3)ram.txt', save_memory_info)
    print_to_file(output_dir + '4)hdd.txt', save_disk_info)
    print_to_file(output_dir + '5)os.txt', save_os_info)
    shutil.copyfile('requirements.txt', output_dir + '6)framework.txt')


if __name__ == '__main__':
    save_env('test_logs/')
