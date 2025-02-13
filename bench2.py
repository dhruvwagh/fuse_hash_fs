#!/usr/bin/env python3
import subprocess
import time
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np

# Configuration
FUSE_BINARY = "/home/dhruv/Documents/fuse_hash_fs/build/my_fs"
MOUNTPOINT = "/home/dhruv/myfuse"
DRIVES_LIST = [1, 2, 4, 8, 12, 16]

# Test scenarios
SCENARIOS = [
    {"name": "many_small_files", "num_files": 1000, "file_size": 64*1024, "block_size": 4*1024},
    {"name": "few_large_files", "num_files": 10, "file_size": 16*1024*1024, "block_size": 4*1024},
    {"name": "medium_small_blocks", "num_files": 100, "file_size": 1024*1024, "block_size": 4*1024},
    {"name": "medium_large_blocks", "num_files": 100, "file_size": 1024*1024, "block_size": 64*1024},
]

def mount_fuse(drives):
    print(f"[Info] Mounting FUSE with drives={drives}")
    cmd = [FUSE_BINARY, '-o', f"drives={drives}", MOUNTPOINT]
    proc = subprocess.Popen(cmd)
    time.sleep(2)
    return proc

def unmount_fuse(proc):
    print("[Info] Unmounting FUSE...")
    subprocess.run(["fusermount", "-u", MOUNTPOINT], check=False)
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except:
        pass

def run_single_test(num_files, file_size, block_size):
    data_block = b"A" * block_size
    total_data = num_files * file_size
    
    # Write test
    write_start = time.time()
    for i in range(num_files):
        filename = os.path.join(MOUNTPOINT, f"test_file_{i}.dat")
        with open(filename, "wb") as f:
            written = 0
            while written < file_size:
                to_write = min(block_size, file_size - written)
                f.write(data_block[:to_write])
                written += to_write
    write_end = time.time()
    
    # Read test
    read_start = time.time()
    for i in range(num_files):
        filename = os.path.join(MOUNTPOINT, f"test_file_{i}.dat")
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
    read_end = time.time()
    
    # Cleanup
    for i in range(num_files):
        filename = os.path.join(MOUNTPOINT, f"test_file_{i}.dat")
        os.remove(filename)
    
    # Calculate throughput
    write_MBps = (total_data / (1024*1024)) / (write_end - write_start)
    read_MBps = (total_data / (1024*1024)) / (read_end - read_start)
    
    return write_MBps, read_MBps

def plot_scenario(scenario_name, results, output_dir):
    drives = [r[0] for r in results]
    write_speeds = [r[1] for r in results]
    read_speeds = [r[2] for r in results]
    
    # Calculate ideal scaling
    base_write = write_speeds[0]
    base_read = read_speeds[0]
    ideal_write = [base_write * d for d in drives]
    ideal_read = [base_read * d for d in drives]
    
    plt.figure(figsize=(12, 6))
    
    # Actual results
    plt.plot(drives, write_speeds, 'bo-', label='Actual Write MB/s')
    plt.plot(drives, read_speeds, 'ro-', label='Actual Read MB/s')
    
    # Ideal scaling
    plt.xlabel('Number of Drives')
    plt.ylabel('Throughput (MB/s)')
    plt.title(f'FUSE Performance Scaling - {scenario_name}')
    plt.grid(True)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{output_dir}/{scenario_name}_{timestamp}.png')
    plt.close()

def main():
    if not os.path.exists(MOUNTPOINT):
        os.makedirs(MOUNTPOINT)
        
    # Create output directory for results
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    for scenario in SCENARIOS:
        print(f"\n[Info] Running scenario: {scenario['name']}")
        print(f"Files: {scenario['num_files']}, Size: {scenario['file_size']/1024:.1f}KB, Block: {scenario['block_size']/1024:.1f}KB")
        
        results = []
        for drives in DRIVES_LIST:
            fuse_proc = mount_fuse(drives)
            try:
                wmb, rmb = run_single_test(
                    scenario['num_files'],
                    scenario['file_size'],
                    scenario['block_size']
                )
                print(f"[Result] drives={drives} | Write={wmb:.2f} MB/s, Read={rmb:.2f} MB/s")
                results.append((drives, wmb, rmb))
            finally:
                unmount_fuse(fuse_proc)
                time.sleep(1)
        
        # Plot this scenario
        plot_scenario(scenario['name'], results, output_dir)
        
        # Store results
        all_results[scenario['name']] = results
    
    # Save numerical results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'{output_dir}/results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()