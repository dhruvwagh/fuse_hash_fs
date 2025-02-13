#!/usr/bin/env python3
import subprocess
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Match original configuration exactly
class BenchmarkConfig:
    def __init__(self):
        self.FUSE_BINARY = "/home/dhruv/Documents/fuse_hash_fs/build/my_fs"
        self.MOUNTPOINT = "/home/dhruv/myfuse"
        self.DRIVES_LIST = [1, 2, 4, 8, 12, 16]
        
        # Match original parameters exactly
        self.TOTAL_BYTES = 32 * 1024 * 1024  # 32 MB total
        self.FILE_SIZE = 4 * 1024 * 1024     # 4 MB per file (original value)
        self.CHUNK_SIZE = 4 * 1024           # 4 KB chunks (original value)
        
        self.ITERATIONS = 5

def run_benchmark(config):
    """Run single benchmark iteration with original parameters"""
    num_files = config.TOTAL_BYTES // config.FILE_SIZE
    data_block = b"A" * config.CHUNK_SIZE
    
    # Write phase
    write_start = time.time()
    for i in range(num_files):
        filename = os.path.join(config.MOUNTPOINT, f"file{i}.dat")
        with open(filename, "wb") as f:
            written = 0
            while written < config.FILE_SIZE:
                f.write(data_block)
                written += config.CHUNK_SIZE
    write_end = time.time()
    
    # Sync to ensure writes are complete
    subprocess.run(["sync"])
    time.sleep(1)
    
    # Read phase
    read_start = time.time()
    for i in range(num_files):
        filename = os.path.join(config.MOUNTPOINT, f"file{i}.dat")
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(config.CHUNK_SIZE)
                if not chunk:
                    break
    read_end = time.time()
    
    # Cleanup
    for i in range(num_files):
        filename = os.path.join(config.MOUNTPOINT, f"file{i}.dat")
        try:
            os.remove(filename)
        except:
            pass
    
    # Calculate throughput
    total_MB = config.TOTAL_BYTES / (1024 * 1024)  # Convert to MB
    write_throughput = total_MB / (write_end - write_start)
    read_throughput = total_MB / (read_end - read_start)
    
    return write_throughput, read_throughput

def ensure_unmounted(mountpoint):
    """Ensure clean unmount"""
    try:
        subprocess.run(["fusermount", "-u", mountpoint], check=False, timeout=5)
    except:
        try:
            subprocess.run(["fusermount", "-uz", mountpoint], check=False, timeout=5)
        except:
            pass
    time.sleep(2)

def plot_results(results, config, timestamp):
    """Plot throughput scaling matching original format"""
    drives = config.DRIVES_LIST
    
    # Separate write and read results
    write_results = {d: [r[0] for r in runs] for d, runs in results.items()}
    read_results = {d: [r[1] for r in runs] for d, runs in results.items()}
    
    # Calculate means and standard deviations
    write_means = [np.mean(write_results[d]) for d in drives]
    read_means = [np.mean(read_results[d]) for d in drives]
    write_stds = [np.std(write_results[d]) for d in drives]
    read_stds = [np.std(read_results[d]) for d in drives]
    
    # Create plot matching original format
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Throughput plot
    ax1.set_title(f"FUSE Parallel Scaling: each file = {config.FILE_SIZE/1024:.1f} KB")
    ax1.errorbar(drives, write_means, yerr=write_stds, 
                marker='o', color='blue', label='Write MB/s')
    ax1.errorbar(drives, read_means, yerr=read_stds, 
                marker='^', color='red', label='Read MB/s')
    ax1.set_xlabel("Number of Drives")
    ax1.set_ylabel("Throughput (MB/s)")
    ax1.grid(True)
    ax1.legend()
    
    # Calculate IOPS
    write_iops = [m * 1024 * 1024 / config.FILE_SIZE for m in write_means]
    read_iops = [m * 1024 * 1024 / config.FILE_SIZE for m in read_means]
    
    # IOPS plot
    ax2.set_title("IOPS (Operations per Second)")
    ax2.plot(drives, write_iops, marker='o', color='green', label='Write IOPS')
    ax2.plot(drives, read_iops, marker='^', color='orange', label='Read IOPS')
    ax2.set_xlabel("Number of Drives")
    ax2.set_ylabel("IOPS")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"scaling_results_{timestamp}.png")
    plt.close()

def main():
    config = BenchmarkConfig()
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure clean start
    ensure_unmounted(config.MOUNTPOINT)
    if not os.path.exists(config.MOUNTPOINT):
        os.makedirs(config.MOUNTPOINT)
    
    for drives in config.DRIVES_LIST:
        results[drives] = []
        
        # Mount FUSE
        ensure_unmounted(config.MOUNTPOINT)
        time.sleep(2)
        
        fuse_proc = subprocess.Popen([
            config.FUSE_BINARY,
            '-o', f"drives={drives}",
            config.MOUNTPOINT
        ])
        time.sleep(3)  # Wait for mount
        
        try:
            # Run iterations
            for i in range(config.ITERATIONS):
                print(f"Running benchmark with {drives} drives, "
                      f"iteration {i+1}/{config.ITERATIONS}")
                
                result = run_benchmark(config)
                results[drives].append(result)
                time.sleep(1)
                
        finally:
            # Cleanup
            ensure_unmounted(config.MOUNTPOINT)
            try:
                fuse_proc.terminate()
                fuse_proc.wait(timeout=5)
            except:
                try:
                    fuse_proc.kill()
                except:
                    pass
            time.sleep(2)
    
    # Plot results
    plot_results(results, config, timestamp)

if __name__ == "__main__":
    main()