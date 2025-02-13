#!/usr/bin/env python3
"""
Benchmark Script for FUSE Filesystem (fuse_fs.py) Across Multiple Real-World Workloads

This script mounts the FUSE filesystem for each drive configuration specified
by --drives. For each configuration, it:
  1. Creates a temporary mount point.
  2. Launches the FUSE filesystem (fuse_fs.py) with parameters:
       --num-drives, --drive-delay, and --block-size.
  3. Verifies the mount by checking that "testfile" exists.
  4. Runs a series of workloads:
       - Sequential Write: repeatedly writes blocks to "/testfile" sequentially.
       - Random Write: writes blocks to random offsets in "/testfile".
       - Sequential Read: reads sequentially from "/testfile".
       - Random Read: reads from random offsets in "/testfile".
       - Metadata Test: rapidly creates and deletes temporary files.
  5. Unmounts the filesystem.
  6. Records throughput (or ops per second) for each workload.
  7. Plots separate graphs for each workload type versus number of drives.

Make sure:
  - fuse_fs.py is in the same directory.
  - /etc/fuse.conf has "user_allow_other" uncommented.
  - Run with sudo (or appropriate privileges).

Usage:
  sudo python3 benchmark_fuse.py --drives 1,2,4,8,16 --duration 10 --rate 1000000 --block-size 4096 --drive-delay 0.01
"""

import os
import time
import random
import argparse
import subprocess
import tempfile
import shutil
import logging
import sys
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def verify_mount(mountpoint, timeout=30):
    """
    Verify the mount by waiting until the mountpoint is accessible and
    checking that "testfile" exists.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            contents = os.listdir(mountpoint)
            logging.info(f"Mountpoint {mountpoint} contents: {contents}")
            if "testfile" in contents:
                logging.info("Mount verification succeeded: 'testfile' found.")
                return True
        except Exception as e:
            logging.debug(f"Waiting for mountpoint {mountpoint}: {e}")
        time.sleep(1)
    return False

def mount_fuse(num_drives, mountpoint, drive_delay, block_size):
    """
    Launch the FUSE filesystem using fuse_fs.py with the desired parameters.
    """
    cmd = [
        "sudo", "python3", "fuse_fs.py", mountpoint,
        "--num-drives", str(num_drives),
        "--drive-delay", str(drive_delay),
        "--block-size", str(block_size)
    ]
    logging.info("Mounting FUSE with command: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc

def unmount_fuse(mountpoint, proc):
    """
    Unmount the FUSE filesystem by terminating the process and using fusermount.
    If fusermount -u fails, attempt a lazy unmount.
    """
    logging.info(f"Unmounting filesystem at {mountpoint}...")
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception as e:
        logging.error(f"Error terminating FUSE process: {e}")
    try:
        subprocess.run(["fusermount", "-u", mountpoint], check=True)
        logging.info("Successfully unmounted using fusermount -u.")
    except Exception as e:
        logging.error(f"Error with fusermount -u: {e}")
        try:
            subprocess.run(["fusermount", "-uz", mountpoint], check=True)
            logging.info("Successfully unmounted using fusermount -uz.")
        except Exception as e:
            logging.error(f"Lazy unmount failed: {e}")

# Workload Functions

def sequential_write(mountpoint, duration, rate, block_size):
    """Sequentially write to /testfile."""
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start = time.time()
    try:
        with open(filename, "ab") as f:
            while time.time() - start < duration:
                f.write(b'A' * block_size)
                f.flush()
                os.fsync(f.fileno())
                total_bytes += block_size
                time.sleep(1.0 / rate)
    except Exception as e:
        logging.error(f"Sequential write error: {e}")
    elapsed = time.time() - start
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    return total_bytes, elapsed, throughput

def random_write(mountpoint, duration, rate, block_size):
    """Randomly write to /testfile."""
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start = time.time()
    try:
        with open(filename, "r+b") as f:
            f.seek(0, os.SEEK_END)
            fsize = f.tell()
            if fsize < block_size:
                fsize = block_size
            while time.time() - start < duration:
                offset = random.randint(0, fsize - block_size)
                f.seek(offset)
                f.write(b'R' * block_size)
                f.flush()
                os.fsync(f.fileno())
                total_bytes += block_size
                time.sleep(1.0 / rate)
    except Exception as e:
        logging.error(f"Random write error: {e}")
    elapsed = time.time() - start
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    return total_bytes, elapsed, throughput

def sequential_read(mountpoint, duration, rate, block_size):
    """Sequentially read from /testfile."""
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start = time.time()
    try:
        with open(filename, "rb") as f:
            while time.time() - start < duration:
                data = f.read(block_size)
                if not data:
                    f.seek(0)
                    continue
                total_bytes += len(data)
                time.sleep(1.0 / rate)
    except Exception as e:
        logging.error(f"Sequential read error: {e}")
    elapsed = time.time() - start
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    return total_bytes, elapsed, throughput

def random_read(mountpoint, duration, rate, block_size):
    """Randomly read from /testfile."""
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start = time.time()
    try:
        with open(filename, "rb") as f:
            f.seek(0, os.SEEK_END)
            fsize = f.tell()
            if fsize < block_size:
                fsize = block_size
            while time.time() - start < duration:
                offset = random.randint(0, fsize - block_size)
                f.seek(offset)
                data = f.read(block_size)
                total_bytes += len(data)
                time.sleep(1.0 / rate)
    except Exception as e:
        logging.error(f"Random read error: {e}")
    elapsed = time.time() - start
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    return total_bytes, elapsed, throughput

def metadata_test(mountpoint, duration):
    """
    Perform metadata operations by rapidly creating and deleting temporary files
    in a dedicated subdirectory.
    """
    meta_dir = os.path.join(mountpoint, "meta_test")
    os.makedirs(meta_dir, exist_ok=True)
    start = time.time()
    op_count = 0
    try:
        while time.time() - start < duration:
            fname = os.path.join(meta_dir, f"temp_{int(time.time() * 1000)}.txt")
            with open(fname, "w") as f:
                f.write("metadata")
            os.remove(fname)
            op_count += 1
            time.sleep(0.001)
    except Exception as e:
        logging.error(f"Metadata test error: {e}")
    elapsed = time.time() - start
    ops_per_sec = op_count / elapsed if elapsed > 0 else 0
    return op_count, elapsed, ops_per_sec

def run_all_workloads(mountpoint, duration, rate, block_size):
    """
    Run all workloads on the mounted filesystem and collect results.
    Returns a dictionary mapping workload names to their results.
    """
    results = {}
    
    print("Running Sequential Write test...")
    total_bytes, elapsed, thr = sequential_write(mountpoint, duration, rate, block_size)
    results["Sequential Write"] = {"bytes": total_bytes, "time": elapsed, "throughput": thr}
    
    print("Running Random Write test...")
    total_bytes, elapsed, thr = random_write(mountpoint, duration, rate, block_size)
    results["Random Write"] = {"bytes": total_bytes, "time": elapsed, "throughput": thr}
    
    print("Running Sequential Read test...")
    total_bytes, elapsed, thr = sequential_read(mountpoint, duration, rate, block_size)
    results["Sequential Read"] = {"bytes": total_bytes, "time": elapsed, "throughput": thr}
    
    print("Running Random Read test...")
    total_bytes, elapsed, thr = random_read(mountpoint, duration, rate, block_size)
    results["Random Read"] = {"bytes": total_bytes, "time": elapsed, "throughput": thr}
    
    print("Running Metadata test...")
    op_count, elapsed, ops_sec = metadata_test(mountpoint, duration)
    results["Metadata"] = {"ops": op_count, "time": elapsed, "throughput": ops_sec}
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark FUSE Filesystem (fuse_fs.py) Across Multiple Workloads")
    parser.add_argument("--drives", type=str, default="1,2,4,8,16",
                        help="Comma-separated list of drive counts to test (e.g., 1,2,4,8,16)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration (seconds) for each workload test")
    parser.add_argument("--rate", type=int, default=1000000,
                        help="I/O operations per second for read/write tests")
    parser.add_argument("--block-size", type=int, default=4096,
                        help="Block size in bytes")
    parser.add_argument("--drive-delay", type=float, default=0.01,
                        help="Simulated drive processing delay per block (seconds)")
    args = parser.parse_args()

    drive_list = [int(x.strip()) for x in args.drives.split(",")]
    overall_results = {}

    base_temp_dir = tempfile.mkdtemp(prefix="fuse_mount_")
    logging.info("Temporary mount base: " + base_temp_dir)

    try:
        for nd in drive_list:
            mount_dir = os.path.join(base_temp_dir, f"mount_{nd}")
            os.makedirs(mount_dir, exist_ok=True)
            print(f"\n=== Testing with {nd} drives ===")
            fuse_proc = mount_fuse(nd, mount_dir, args.drive_delay, args.block_size)
            if not verify_mount(mount_dir, timeout=30):
                logging.error(f"Mount verification failed for {mount_dir}. Skipping configuration {nd}.")
                unmount_fuse(mount_dir, fuse_proc)
                continue
            
            # Run all workloads for this configuration.
            results = run_all_workloads(mount_dir, args.duration, args.rate, args.block_size)
            overall_results[nd] = results
            
            unmount_fuse(mount_dir, fuse_proc)
            time.sleep(5)  # Pause between configurations
    finally:
        shutil.rmtree(base_temp_dir)

    # Plot results for each workload.
    workload_names = list(next(iter(overall_results.values())).keys())
    for workload in workload_names:
        drive_nums = sorted(overall_results.keys())
        throughputs = []
        for d in drive_nums:
            if workload == "Metadata":
                # For metadata, we use ops per second.
                throughputs.append(overall_results[d][workload]["throughput"])
            else:
                throughputs.append(overall_results[d][workload]["throughput"])
        plt.figure(figsize=(8,6))
        plt.plot(drive_nums, throughputs, marker='o', linestyle='-')
        plt.xlabel("Number of Drives")
        if workload == "Metadata":
            plt.ylabel("Metadata Ops per Second")
            plt.title(f"Metadata Performance vs. Number of Drives")
        else:
            plt.ylabel("Throughput (MB/s)")
            plt.title(f"{workload} Throughput vs. Number of Drives")
        plt.grid(True)
        filename = f"fuse_benchmark_{workload.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graph for {workload} saved as {filename}")
        plt.show()

if __name__ == '__main__':
    main()
