#!/usr/bin/env python3
"""
Automated Benchmark for fuse_fs.py with No User Inputs.

Hardcoded configurations:
  - Drive counts: [1, 2, 4, 8, 16]
  - Duration: 10 seconds
  - Write rate: 1000 ops/sec
  - Block size: 4096 bytes
  - Drive delay: 0.01s

Procedure for each drive count:
  1. Create a temporary mount directory.
  2. Launch fuse_fs.py mounting the FS at that directory.
  3. Verify "testfile" is in the mount.
  4. Perform a sequential write workload (appending data) for 10 seconds at 1000 ops/sec.
  5. Compute throughput.
  6. Unmount the filesystem.
  7. Store the result.

Finally, a plot of throughput vs. number of drives is created.
Requires:
  - fuse_fs.py in the same directory
  - /etc/fuse.conf must have "user_allow_other" uncommented
  - Run with sudo or appropriate privileges.
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Hardcoded test parameters
DRIVE_LIST = [1, 2, 4, 8, 16]
DURATION = 10           # seconds
RATE = 1000             # ops/sec
BLOCK_SIZE = 4096       # bytes
DRIVE_DELAY = 0.01      # seconds per block

def verify_mount(mountpoint, timeout=30):
    """
    Verify that 'testfile' appears in the root of mountpoint within 'timeout' seconds.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            contents = os.listdir(mountpoint)
            logging.debug(f"Contents of {mountpoint}: {contents}")
            if "testfile" in contents:
                logging.info("Mount verification succeeded: 'testfile' found.")
                return True
        except Exception as e:
            logging.debug(f"Error listing {mountpoint}: {e}")
        time.sleep(1)
    logging.error("Mount verification timed out; 'testfile' not found.")
    return False

def mount_fuse(num_drives, mountpoint):
    """
    Launch fuse_fs.py with our hardcoded parameters for a given drive count.
    Returns the subprocess.Popen object.
    """
    cmd = [
        "sudo", "python3", "fuse_fs.py", mountpoint,
        "--num-drives", str(num_drives),
        "--drive-delay", str(DRIVE_DELAY),
        "--block-size", str(BLOCK_SIZE)
    ]
    logging.info(f"Mounting fuse_fs.py for {num_drives} drives with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Give it a few seconds to initialize
    time.sleep(3)
    return proc

def unmount_fuse(mountpoint, proc):
    """
    Unmount the filesystem and terminate the FUSE process.
    """
    logging.info(f"Unmounting filesystem at {mountpoint}...")
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception as e:
        logging.error(f"Error terminating FUSE process: {e}")
    try:
        subprocess.run(["fusermount", "-u", mountpoint], check=True)
        logging.info("Unmounted with fusermount -u.")
    except Exception as e:
        logging.error(f"Error unmounting normally: {e}")
        logging.info("Attempting lazy unmount with fusermount -uz...")
        try:
            subprocess.run(["fusermount", "-uz", mountpoint], check=True)
            logging.info("Unmounted with fusermount -uz.")
        except Exception as e2:
            logging.error(f"Lazy unmount failed: {e2}")

def sequential_write_workload(mountpoint, duration, rate, block_size):
    """
    Writes 'block_size' bytes in append mode to 'testfile' at 'rate' ops/sec for 'duration' seconds.
    Returns (total_bytes, elapsed_time).
    """
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start = time.time()
    data = b'A' * block_size
    interval = 1.0 / rate
    ops = 0
    try:
        with open(filename, "ab") as f:
            while True:
                now = time.time()
                if now - start >= duration:
                    break
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                total_bytes += len(data)
                ops += 1
                if ops % 200 == 0:
                    logging.info(f"Progress: {ops} ops, {total_bytes} bytes written so far")
                time.sleep(interval)
    except Exception as e:
        logging.error(f"Error in sequential write: {e}")
    elapsed = time.time() - start
    return total_bytes, elapsed

def main():
    logging.info("Starting FUSE FS Benchmark with hardcoded parameters.")
    # Prepare a dictionary to store results
    results = {}

    base_temp_dir = tempfile.mkdtemp(prefix="fuse_fs_bench_")
    logging.info(f"Base temp directory: {base_temp_dir}")

    try:
        for nd in DRIVE_LIST:
            mount_dir = os.path.join(base_temp_dir, f"mnt_{nd}")
            os.makedirs(mount_dir, exist_ok=True)

            logging.info(f"\n=== Testing with {nd} drives ===")
            fuse_proc = mount_fuse(nd, mount_dir)

            # Verify the mount
            if not verify_mount(mount_dir, timeout=30):
                logging.error("Mount verification failed. Unmounting and skipping...")
                unmount_fuse(mount_dir, fuse_proc)
                continue

            # Run the sequential write test
            logging.info(f"Running sequential write: duration={DURATION}s rate={RATE} ops/s block_size={BLOCK_SIZE} bytes")
            total_bytes, elapsed = sequential_write_workload(mount_dir, DURATION, RATE, BLOCK_SIZE)
            throughput = total_bytes / elapsed / (1024*1024) if elapsed > 0 else 0
            logging.info(f"Result for {nd} drives: {total_bytes} bytes written in {elapsed:.2f}s => {throughput:.2f} MB/s")
            results[nd] = throughput

            # Unmount after test
            unmount_fuse(mount_dir, fuse_proc)
            time.sleep(3)

    finally:
        shutil.rmtree(base_temp_dir)

    # Plot results
    drive_nums = sorted(results.keys())
    throughputs = [results[d] for d in drive_nums]

    plt.figure(figsize=(8,6))
    plt.plot(drive_nums, throughputs, marker='o', linestyle='-')
    plt.xlabel("Number of Drives")
    plt.ylabel("Throughput (MB/s)")
    plt.title("FUSE FS: Seq Write Throughput (No User Input)")
    plt.grid(True)
    plt.savefig("fuse_fs_benchmark_no_input.png")
    logging.info("Benchmark complete. Plot saved as fuse_fs_benchmark_no_input.png.")
    plt.show()

if __name__ == "__main__":
    main()
