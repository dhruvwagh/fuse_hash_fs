#!/usr/bin/env python3
"""
Simple Benchmark for FUSE Filesystem

This script:
  1. Creates a temporary mount point.
  2. Mounts the FUSE filesystem (fuse_fs.py) using the provided parameters.
  3. Verifies that the mount is working by checking that "testfile" exists.
  4. Runs a sequential write workload on "/testfile" for a specified duration and op rate.
  5. Calculates and prints the throughput (MB/s).
  6. Unmounts the filesystem and cleans up.

Usage:
  sudo python3 benchmark_fuse_simple.py --duration 10 --rate 1000 --block-size 4096 --drive-delay 0.01 --num-drives 4
"""

import os
import time
import argparse
import subprocess
import tempfile
import shutil
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def verify_mount(mountpoint, timeout=30):
    """
    Verify that the mount point is functional by checking that "testfile" exists.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            contents = os.listdir(mountpoint)
            logging.info(f"Mountpoint contents: {contents}")
            if "testfile" in contents:
                logging.info("Mount verification succeeded: 'testfile' found.")
                return True
        except Exception as e:
            logging.debug(f"Error listing {mountpoint}: {e}")
        time.sleep(1)
    logging.error("Mount verification timed out.")
    return False

def stream_process_output(proc):
    """Spawn threads to read and log the FUSE process's stdout and stderr."""
    import threading
    def stream_output(pipe, level):
        for line in iter(pipe.readline, b''):
            logging.log(level, line.decode().rstrip())
    threading.Thread(target=stream_output, args=(proc.stdout, logging.DEBUG), daemon=True).start()
    threading.Thread(target=stream_output, args=(proc.stderr, logging.DEBUG), daemon=True).start()

def mount_fuse(num_drives, mountpoint, drive_delay, block_size):
    """
    Launch the FUSE filesystem by running fuse_fs.py with the desired parameters.
    Returns the subprocess.Popen object.
    """
    cmd = [
        "sudo", "python3", "fuse_fs.py", mountpoint,
        "--num-drives", str(num_drives),
        "--drive-delay", str(drive_delay),
        "--block-size", str(block_size)
    ]
    logging.info("Mounting FUSE with command: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Start threads to stream the FUSE process output
    stream_process_output(proc)
    # Give the FUSE process a few seconds to initialize
    time.sleep(3)
    return proc

def unmount_fuse(mountpoint, proc):
    """
    Unmount the FUSE filesystem.
    First, attempt to terminate the FUSE process; then use fusermount to unmount.
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
        logging.error(f"Error unmounting with fusermount -u: {e}")
        logging.info("Attempting lazy unmount with fusermount -uz...")
        try:
            subprocess.run(["fusermount", "-uz", mountpoint], check=True)
            logging.info("Successfully unmounted using fusermount -uz.")
        except Exception as e:
            logging.error(f"Lazy unmount failed: {e}")

def sequential_write_workload(mountpoint, duration, rate, block_size):
    """
    Sequentially write blocks to "/testfile" in append mode.
    Returns (total_bytes, elapsed_time).
    """
    filename = os.path.join(mountpoint, "testfile")
    total_bytes = 0
    start_time = time.time()
    data = b'A' * block_size
    interval = 1.0 / rate
    ops = 0
    try:
        with open(filename, "ab") as f:
            end = time.time() + duration
            while time.time() < end:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                total_bytes += len(data)
                ops += 1
                # Log progress every 100 operations for debugging.
                if ops % 100 == 0:
                    logging.info(f"Sequential write progress: {ops} ops, {total_bytes} bytes written")
                time.sleep(interval)
    except Exception as e:
        logging.error(f"Sequential write error: {e}")
    elapsed = time.time() - start_time
    return total_bytes, elapsed

def main():
    parser = argparse.ArgumentParser(description="Simple Benchmark for FUSE Filesystem")
    parser.add_argument("--duration", type=int, default=10, help="Duration (in seconds) of the workload")
    parser.add_argument("--rate", type=int, default=1000, help="Write operations per second")
    parser.add_argument("--block-size", type=int, default=4096, help="Block size in bytes")
    parser.add_argument("--drive-delay", type=float, default=0.01, help="Simulated drive delay per block (seconds)")
    parser.add_argument("--num-drives", type=int, default=4, help="Number of simulated drives")
    args = parser.parse_args()

    # Create a temporary mount point
    base_mount = tempfile.mkdtemp(prefix="fuse_bench_")
    mountpoint = os.path.join(base_mount, "mnt")
    os.makedirs(mountpoint, exist_ok=True)
    logging.info(f"Temporary mount point: {mountpoint}")

    # Mount the FUSE filesystem
    fuse_proc = mount_fuse(args.num_drives, mountpoint, args.drive_delay, args.block_size)
    if not verify_mount(mountpoint, timeout=30):
        logging.error("Mount verification failed.")
        unmount_fuse(mountpoint, fuse_proc)
        shutil.rmtree(base_mount)
        sys.exit(1)
    logging.info("Mount verified.")

    # Run the sequential write workload
    total_bytes, elapsed = sequential_write_workload(mountpoint, args.duration, args.rate, args.block_size)
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    logging.info(f"Sequential Write: {total_bytes} bytes written in {elapsed:.2f} seconds, throughput: {throughput:.2f} MB/s")

    # Unmount the filesystem and clean up
    unmount_fuse(mountpoint, fuse_proc)
    shutil.rmtree(base_mount)

if __name__ == '__main__':
    main()
