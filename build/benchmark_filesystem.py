#!/usr/bin/env python3

import os
import time
import csv
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================
# Configuration Parameters
# ==============================

DEFAULT_MOUNT_POINT = "/tmp/fuse_mount"          # Default mount point
DEFAULT_TEST_DIR = "benchmark_test_dir"          # Directory within mount point for testing
DEFAULT_RESULTS_DIR = "./benchmark_results"      # Directory to store benchmark results
DEFAULT_SUMMARY_FILE = "summary.csv"             # Summary CSV file name

# Benchmark Parameters
SEQ_WRITE_SIZE_MB = 1000      # Size for sequential write in MB (e.g., 1000 MB = 1 GB)
SEQ_READ_SIZE_MB = 1000       # Size for sequential read in MB
RANDOM_IOPS_COUNT = 10000     # Number of random I/O operations
RANDOM_IOPS_BLOCK_SIZE = 4 * 1024  # Block size for random I/O in bytes (4 KB)

# ==============================
# Utility Functions
# ==============================

def log(message, log_file=None):
    """Prints and optionally logs messages to a file."""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def convert_bytes(size_str):
    """Converts human-readable byte sizes to bytes."""
    size_str = size_str.upper()
    if size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024 ** 2)
    elif size_str.endswith('G'):
        return int(float(size_str[:-1]) * 1024 ** 3)
    else:
        return int(size_str)

def setup_test_environment(mount_point, test_dir):
    """Sets up the test directory within the mount point."""
    full_test_dir = os.path.join(mount_point, test_dir)
    os.makedirs(full_test_dir, exist_ok=True)
    return full_test_dir

def cleanup_test_environment(full_test_dir):
    """Removes the test directory and its contents."""
    try:
        for root, dirs, files in os.walk(full_test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(full_test_dir)
    except Exception as e:
        log(f"Warning: Failed to clean up test environment. {e}")

# ==============================
# Benchmarking Functions
# ==============================

def sequential_write(test_file_path, size_mb):
    """Performs a sequential write test."""
    log(f"Starting Sequential Write Test: Writing {size_mb} MB to {test_file_path}")
    data = os.urandom(1024 * 1024)  # 1 MB of random data
    start_time = time.time()
    try:
        with open(test_file_path, 'wb') as f:
            for _ in range(size_mb):
                f.write(data)
        end_time = time.time()
        duration = end_time - start_time
        bandwidth = size_mb / duration
        log(f"Sequential Write: {bandwidth:.2f} MB/s over {duration:.2f} seconds")
        return bandwidth, duration
    except Exception as e:
        log(f"Error during Sequential Write Test: {e}")
        return None, None

def sequential_read(test_file_path, size_mb):
    """Performs a sequential read test."""
    log(f"Starting Sequential Read Test: Reading {size_mb} MB from {test_file_path}")
    start_time = time.time()
    try:
        with open(test_file_path, 'rb') as f:
            for _ in range(size_mb):
                f.read(1024 * 1024)  # Read 1 MB
        end_time = time.time()
        duration = end_time - start_time
        bandwidth = size_mb / duration
        log(f"Sequential Read: {bandwidth:.2f} MB/s over {duration:.2f} seconds")
        return bandwidth, duration
    except Exception as e:
        log(f"Error during Sequential Read Test: {e}")
        return None, None

def random_write_operation(file_path, block_size):
    """Performs a single random write operation."""
    try:
        with open(file_path, 'wb') as f:
            f.write(os.urandom(block_size))
    except Exception as e:
        log(f"Error during Random Write Operation: {e}")

def random_read_operation(file_path, block_size):
    """Performs a single random read operation."""
    try:
        with open(file_path, 'rb') as f:
            f.read(block_size)
    except Exception as e:
        log(f"Error during Random Read Operation: {e}")

def random_write_iops(test_dir, count, block_size):
    """Performs random write IOPS test."""
    log(f"Starting Random Write IOPS Test: {count} operations with {block_size} bytes each")
    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(count):
                file_path = os.path.join(test_dir, f"rw_test_write_{i}.bin")
                futures.append(executor.submit(random_write_operation, file_path, block_size))
            for future in as_completed(futures):
                pass  # Errors are already logged in the operation functions
        end_time = time.time()
        duration = end_time - start_time
        iops = count / duration
        log(f"Random Write IOPS: {iops:.2f} IOPS over {duration:.2f} seconds")
        return iops, duration
    except Exception as e:
        log(f"Error during Random Write IOPS Test: {e}")
        return None, None

def random_read_iops(test_dir, count, block_size):
    """Performs random read IOPS test."""
    log(f"Starting Random Read IOPS Test: {count} operations with {block_size} bytes each")
    start_time = time.time()
    try:
        # Pre-create files to read
        for i in range(count):
            file_path = os.path.join(test_dir, f"rw_test_read_{i}.bin")
            with open(file_path, 'wb') as f:
                f.write(os.urandom(block_size))
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(count):
                file_path = os.path.join(test_dir, f"rw_test_read_{i}.bin")
                futures.append(executor.submit(random_read_operation, file_path, block_size))
            for future in as_completed(futures):
                pass  # Errors are already logged in the operation functions
        end_time = time.time()
        duration = end_time - start_time
        iops = count / duration
        log(f"Random Read IOPS: {iops:.2f} IOPS over {duration:.2f} seconds")
        return iops, duration
    except Exception as e:
        log(f"Error during Random Read IOPS Test: {e}")
        return None, None

# ==============================
# Main Benchmarking Logic
# ==============================

def run_benchmarks(args):
    """Runs all benchmarks and logs the results."""
    mount_point = args.mount_point
    test_dir_name = args.test_dir
    results_dir = args.results_dir
    summary_file = os.path.join(results_dir, DEFAULT_SUMMARY_FILE)

    # Setup test environment
    full_test_dir = setup_test_environment(mount_point, test_dir_name)
    log(f"Test directory: {full_test_dir}")

    # Initialize Summary CSV
    with open(summary_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Tests will append rows below

    # Sequential Write Test
    seq_write_bandwidth, seq_write_duration = sequential_write(
        os.path.join(full_test_dir, "seq_write_test.bin"),
        SEQ_WRITE_SIZE_MB
    )
    if seq_write_bandwidth:
        with open(summary_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Sequential Write", "Bandwidth (MB/s)", f"{seq_write_bandwidth:.2f}"])

    # Sequential Read Test
    seq_read_bandwidth, seq_read_duration = sequential_read(
        os.path.join(full_test_dir, "seq_write_test.bin"),
        SEQ_READ_SIZE_MB
    )
    if seq_read_bandwidth:
        with open(summary_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Sequential Read", "Bandwidth (MB/s)", f"{seq_read_bandwidth:.2f}"])

    # Random Write IOPS Test
    rand_write_iops, rand_write_duration = random_write_iops(
        full_test_dir,
        RANDOM_IOPS_COUNT,
        RANDOM_IOPS_BLOCK_SIZE
    )
    if rand_write_iops:
        with open(summary_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Random Write IOPS", "IOPS", f"{rand_write_iops:.2f}"])

    # Random Read IOPS Test
    rand_read_iops, rand_read_duration = random_read_iops(
        full_test_dir,
        RANDOM_IOPS_COUNT,
        RANDOM_IOPS_BLOCK_SIZE
    )
    if rand_read_iops:
        with open(summary_file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Random Read IOPS", "IOPS", f"{rand_read_iops:.2f}"])

    # Cleanup test environment
    cleanup_test_environment(full_test_dir)

    log(f"Benchmarking completed. Results are stored in '{summary_file}'.")

# ==============================
# Argument Parsing
# ==============================

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmarking Tool for Mounted FUSE-Based Filesystem")
    parser.add_argument(
        '--mount-point',
        type=str,
        default=DEFAULT_MOUNT_POINT,
        help=f"Mount point directory (default: {DEFAULT_MOUNT_POINT})"
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=DEFAULT_TEST_DIR,
        help=f"Test directory name within mount point (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to store benchmark results (default: {DEFAULT_RESULTS_DIR})"
    )
    return parser.parse_args()

# ==============================
# Entry Point
# ==============================

def main():
    args = parse_arguments()

    # Check if mount point exists and is mounted
    if not os.path.ismount(args.mount_point):
        log(f"Error: '{args.mount_point}' is not a mounted filesystem.")
        sys.exit(1)
    else:
        log(f"Mounted filesystem detected at '{args.mount_point}'.")

    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)

    # Run benchmarks
    run_benchmarks(args)

if __name__ == "__main__":
    main()
