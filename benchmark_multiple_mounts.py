#!/usr/bin/env python3

import os
import time
import csv
import sys
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress specific warnings (like libEGL and libGL warnings)
warnings.filterwarnings("ignore", message=".*libEGL.*")
warnings.filterwarnings("ignore", message=".*libGL.*")

# ==============================
# Configuration Parameters
# ==============================

# Drive configurations array
DRIVE_CONFIGURATIONS = [1, 2, 3, 4, 5, 8, 12, 16]

# Base directory for mount points
BASE_MOUNT_DIR = "/tmp"

# Default test directory within each mount point
DEFAULT_TEST_DIR = "benchmark_test_dir"

# Directory to store benchmark results and graphs
RESULTS_DIR = "./benchmark_results"

# Summary CSV file
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary.csv")

# Benchmark Parameters
SEQ_TEST_FILE_SIZE_MB = 500            # Fixed Sequential Test file size in MB
RANDOM_IOPS_BLOCK_SIZES_KB = [1, 2, 4, 8, 16]  # Block sizes in KB for Random IOPS
RANDOM_IOPS_COUNT = 3000                # Number of random operations per block size

# Graphing Parameters
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")

# ==============================
# Utility Functions
# ==============================y


def log(message):
    """Prints messages to stdout with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def setup_test_environment(mount_point, test_dir_name):
    """Creates the test directory within the mount point."""
    full_test_dir = os.path.join(mount_point, test_dir_name)
    try:
        os.makedirs(full_test_dir, exist_ok=True)
        log(f"Created test directory: {full_test_dir}")
    except Exception as e:
        log(f"Error creating test directory '{full_test_dir}': {e}")
        sys.exit(1)
    return full_test_dir

def cleanup_test_environment(full_test_dir):
    """Removes the test directory and its contents."""
    try:
        if not os.path.exists(full_test_dir):
            log(f"Test directory '{full_test_dir}' does not exist. Skipping cleanup.")
            return
        for root, dirs, files in os.walk(full_test_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                    log(f"Deleted file: {file_path}")
                except Exception as e:
                    log(f"Error deleting file '{file_path}': {e}")
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.rmdir(dir_path)
                    log(f"Deleted directory: {dir_path}")
                except Exception as e:
                    log(f"Error deleting directory '{dir_path}': {e}")
        os.rmdir(full_test_dir)
        log(f"Cleaned up test directory: {full_test_dir}")
    except Exception as e:
        log(f"Warning: Failed to clean up test directory '{full_test_dir}': {e}")

def initialize_results_csv():
    """Initializes the summary CSV with headers."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Drive Count", "Mount Point", "Test Type", "Operation Size", "Metric", "Value"])
        log(f"Initialized summary CSV at '{SUMMARY_CSV}'")
    else:
        log(f"Summary CSV already exists at '{SUMMARY_CSV}'")

def record_result(drive_count, mount_point, test_type, operation_size, metric, value):
    """Appends a benchmark result to the summary CSV."""
    with open(SUMMARY_CSV, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([drive_count, mount_point, test_type, operation_size, metric, value])

def convert_kb_to_bytes(kb):
    """Converts kilobytes to bytes."""
    return kb * 1024

def parse_operation_size(operation_size_str, test_type):
    """
    Parses the operation size string and returns size in KB or MB based on test type.
    For Random IOPS, returns block size in KB.
    For Sequential I/O, returns file size in MB.
    """
    size_match = re.match(r"(\d+)\s*(Bytes|KB|MB|GB)", operation_size_str, re.I)
    if not size_match:
        log(f"Unable to parse operation size: '{operation_size_str}'")
        return None
    size, unit = size_match.groups()
    size = int(size)
    unit = unit.upper()
    if test_type in ['Random Write IOPS', 'Random Read IOPS']:
        # For Random IOPS, operation size is block size
        if unit == 'BYTES':
            return size / 1024  # Convert to KB for consistency
        elif unit == 'KB':
            return size
        elif unit == 'MB':
            return size * 1024
        elif unit == 'GB':
            return size * 1024 * 1024
    elif test_type in ['Sequential Write', 'Sequential Read']:
        # For Sequential I/O, operation size is total file size
        if unit == 'MB':
            return size
        elif unit == 'GB':
            return size * 1024
        elif unit == 'KB':
            return size / 1024
        elif unit == 'BYTES':
            return size / (1024 * 1024)
    return None

# ==============================
# Benchmarking Functions
# ==============================

def sequential_write(test_file_path, size_mb):
    """Performs a sequential write test."""
    log(f"Starting Sequential Write: Writing {size_mb} MB to '{test_file_path}'")
    data = os.urandom(1024 * 1024)  # 1 MB of random data
    start_time = time.time()
    try:
        with open(test_file_path, 'wb') as f:
            for _ in range(size_mb):
                f.write(data)
        end_time = time.time()
        duration = end_time - start_time
        bandwidth = size_mb / duration
        log(f"Sequential Write Completed: {bandwidth:.2f} MB/s over {duration:.2f} seconds")
        return bandwidth, duration
    except Exception as e:
        log(f"Error during Sequential Write: {e}")
        return None, None

def sequential_read(test_file_path, size_mb):
    """Performs a sequential read test."""
    log(f"Starting Sequential Read: Reading {size_mb} MB from '{test_file_path}'")
    start_time = time.time()
    try:
        with open(test_file_path, 'rb') as f:
            for _ in range(size_mb):
                f.read(1024 * 1024)  # Read 1 MB
        end_time = time.time()
        duration = end_time - start_time
        bandwidth = size_mb / duration
        log(f"Sequential Read Completed: {bandwidth:.2f} MB/s over {duration:.2f} seconds")
        return bandwidth, duration
    except Exception as e:
        log(f"Error during Sequential Read: {e}")
        return None, None

def random_write_operation(file_path, block_size):
    """Performs a single random write operation."""
    try:
        with open(file_path, 'wb') as f:
            f.write(os.urandom(block_size))
    except Exception as e:
        log(f"Error during Random Write Operation on '{file_path}': {e}")

def random_read_operation(file_path, block_size):
    """Performs a single random read operation."""
    try:
        with open(file_path, 'rb') as f:
            f.read(block_size)
    except Exception as e:
        log(f"Error during Random Read Operation on '{file_path}': {e}")

def random_write_iops(test_dir, count, block_size_bytes):
    """Performs random write IOPS test."""
    log(f"Starting Random Write IOPS Test: {count} operations with {block_size_bytes} bytes each")
    start_time = time.time()
    try:
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(count):
                file_path = os.path.join(test_dir, f"rw_test_write_{i}.bin")
                futures.append(executor.submit(random_write_operation, file_path, block_size_bytes))
            # Wait for all operations to complete
            for future in as_completed(futures):
                pass  # Errors are already logged
        end_time = time.time()
        duration = end_time - start_time
        iops = count / duration
        log(f"Random Write IOPS Completed: {iops:.2f} IOPS over {duration:.2f} seconds")
        return iops, duration
    except Exception as e:
        log(f"Error during Random Write IOPS Test: {e}")
        return None, None

def random_read_iops(test_dir, count, block_size_bytes):
    """Performs random read IOPS test."""
    log(f"Starting Random Read IOPS Test: {count} operations with {block_size_bytes} bytes each")
    start_time = time.time()
    try:
        # Pre-create files to read
        for i in range(count):
            file_path = os.path.join(test_dir, f"rw_test_read_{i}.bin")
            try:
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(block_size_bytes))
            except Exception as e:
                log(f"Error pre-creating file '{file_path}' for Random Read IOPS: {e}")
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for i in range(count):
                file_path = os.path.join(test_dir, f"rw_test_read_{i}.bin")
                futures.append(executor.submit(random_read_operation, file_path, block_size_bytes))
            # Wait for all operations to complete
            for future in as_completed(futures):
                pass  # Errors are already logged
        end_time = time.time()
        duration = end_time - start_time
        iops = count / duration
        log(f"Random Read IOPS Completed: {iops:.2f} IOPS over {duration:.2f} seconds")
        return iops, duration
    except Exception as e:
        log(f"Error during Random Read IOPS Test: {e}")
        return None, None

# ==============================
# Graphing Functions
# ==============================

def generate_graphs():
    """Generates and saves graphs based on the benchmark summary CSV."""
    log("Generating graphs...")
    
    # Ensure the graph directory exists
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    # Load the summary CSV
    try:
        df = pd.read_csv(SUMMARY_CSV)
        log(f"Loaded benchmark data from '{SUMMARY_CSV}'")
    except Exception as e:
        log(f"Error loading summary CSV: {e}")
        return
    
    # Ensure 'Value' is numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # Drop rows with NaN in 'Value'
    df = df.dropna(subset=['Value'])
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # ==============================
    # Data Parsing and Preparation
    # ==============================
    
    # Parse 'Operation Size' to extract numerical values
    df['Operation Size (Numeric)'] = df.apply(
        lambda row: parse_operation_size(row['Operation Size'], row['Test Type']),
        axis=1
    )
    
    # Convert 'Drive Count' to string for categorical plotting
    df['Drive Count'] = df['Drive Count'].astype(int).astype(str)
    
    # Handle potential parsing errors
    if df['Operation Size (Numeric)'].isnull().any():
        log("Warning: Some operation sizes could not be parsed and will be excluded from graphs.")
        df = df.dropna(subset=['Operation Size (Numeric)'])
    
    # ==============================
    # Sequential Bandwidth Graph
    # ==============================
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df[df['Test Type'].isin(['Sequential Write', 'Sequential Read'])],
        x='Drive Count',
        y='Value',
        hue='Test Type',
        marker='o',
        palette='viridis'
    )
    plt.title('Sequential Write and Read Bandwidth vs Number of Drives')
    plt.xlabel('Number of Drives')
    plt.ylabel('Bandwidth (MB/s)')
    plt.xticks(range(1, max(df['Drive Count'].astype(int)) + 1))
    plt.legend(title='Test Type')
    plt.tight_layout()
    seq_bw_graph = os.path.join(GRAPH_DIR, 'sequential_bandwidth.png')
    plt.savefig(seq_bw_graph)
    plt.close()
    log(f"Saved Sequential Bandwidth graph to '{seq_bw_graph}'")
    
    # ==============================
    # Random Write IOPS vs Drive Count Graph
    # ==============================
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df[df['Test Type'] == 'Random Write IOPS'],
        x='Drive Count',
        y='Value',
        hue='Operation Size (Numeric)',
        marker='o',
        palette='magma'
    )
    plt.title('Random Write IOPS vs Number of Drives for Different Block Sizes')
    plt.xlabel('Number of Drives')
    plt.ylabel('IOPS')
    plt.xticks(range(1, max(df['Drive Count'].astype(int)) + 1))
    plt.legend(title='Block Size (KB)', title_fontsize='13', fontsize='11')
    plt.tight_layout()
    rand_write_iops_graph = os.path.join(GRAPH_DIR, 'random_write_iops.png')
    plt.savefig(rand_write_iops_graph)
    plt.close()
    log(f"Saved Random Write IOPS graph to '{rand_write_iops_graph}'")
    
    # ==============================
    # Random Read IOPS vs Drive Count Graph
    # ==============================
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df[df['Test Type'] == 'Random Read IOPS'],
        x='Drive Count',
        y='Value',
        hue='Operation Size (Numeric)',
        marker='o',
        palette='magma'
    )
    plt.title('Random Read IOPS vs Number of Drives for Different Block Sizes')
    plt.xlabel('Number of Drives')
    plt.ylabel('IOPS')
    plt.xticks(range(1, max(df['Drive Count'].astype(int)) + 1))
    plt.legend(title='Block Size (KB)', title_fontsize='13', fontsize='11')
    plt.tight_layout()
    rand_read_iops_graph = os.path.join(GRAPH_DIR, 'random_read_iops.png')
    plt.savefig(rand_read_iops_graph)
    plt.close()
    log(f"Saved Random Read IOPS graph to '{rand_read_iops_graph}'")
    
    # ==============================
    # Random IOPS vs Block Size across Different Drives
    # ==============================
    
    # Filter for Random IOPS Tests
    random_iops_df = df[df['Test Type'].isin(['Random Write IOPS', 'Random Read IOPS'])]
    
    # Create a pivot table for plotting
    pivot_df = random_iops_df.pivot_table(
        index=['Test Type', 'Operation Size (Numeric)'],
        columns='Drive Count',
        values='Value',
        aggfunc='mean'
    ).reset_index()
    
    # Melt the pivot table for seaborn
    melt_df = pivot_df.melt(id_vars=['Test Type', 'Operation Size (Numeric)'], var_name='Drive Count', value_name='IOPS')
    
    # Plot Random IOPS vs Block Size for different Drive Counts
    plt.figure(figsize=(14, 10))
    sns.lineplot(
        data=melt_df,
        x='Operation Size (Numeric)',
        y='IOPS',
        hue='Drive Count',
        style='Test Type',
        markers=True,
        dashes=False,
        palette='plasma'
    )
    plt.title('Random IOPS vs Block Size across Different Number of Drives')
    plt.xlabel('Block Size (KB)')
    plt.ylabel('IOPS')
    plt.legend(title='Number of Drives', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    rand_iops_block_size_graph = os.path.join(GRAPH_DIR, 'random_iops_block_size_drives.png')
    plt.savefig(rand_iops_block_size_graph)
    plt.close()
    log(f"Saved Random IOPS vs Block Size across Drives graph to '{rand_iops_block_size_graph}'")
    
    # ==============================
    # Combined Sequential and Random IOPS Graph
    # ==============================
    
    # Combined Sequential and Random IOPS
    plt.figure(figsize=(14, 10))
    sns.lineplot(
        data=df[df['Test Type'].isin(['Sequential Write', 'Sequential Read', 'Random Write IOPS', 'Random Read IOPS'])],
        x='Drive Count',
        y='Value',
        hue='Test Type',
        style='Operation Size (Numeric)',
        markers=True,
        dashes=False,
        palette='Set2'
    )
    plt.title('Combined Sequential and Random IOPS vs Number of Drives')
    plt.xlabel('Number of Drives')
    plt.ylabel('Performance Metric')
    plt.xticks(range(1, max(df['Drive Count'].astype(int)) + 1))
    plt.legend(title='Test Type & Block Size', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    combined_iops_graph = os.path.join(GRAPH_DIR, 'combined_iops.png')
    plt.savefig(combined_iops_graph)
    plt.close()
    log(f"Saved Combined IOPS graph to '{combined_iops_graph}'")
    
    log("Graph generation completed.")

# ==============================
# Main Benchmarking Logic
# ==============================

def benchmark_mount_point(drive_count, mount_point, seq_test_size_mb, rand_iops_block_sizes_kb, rand_iops_count):
    """Benchmarks a single mount point with specified parameters."""
    log(f"\n=== Benchmarking Mount Point: {mount_point} (Drives: {drive_count}) ===")
    
    # Verify that the mount point is mounted
    if not os.path.ismount(mount_point):
        log(f"Error: Mount point '{mount_point}' is not mounted. Skipping benchmarks for this mount.")
        return
    
    # Sequential Write and Read Test
    test_dir = setup_test_environment(mount_point, DEFAULT_TEST_DIR)
    
    # Define test file paths
    seq_write_file = os.path.join(test_dir, f"seq_write_test_{seq_test_size_mb}MB.bin")
    seq_read_file = seq_write_file  # Read the same file
    
    # Perform Sequential Write Test
    bandwidth, duration = sequential_write(seq_write_file, seq_test_size_mb)
    if bandwidth:
        record_result(drive_count, mount_point, "Sequential Write", f"{seq_test_size_mb} MB", "Bandwidth (MB/s)", f"{bandwidth:.2f}")
    
    # Perform Sequential Read Test
    bandwidth, duration = sequential_read(seq_read_file, seq_test_size_mb)
    if bandwidth:
        record_result(drive_count, mount_point, "Sequential Read", f"{seq_test_size_mb} MB", "Bandwidth (MB/s)", f"{bandwidth:.2f}")
    
    # Cleanup after sequential tests
    cleanup_test_environment(test_dir)
    
    # Random IOPS Tests with varying block sizes
    for block_size_kb in rand_iops_block_sizes_kb:
        block_size_bytes = convert_kb_to_bytes(block_size_kb)
        test_dir = setup_test_environment(mount_point, DEFAULT_TEST_DIR)
        
        # Perform Random Write IOPS Test
        iops, duration = random_write_iops(test_dir, rand_iops_count, block_size_bytes)
        if iops:
            record_result(drive_count, mount_point, "Random Write IOPS", f"{block_size_bytes} Bytes", "IOPS", f"{iops:.2f}")
        
        # Perform Random Read IOPS Test
        iops, duration = random_read_iops(test_dir, rand_iops_count, block_size_bytes)
        if iops:
            record_result(drive_count, mount_point, "Random Read IOPS", f"{block_size_bytes} Bytes", "IOPS", f"{iops:.2f}")
        
        # Cleanup after random IOPS tests
        cleanup_test_environment(test_dir)

def run_benchmarks(drive_configurations, seq_test_size_mb, rand_iops_block_sizes_kb, rand_iops_count):
    """Runs benchmarks on all specified mount points with fixed operation sizes."""
    initialize_results_csv()
    
    for drive_count in drive_configurations:
        mount_point = os.path.join(BASE_MOUNT_DIR, f"fuse_mount{drive_count}")
        benchmark_mount_point(drive_count, mount_point, seq_test_size_mb, rand_iops_block_sizes_kb, rand_iops_count)
    
    log(f"\nAll benchmarks completed. Results are stored in '{SUMMARY_CSV}'.")
    
    # Generate graphs based on the benchmark results
    generate_graphs()

# ==============================
# Entry Point
# ==============================

def main():
    """Main function to execute the benchmarking script."""
    log("=== Welcome to the FUSE-Based Filesystem Benchmarking Tool ===\n")
    
    # Log the drive configurations being used
    log("Drive Configurations:")
    for drive_count in DRIVE_CONFIGURATIONS:
        mount_point = os.path.join(BASE_MOUNT_DIR, f"fuse_mount{drive_count}")
        log(f"  Drive Count: {drive_count}, Mount Point: {mount_point}")
    
    # Confirm to the user before starting benchmarks
    confirm = input("\nProceed with benchmarking? (y/n): ").strip().lower()
    if confirm != 'y':
        log("Benchmarking aborted by the user.")
        sys.exit(0)
    
    run_benchmarks(DRIVE_CONFIGURATIONS, SEQ_TEST_FILE_SIZE_MB, RANDOM_IOPS_BLOCK_SIZES_KB, RANDOM_IOPS_COUNT)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nBenchmarking interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
