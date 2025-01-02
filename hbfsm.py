#!/usr/bin/env python3
import os
import time
import random
import subprocess
import matplotlib.pyplot as plt
import csv
import itertools

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
FUSE_BINARY      = "/home/parallels/Documents/framework/fuse_hash_fs/build/my_fs"
MOUNTPOINT       = "/tmp/myfuse"

DRIVES_LIST      = [1, 2, 4, 8]         # Number of drives to test
FILE_SIZES       = [64*1024, 1*1024*1024]  # 64 KB, 1 MB
NUM_FILES_LIST   = [10, 100]            # 10 files, 100 files
PATTERNS         = ["sequential", "random"]  # Access patterns
WORKLOAD_TYPES   = ["read", "write", "mixed"] # Workload types
BLOCK_SIZE       = 4 * 1024             # Fixed block size: 4 KB

# Filenames for saved plots
SEQ_FIG_FILENAME  = "sequential_performance.png"
RAND_FIG_FILENAME = "random_performance.png"

# CSV filename
CSV_FILENAME      = "benchmark_results.csv"

# -------------------------------------------------------------------------

def ensure_mountpoint():
    """Ensure the mountpoint directory exists."""
    if not os.path.exists(MOUNTPOINT):
        os.makedirs(MOUNTPOINT, exist_ok=True)

def mount_fuse(drives):
    """
    Mount the FUSE filesystem with `-o drives=<drives>`.
    Returns a subprocess handle to manage the FUSE process.
    """
    print(f"[Info] Mounting FUSE with drives={drives}")
    cmd = [FUSE_BINARY, '-o', f"drives={drives}", MOUNTPOINT]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait a bit to ensure FUSE mounts successfully
    time.sleep(2)
    return proc

def unmount_fuse(proc):
    """
    Unmount the FUSE filesystem and terminate the FUSE process.
    """
    print("[Info] Unmounting FUSE...")
    subprocess.run(["fusermount", "-u", MOUNTPOINT], check=False)
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except:
        pass

def do_write_files(num_files, file_size, pattern):
    """
    Write `num_files` each of size `file_size` in increments of BLOCK_SIZE.
    If `pattern == "random"`, perform random offset writes; else, sequential writes.
    Returns (total_bytes_written, elapsed_time).
    """
    data_block = b"A" * BLOCK_SIZE

    start = time.time()
    for i in range(num_files):
        path = os.path.join(MOUNTPOINT, f"f{i}.dat")
        with open(path, "wb") as f:
            written = 0
            while written < file_size:
                if pattern == "random":
                    max_offset_blocks = max(file_size // BLOCK_SIZE, 1)  # Avoid division by zero
                    off_blocks = random.randint(0, max_offset_blocks - 1)
                    f.seek(off_blocks * BLOCK_SIZE, 0)
                else:
                    # Sequential write
                    f.seek(written, 0)

                to_write = min(BLOCK_SIZE, file_size - written)
                f.write(data_block[:to_write])
                written += to_write

    end = time.time()
    total_bytes = num_files * file_size
    return total_bytes, (end - start)

def do_read_files(num_files, file_size, pattern):
    """
    Read `num_files` each of size `file_size` in increments of BLOCK_SIZE.
    If `pattern == "random"`, perform random offset reads; else, sequential reads.
    Returns (total_bytes_read, elapsed_time).
    """
    start = time.time()
    for i in range(num_files):
        path = os.path.join(MOUNTPOINT, f"f{i}.dat")
        with open(path, "rb") as f:
            read_amount = 0
            while read_amount < file_size:
                if pattern == "random":
                    max_offset_blocks = max(file_size // BLOCK_SIZE, 1)
                    off_blocks = random.randint(0, max_offset_blocks - 1)
                    f.seek(off_blocks * BLOCK_SIZE, 0)
                else:
                    # Sequential read
                    f.seek(read_amount, 0)

                chunk = f.read(BLOCK_SIZE)
                if not chunk:
                    break
                read_amount += len(chunk)
    end = time.time()
    total_bytes = num_files * file_size
    return total_bytes, (end - start)

def remove_files(num_files):
    """Remove the test files."""
    for i in range(num_files):
        path = os.path.join(MOUNTPOINT, f"f{i}.dat")
        if os.path.exists(path):
            os.remove(path)

def run_test(drives, file_size, num_files, pattern, workload):
    """
    1) Mount FUSE with specified drives.
    2) Perform the specified workload (read, write, mixed).
    3) Unmount FUSE.
    Returns (MBps, IOPS).
    """
    proc = mount_fuse(drives)
    start_all = time.time()
    total_ops = 0
    total_bytes = 0

    if workload == "write":
        # Only write operations
        wbytes, wtime = do_write_files(num_files, file_size, pattern)
        total_bytes += wbytes
        total_ops   += num_files  # File-level operations

    elif workload == "read":
        # Ensure files exist by performing a quick sequential write
        do_write_files(num_files, file_size, "sequential")
        rbytes, rtime = do_read_files(num_files, file_size, pattern)
        total_bytes += rbytes
        total_ops   += num_files

    else:  # "mixed"
        # Perform write followed by read
        wbytes, _ = do_write_files(num_files, file_size, pattern)
        rbytes, _ = do_read_files(num_files, file_size, pattern)
        total_bytes += (wbytes + rbytes)
        total_ops   += (2 * num_files)

    # Clean up
    remove_files(num_files)
    end_all = time.time()

    # Unmount FUSE
    unmount_fuse(proc)

    # Calculate elapsed time
    elapsed = end_all - start_all
    if elapsed <= 0:
        return 0, 0

    # Calculate MBps and IOPS
    mbps = (total_bytes / (1024 * 1024)) / elapsed
    iops = total_ops / elapsed
    return mbps, iops

def main():
    ensure_mountpoint()

    # Prepare CSV file
    with open(CSV_FILENAME, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["pattern", "workload", "drives", "file_size_KB", "num_files", "MBps", "IOPS"])

        # Iterate over all parameter combinations
        for pattern, workload, drives, file_size, num_files in itertools.product(
            PATTERNS, WORKLOAD_TYPES, DRIVES_LIST, FILE_SIZES, NUM_FILES_LIST
        ):
            print(f"=== Running: pattern={pattern}, workload={workload}, drives={drives}, "
                  f"file_size={file_size//1024}KB, num_files={num_files} ===")
            mbps, iops = run_test(drives, file_size, num_files, pattern, workload)
            print(f"[Result] MBps={mbps:.2f}, IOPS={iops:.2f}\n")
            csvwriter.writerow([pattern, workload, drives, file_size//1024, num_files, f"{mbps:.2f}", f"{iops:.2f}"])

    # Read results back for plotting
    results = []
    with open(CSV_FILENAME, "r") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            results.append(row)

    # Organize data for plotting
    from collections import defaultdict

    # Separate data for sequential and random patterns
    data_seq = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # workload -> (file_size, num_files) -> drives -> MBps
    data_rand = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) # similar for random

    for row in results:
        pattern = row["pattern"]
        workload = row["workload"]
        drives = int(row["drives"])
        file_size = int(row["file_size_KB"])
        num_files = int(row["num_files"])
        mbps = float(row["MBps"])
        iops = float(row["IOPS"])

        key = (file_size, num_files)
        if pattern == "sequential":
            data_seq[workload][key][drives] = mbps
        else:
            data_rand[workload][key][drives] = mbps

    # Define a helper function to plot
    def plot_performance(data, pattern_label, fig_filename):
        """
        Plot performance data.
        `data` is a dict: workload -> (file_size, num_files) -> drives -> MBps
        `pattern_label` is a string indicating the access pattern
        `fig_filename` is the filename to save the plot
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"{pattern_label.capitalize()} Access Performance", fontsize=16)

        workload_order = ["read", "write", "mixed"]
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

        for idx, workload in enumerate(workload_order):
            ax = axs[idx]
            ax.set_title(f"{workload.capitalize()} Workload", fontsize=14)
            ax.set_xlabel("Number of Drives", fontsize=12)
            if idx == 0:
                ax.set_ylabel("Throughput (MB/s)", fontsize=12)

            # For each (file_size, num_files), plot a separate line
            for i, ((file_size, num_files), drives_dict) in enumerate(data[workload].items()):
                drives_sorted = sorted(drives_dict.keys())
                mbps_sorted = [drives_dict[dr] for dr in drives_sorted]
                label = f"{file_size//1024}KB, {num_files} files"
                color = colors[i % len(colors)]
                ax.plot(drives_sorted, mbps_sorted, marker='o', label=label, color=color)

            ax.grid(True)
            # Place the legend outside the plot
            ax.legend(title="File Size & Count", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the main title
        plt.savefig(fig_filename)
        print(f"[Info] Saved plot to {fig_filename}")
        plt.close()

    # Plot for Sequential Access
    plot_performance(data_seq, "sequential", SEQ_FIG_FILENAME)

    # Plot for Random Access
    plot_performance(data_rand, "random", RAND_FIG_FILENAME)

    print("[Info] Benchmarking completed successfully.")

if __name__ == "__main__":
    main()
