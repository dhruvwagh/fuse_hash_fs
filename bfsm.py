#!/usr/bin/env python3
import subprocess
import time
import os
import math
import threading
import argparse
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
FUSE_BINARY  = "/home/parallels/Documents/framework/fuse_hash_fs/build/my_fs"
MOUNTPOINT   = "/tmp/myfuse"
DRIVES_LIST  = [1, 2, 4, 8, 12, 16]

# Total data we want to write+read
TOTAL_BYTES  = 32 * 1024 * 1024   # 32 MB total

# We define how large each file should be
FILE_SIZE    = 64 * 1024          # e.g., 64 KB per file to allow more concurrency overlap
CHUNK_SIZE   = 4 * 1024           # We do our writes/reads in 4 KB increments

# Number of concurrency threads
THREADS = 8

# -------------------------------------------------------------------------

def ensure_mountpoint():
    """Make sure our mountpoint directory exists."""
    if not os.path.exists(MOUNTPOINT):
        os.makedirs(MOUNTPOINT, exist_ok=True)

def mount_fuse(drives):
    """
    Mount the FUSE filesystem with `-o drives=<drives>` and return the process handle.
    """
    print(f"[Info] Mounting FUSE with drives={drives}")
    cmd = [FUSE_BINARY, '-o', f"drives={drives}", MOUNTPOINT]
    proc = subprocess.Popen(cmd)
    time.sleep(2)  # Give FUSE time to mount
    return proc

def unmount_fuse(proc):
    """
    Unmount the FUSE FS, then terminate the FUSE process if still alive.
    """
    print("[Info] Unmounting FUSE...")
    subprocess.run(["fusermount", "-u", MOUNTPOINT], check=False)
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except:
        pass

def worker_func(thread_idx, files_per_thread, results_dict):
    """
    Each thread:
      1) Creates & writes 'files_per_thread' distinct files (FILE_SIZE each) in CHUNK_SIZE increments
      2) Reads them back
      3) Removes them
    We'll measure how long the writes and reads took in total, returning them via results_dict.
    """

    # We'll create "files_per_thread" files named uniquely per thread
    data_block = b"A" * CHUNK_SIZE

    # Time write phase
    write_start = time.time()
    for i in range(files_per_thread):
        filename = os.path.join(MOUNTPOINT, f"th{thread_idx}_file{i}.dat")
        with open(filename, "wb") as f:
            written = 0
            while written < FILE_SIZE:
                f.write(data_block)
                written += CHUNK_SIZE
    write_end = time.time()

    # Time read phase
    read_start = time.time()
    buf = bytearray(CHUNK_SIZE)
    for i in range(files_per_thread):
        filename = os.path.join(MOUNTPOINT, f"th{thread_idx}_file{i}.dat")
        with open(filename, "rb") as f:
            read_total = 0
            while read_total < FILE_SIZE:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                read_total += len(chunk)
    read_end = time.time()

    # Remove files
    for i in range(files_per_thread):
        filename = os.path.join(MOUNTPOINT, f"th{thread_idx}_file{i}.dat")
        if os.path.exists(filename):
            os.remove(filename)

    # Store results
    results_dict[thread_idx] = {
        "write_time": (write_end - write_start),
        "read_time": (read_end - read_start)
    }

def run_benchmark():
    """
    We want to create + read 'TOTAL_BYTES' across multiple threads and files.

    We'll:
      1) Decide how many total files we want (e.g., 32 MB total / 64 KB each => 512 files)
      2) Split among 'THREADS' threads
      3) Each thread writes & reads its own subset of files
      4) We measure total times for all threads combined
    """
    # total_files = TOTAL_BYTES / FILE_SIZE
    total_files = TOTAL_BYTES // FILE_SIZE
    if total_files < 1:
        print("[Warning] FILE_SIZE is too big, we won't even create 1 file.")
        return (0,0,0,0)

    files_per_thread = total_files // THREADS
    # If not divisible, let's distribute remainder:
    remainder = total_files % THREADS

    # We'll track each thread's time
    results_dict = {}

    # Start all threads
    threads_list = []
    start_global = time.time()

    for t in range(THREADS):
        # Give 1 extra file if remainder > 0
        nfiles = files_per_thread + (1 if t < remainder else 0)
        th = threading.Thread(target=worker_func, args=(t, nfiles, results_dict))
        threads_list.append(th)
        th.start()

    for th in threads_list:
        th.join()

    end_global = time.time()
    elapsed_global = end_global - start_global

    # Consolidate times
    # For write MB/s, we sum all threads' data. Similarly for read MB/s.
    # Each thread wrote "nfiles * FILE_SIZE" bytes, then read same amount
    total_written = 0
    total_read = 0
    total_write_time = 0.0
    total_read_time  = 0.0

    for t, rd in results_dict.items():
        # We can find how many files that thread wrote:
        # but let's just store it
        pass

    # More accurate approach: each thread wrote "files_per_thread[t]" files if we tracked it
    # but for simplicity let's re-calc
    sum_files = 0
    for t in range(THREADS):
        nfiles = files_per_thread + (1 if t < remainder else 0)
        sum_files += nfiles

    # sum_files * FILE_SIZE is total written and also total read
    total_written = sum_files * FILE_SIZE
    total_read    = sum_files * FILE_SIZE

    # Now let's just measure global times for MB/s
    # We are focusing on overall concurrency, so the global elapsed time is what matters
    # We do not just sum all threads' times, because they run in parallel.
    write_MB = (total_written / (1024*1024))
    read_MB  = (total_read / (1024*1024))

    write_MBps = write_MB / elapsed_global
    read_MBps  = read_MB / elapsed_global

    # For IOPS, we consider each file creation as 1 "write op" and each file read as 1 "read op"
    # The concurrency doesn't reduce total op count, but the time is parallel
    total_write_ops = sum_files
    total_read_ops  = sum_files
    # IOPS is total ops / total time
    write_iops = total_write_ops / elapsed_global
    read_iops  = total_read_ops / elapsed_global

    return (write_MBps, read_MBps, write_iops, read_iops)

def main():
    ensure_mountpoint()

    results = []
    for drives in DRIVES_LIST:
        # 1) Mount FUSE
        fuse_proc = mount_fuse(drives)

        # 2) Run concurrency benchmark
        try:
            print(f"[Info] Starting concurrency benchmark with total {TOTAL_BYTES} bytes, file size {FILE_SIZE}, {THREADS} threads.")
            wmb, rmb, wiops, riops = run_benchmark()
            print(f"[Result] drives={drives} | "
                  f"Write={wmb:.2f} MB/s, Read={rmb:.2f} MB/s, "
                  f"WriteIOPS={wiops:.2f}, ReadIOPS={riops:.2f}\n")
            results.append((drives, wmb, rmb, wiops, riops))
        finally:
            # 3) Unmount
            unmount_fuse(fuse_proc)
            time.sleep(1)

    # 4) Plot results
    plot_results(results)

def plot_results(results):
    # results is list of tuples: (drives, writeMB, readMB, writeIOPS, readIOPS)
    drives_vals    = [r[0] for r in results]
    write_mb_vals  = [r[1] for r in results]
    read_mb_vals   = [r[2] for r in results]
    write_iops_vals= [r[3] for r in results]
    read_iops_vals = [r[4] for r in results]

    # We'll create two subplots: top for MB/s, bottom for IOPS
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # --- Plot MB/s on ax1 ---
    ax1.set_title(f"FUSE Parallel Scaling: {THREADS} threads, each file = {FILE_SIZE/1024} KB")
    ax1.set_xlabel("Number of Drives")
    ax1.set_ylabel("Throughput (MB/s)")
    ax1.plot(drives_vals, write_mb_vals, marker='o', color='blue', label='Write MB/s')
    ax1.plot(drives_vals, read_mb_vals, marker='^', color='red',  label='Read MB/s')
    ax1.legend()
    ax1.grid(True)

    # --- Plot IOPS on ax2 ---
    ax2.set_title("IOPS (Operations per Second)")
    ax2.set_xlabel("Number of Drives")
    ax2.set_ylabel("IOPS")
    ax2.plot(drives_vals, write_iops_vals, marker='o', color='green', label='Write IOPS')
    ax2.plot(drives_vals, read_iops_vals, marker='^', color='orange', label='Read IOPS')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig("scaling_results.png")
    print("[Info] Saved plot to scaling_results.png")
    plt.show()
    plt.savefig("scaling_results.png")

if __name__ == "__main__":
    main()
