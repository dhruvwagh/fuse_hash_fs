#!/usr/bin/env python3
"""
Rigorous FUSE Filesystem Benchmarking Script

This script:
  1) Mounts the FUSE filesystem with various 'drives=' settings
  2) Performs multiple workloads:
     - Sequential large-file R/W
     - Random-access R/W
     - Small-file metadata stress
  3) Logs throughput in MB/s and also times metadata ops
  4) Unmounts and repeats for next drive count
  5) Plots the results

Dependencies:
    - Python 3
    - matplotlib (for plotting)
    - 'fusermount' for unmounting
    - Possibly 'dd' or a local random approach for random IO
"""

import subprocess
import time
import os
import random
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
FUSE_BINARY  = "/home/dhruv/Documents/fuse_hash_fs/build/my_fs"
MOUNTPOINT   = "/home/dhruv/myfuse"
DRIVES_LIST  = [1, 2, 4, 8, 12, 16]  # Test these drive counts

# Shared test parameters
LARGEFILE_SIZE_MB   = 4       # per file (for sequential big-file test)
LARGEFILE_NUM_FILES = 100     # number of large files
LARGEFILE_CHUNK_KB  = 4       # chunk size in KB for sequential writes/reads

RANDOM_IO_FILE_SIZE_MB = 2    # single file size for random tests
RANDOM_IO_NUM_OPS      = 1000 # number of random reads/writes
RANDOM_IO_CHUNK_KB     = 4    # random read/write chunk size
RANDOM_IO_WRITE_RATIO  = 0.5  # fraction that are writes

SMALLFILE_NUM_FILES  = 1000   # how many small files to create
SMALLFILE_SIZE_BYTES = 4096   # each small file size
SMALLFILE_CHUNK_BYTES= 1024

# We'll store results as { scenario_name : [ (drives, throughput1, throughput2, ...), ... ] }
# Where scenario_name might be "seq_write", "seq_read", "rand_mix", "smallfile_metatest", etc.


# -------------------------------------------------------------------
# Basic Mount / Unmount
# -------------------------------------------------------------------
def mount_fuse(drives):
    """
    Spawns the FUSE filesystem as a subprocess with -o drives=NN
    Waits a bit to let it mount. Returns the Popen object.
    """
    print(f"[Info] Mounting FUSE with drives={drives}")
    cmd = [FUSE_BINARY, '-o', f"drives={drives}", MOUNTPOINT]
    proc = subprocess.Popen(cmd)
    time.sleep(2)  # give it time to mount
    return proc

def unmount_fuse(proc):
    """
    Unmounts from MOUNTPOINT, terminates the FUSE process gracefully.
    """
    print("[Info] Unmounting FUSE...")
    subprocess.run(["fusermount", "-u", MOUNTPOINT], check=False)
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except:
        pass

# -------------------------------------------------------------------
# 1) Sequential large-file test (write + read)
# -------------------------------------------------------------------
def scenario_sequential():
    """
    Writes LARGEFILE_NUM_FILES each of size LARGEFILE_SIZE_MB.
    Reads them back in full. Removes them.
    Returns (write_MBps, read_MBps).
    """
    # convert MB -> bytes
    file_size_bytes = LARGEFILE_SIZE_MB * 1024 * 1024
    chunk_bytes     = LARGEFILE_CHUNK_KB * 1024

    # --- Write test ---
    write_start = time.time()
    data_block = b"A" * chunk_bytes
    for i in range(LARGEFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"seq_{i}.dat")
        with open(path, "wb") as f:
            written = 0
            while written < file_size_bytes:
                f.write(data_block)
                written += chunk_bytes
    write_end = time.time()

    # --- Read test ---
    read_start = time.time()
    for i in range(LARGEFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"seq_{i}.dat")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_bytes)
                if not chunk:
                    break
    read_end = time.time()

    # Cleanup
    for i in range(LARGEFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"seq_{i}.dat")
        os.remove(path)

    # Compute throughput
    total_bytes = file_size_bytes * LARGEFILE_NUM_FILES
    write_MBps = (total_bytes / (1024*1024)) / (write_end - write_start)
    read_MBps  = (total_bytes / (1024*1024)) / (read_end - read_start)
    return (write_MBps, read_MBps)

# -------------------------------------------------------------------
# 2) Random IO test in a single file
# -------------------------------------------------------------------
def scenario_random_io():
    """
    Creates one file of size RANDOM_IO_FILE_SIZE_MB.
    Then does RANDOM_IO_NUM_OPS random R/W at random offsets.
    Returns (rw_iops, avg_latency_ms) or (rw_MBps, ???).
    We'll measure total time and compute MB/s for the portion that are reads/writes.
    """
    file_size_bytes = RANDOM_IO_FILE_SIZE_MB * 1024 * 1024
    chunk_bytes     = RANDOM_IO_CHUNK_KB * 1024

    path = os.path.join(MOUNTPOINT, "random_io.dat")

    # Create the file (pre-allocate)
    with open(path, "wb") as f:
        f.seek(file_size_bytes-1)
        f.write(b"\x00")

    # Let's do random offsets
    random_ops = RANDOM_IO_NUM_OPS
    write_ops = int(random_ops * RANDOM_IO_WRITE_RATIO)
    read_ops  = random_ops - write_ops

    data_block = b"Z" * chunk_bytes
    start = time.time()
    with open(path, "r+b") as f:
        for i in range(random_ops):
            offset = random.randint(0, file_size_bytes - chunk_bytes)
            f.seek(offset, 0)
            if i < write_ops:
                # do a write
                f.write(data_block)
            else:
                # do a read
                f.read(chunk_bytes)
    end = time.time()

    # remove
    os.remove(path)

    total_rw_bytes = random_ops * chunk_bytes  # total read+write
    rw_MBps = (total_rw_bytes / (1024*1024)) / (end - start)
    return rw_MBps

# -------------------------------------------------------------------
# 3) Small-file metadata stress test
# -------------------------------------------------------------------
def scenario_smallfiles():
    """
    Creates SMALLFILE_NUM_FILES small files, each size ~ SMALLFILE_SIZE_BYTES
    in increments of SMALLFILE_CHUNK_BYTES.
    Then stats them, then deletes them.
    Returns (create_speed, stat_speed, remove_speed) in ops/sec.
    """
    # creation
    create_start = time.time()
    data_block = b"M" * SMALLFILE_CHUNK_BYTES
    for i in range(SMALLFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"meta_{i}.dat")
        with open(path, "wb") as f:
            written = 0
            while written < SMALLFILE_SIZE_BYTES:
                f.write(data_block)
                written += SMALLFILE_CHUNK_BYTES
    create_end = time.time()

    # stat
    stat_start = time.time()
    for i in range(SMALLFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"meta_{i}.dat")
        st = os.stat(path)
    stat_end = time.time()

    # remove
    remove_start = time.time()
    for i in range(SMALLFILE_NUM_FILES):
        path = os.path.join(MOUNTPOINT, f"meta_{i}.dat")
        os.remove(path)
    remove_end = time.time()

    create_ops  = SMALLFILE_NUM_FILES / (create_end - create_start)
    stat_ops    = SMALLFILE_NUM_FILES / (stat_end - stat_start)
    remove_ops  = SMALLFILE_NUM_FILES / (remove_end - remove_start)
    return (create_ops, stat_ops, remove_ops)


# -------------------------------------------------------------------
# Main test harness
# -------------------------------------------------------------------
def main():
    if not os.path.exists(MOUNTPOINT):
        os.makedirs(MOUNTPOINT)

    # We'll store results in a dict: scenario -> list of tuples (drives, metricA, metricB, ...)
    # e.g. for sequential we'll store scenario_seq["seq"] -> [ (drives, wMBps, rMBps) ... ]
    scenario_seq = []
    scenario_rand= []
    scenario_meta= []

    for drives in DRIVES_LIST:
        # 1) start FUSE
        fuse_proc = mount_fuse(drives)

        # 2) run tests
        # a) sequential
        try:
            wmb, rmb = scenario_sequential()
            # b) random
            rand_mb  = scenario_random_io()
            # c) smallfiles
            create_ops, stat_ops, remove_ops = scenario_smallfiles()

            scenario_seq.append( (drives, wmb, rmb) )
            scenario_rand.append( (drives, rand_mb) )
            scenario_meta.append( (drives, create_ops, stat_ops, remove_ops) )

            print(f"[Result] drives={drives}")
            print(f"  [Seq ] Write={wmb:.2f} MB/s, Read={rmb:.2f} MB/s")
            print(f"  [Rand] Mixed RW={rand_mb:.2f} MB/s")
            print(f"  [Meta] Create={create_ops:.2f} ops/s, Stat={stat_ops:.2f} ops/s, Remove={remove_ops:.2f} ops/s")

        finally:
            # 3) unmount
            unmount_fuse(fuse_proc)
            time.sleep(1)

    # -------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------
    # Seq throughput
    fig1 = plt.figure(figsize=(10,6))
    ax1  = fig1.add_subplot(111)
    x_vals = [row[0] for row in scenario_seq]
    wvals  = [row[1] for row in scenario_seq]
    rvals  = [row[2] for row in scenario_seq]
    ax1.plot(x_vals, wvals, 'bo-', label='Seq Write MB/s')
    ax1.plot(x_vals, rvals, 'ro-', label='Seq Read MB/s')
    ax1.set_title('Sequential R/W Throughput vs. # Drives')
    ax1.set_xlabel('# Drives')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.grid(True)
    ax1.legend()
    fig1.savefig('seq_performance.png')

    # Random
    fig2 = plt.figure(figsize=(10,6))
    ax2  = fig2.add_subplot(111)
    x_vals = [row[0] for row in scenario_rand]
    randvals = [row[1] for row in scenario_rand]
    ax2.plot(x_vals, randvals, 'go-', label='Random IO MB/s')
    ax2.set_title('Random R/W Throughput vs. # Drives')
    ax2.set_xlabel('# Drives')
    ax2.set_ylabel('Throughput (MB/s)')
    ax2.grid(True)
    ax2.legend()
    fig2.savefig('rand_performance.png')

    # Metadata
    fig3 = plt.figure(figsize=(10,6))
    ax3  = fig3.add_subplot(111)
    x_vals = [row[0] for row in scenario_meta]
    create_vals= [row[1] for row in scenario_meta]
    stat_vals  = [row[2] for row in scenario_meta]
    remove_vals= [row[3] for row in scenario_meta]
    ax3.plot(x_vals, create_vals, 'b^-', label='Create ops/s')
    ax3.plot(x_vals, stat_vals,   'r^-', label='Stat   ops/s')
    ax3.plot(x_vals, remove_vals, 'g^-', label='Remove ops/s')
    ax3.set_title('Small-file Metadata ops vs. # Drives')
    ax3.set_xlabel('# Drives')
    ax3.set_ylabel('Ops/s')
    ax3.grid(True)
    ax3.legend()
    fig3.savefig('meta_performance.png')

    plt.show()


if __name__=="__main__":
    main()
