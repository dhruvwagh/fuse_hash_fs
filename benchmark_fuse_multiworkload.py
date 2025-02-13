#!/usr/bin/env python3
"""
FUSE In-Memory Filesystem Simulation with Striped Drive Processing

This filesystem exposes a single file, "/testfile", on a standard directory tree.
Write operations to "/testfile" are handled in two ways:
  1. The file's content is updated synchronously in memory for correct read behavior.
  2. The written data is dispatched to a simulated drive subsystem:
     - A global block counter assigns each write a block offset.
     - A switch function computes:
           drive = (offset // BLOCK_SIZE) % num_drives
     - The block data is enqueued to the corresponding drive's queue.
     
Each simulated drive is represented by a worker thread that, upon receiving a block,
sleeps for a configured delay (to simulate I/O latency) and then updates its total bytes counter.

When the filesystem is unmounted, the destroy() callback prints the elapsed time and
the effective throughput (MB/s) based on the total bytes processed by the simulated drives.

Usage:
    sudo python3 fuse_simulation.py <mountpoint> --num-drives N --drive-delay D --block-size B

Example:
    sudo python3 fuse_simulation.py /mnt/myfuse --num-drives 4 --drive-delay 0.01 --block-size 4096

Ensure that /etc/fuse.conf has "user_allow_other" uncommented.
"""

import os
import sys
import stat
import time
import logging
import argparse
import threading
import queue
import itertools
import errno

from fuse import FUSE, FuseOSError, Operations

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Global parameter for block size; can be overridden by command-line
BLOCK_SIZE = 4096

# Global atomic counter for sequential block numbers
global_block_counter = itertools.count(start=0)

def switch_drive(offset, num_drives):
    """
    Given a file offset (in bytes), compute the drive index for this block.
    For block-aligned writes:
         drive = (offset // BLOCK_SIZE) % num_drives
    """
    block_index = offset // BLOCK_SIZE
    drive_index = block_index % num_drives
    logging.debug(f"[Switch] offset={offset} (block {block_index}) -> drive {drive_index}")
    return drive_index

class Drive:
    """
    Simulated drive.
    Each drive has its own queue and a worker thread that processes blocks.
    Processing a block is simulated by sleeping for drive_delay seconds, then
    updating the drive's total_bytes counter.
    """
    def __init__(self, drive_id, drive_delay):
        self.drive_id = drive_id
        self.drive_delay = drive_delay
        self.queue = queue.Queue()
        self.total_bytes = 0
        self.worker = threading.Thread(target=self.worker_func, daemon=True)
        self.worker.start()

    def worker_func(self):
        while True:
            try:
                data = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            # Simulate I/O latency
            time.sleep(self.drive_delay)
            self.total_bytes += len(data)
            self.queue.task_done()

def init_drives(num_drives, drive_delay):
    """
    Initialize and return a dictionary of simulated drives.
    """
    return {i: Drive(i, drive_delay) for i in range(num_drives)}

# --- In-Memory Filesystem Structure ---
#
# For simplicity, we maintain a global dictionary representing a minimal file system tree.
# We expose a single file "/testfile" whose content is stored in memory.
#
# The structure is as follows:
#
#   fs = {
#       "/": {
#           "type": "dir",
#           "mode": 0o755,
#           "children": {
#               "testfile": {
#                   "type": "file",
#                   "mode": 0o644,
#                   "size": 0,
#                   "content": b""
#               }
#           }
#       }
#   }
#
fs = {
    "/": {
        "type": "dir",
        "mode": 0o755,
        "children": {
            "testfile": {
                "type": "file",
                "mode": 0o644,
                "size": 0,
                "content": b""
            }
        }
    }
}
fs_lock = threading.Lock()

def get_node(path):
    """Traverse the filesystem tree and return the node for the given path."""
    if path == "/" or path == "":
        return fs["/"]
    parts = [p for p in path.split("/") if p]
    node = fs["/"]
    for part in parts:
        if node["type"] != "dir" or part not in node["children"]:
            raise FuseOSError(errno.ENOENT)
        node = node["children"][part]
    return node

def get_parent_and_name(path):
    """Return (parent_node, basename) for a given path."""
    if path == "/" or path == "":
        raise FuseOSError(errno.EINVAL)
    parts = [p for p in path.split("/") if p]
    basename = parts[-1]
    parent = fs["/"]
    for part in parts[:-1]:
        if part not in parent["children"]:
            raise FuseOSError(errno.ENOENT)
        parent = parent["children"][part]
    return parent, basename

# --- FUSE Filesystem Implementation ---
class SimulatedFS(Operations):
    def __init__(self, num_drives, drive_delay):
        self.num_drives = num_drives
        self.drive_delay = drive_delay
        self.start_time = time.time()
        # Initialize the simulated drives
        self.drives = init_drives(num_drives, drive_delay)
        logging.info(f"SimulatedFS: {num_drives} drives, drive delay {drive_delay}s.")
    
    def getattr(self, path, fh=None):
        logging.debug(f"getattr({path})")
        with fs_lock:
            node = get_node(path)
        st = {}
        st['st_ctime'] = st['st_mtime'] = st['st_atime'] = time.time()
        st['st_uid'] = os.getuid()
        st['st_gid'] = os.getgid()
        if node["type"] == "dir":
            st['st_mode'] = stat.S_IFDIR | node["mode"]
            st['st_nlink'] = 2 + len(node["children"])
            st['st_size'] = 0
        else:
            st['st_mode'] = stat.S_IFREG | node["mode"]
            st['st_nlink'] = 1
            st['st_size'] = node["size"]
        return st

    def readdir(self, path, fh):
        logging.debug(f"readdir({path})")
        with fs_lock:
            node = get_node(path)
            if node["type"] != "dir":
                raise FuseOSError(errno.ENOTDIR)
            return [".", ".."] + list(node["children"].keys())

    def open(self, path, flags):
        logging.debug(f"open({path})")
        with fs_lock:
            node = get_node(path)
            if node["type"] != "file":
                raise FuseOSError(errno.EISDIR)
        return 0

    def read(self, path, size, offset, fh):
        logging.debug(f"read({path}, size={size}, offset={offset})")
        with fs_lock:
            node = get_node(path)
            if node["type"] != "file":
                raise FuseOSError(errno.EISDIR)
            content = node["content"]
        return content[offset:offset+size]

    def write(self, path, data, offset, fh):
        logging.debug(f"write({path}, size={len(data)}, offset={offset})")
        # Synchronously update the file's content so that reads reflect writes.
        with fs_lock:
            node = get_node(path)
            if node["type"] != "file":
                raise FuseOSError(errno.EISDIR)
            content = node["content"]
            # If offset is beyond current content, pad with zeros.
            if offset > len(content):
                content += b'\x00' * (offset - len(content))
            new_content = content[:offset] + data
            if len(content) > offset + len(data):
                new_content += content[offset+len(data):]
            node["content"] = new_content
            node["size"] = len(new_content)
        # Now, simulate asynchronous processing by dispatching the write to a drive.
        # Get a block number from the global counter.
        block_number = next(global_block_counter)
        simulated_offset = block_number * BLOCK_SIZE
        drive_index = switch_drive(simulated_offset, self.num_drives)
        logging.debug(f"Dispatching write block (size {len(data)}) to drive {drive_index}")
        self.drives[drive_index].queue.put(data)
        return len(data)

    def truncate(self, path, length):
        logging.debug(f"truncate({path}, length={length})")
        with fs_lock:
            node = get_node(path)
            if node["type"] != "file":
                raise FuseOSError(errno.EISDIR)
            node["content"] = node["content"][:length]
            node["size"] = length
        return 0

    def utimens(self, path, times=None):
        logging.debug(f"utimens({path}, times={times})")
        # For simplicity, ignore time updates.
        return 0

    def flush(self, path, fh):
        logging.debug(f"flush({path})")
        return 0

    def release(self, path, fh):
        logging.debug(f"release({path})")
        return 0

    def access(self, path, mode):
        logging.debug(f"access({path}, mode={mode})")
        with fs_lock:
            node = get_node(path)
        if node is None:
            raise FuseOSError(errno.ENOENT)
        return 0

    def statfs(self, path):
        logging.debug(f"statfs({path})")
        return {
            'f_bsize': BLOCK_SIZE,
            'f_frsize': BLOCK_SIZE,
            'f_blocks': 1024 * 1024,
            'f_bfree': 1024 * 1024,
            'f_bavail': 1024 * 1024,
            'f_files': 100000,
            'f_ffree': 100000,
            'f_favail': 100000,
            'f_flag': 0,
            'f_namemax': 255
        }

    def destroy(self, private_data):
        elapsed = time.time() - self.start_time
        total_bytes = sum(d.total_bytes for d in self.drives.values())
        throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
        print("\n=== Filesystem Unmounted ===")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Total bytes processed by simulated drives: {total_bytes}")
        print(f"Effective throughput: {throughput:.2f} MB/s")

def main():
    parser = argparse.ArgumentParser(description="FUSE In-Memory Filesystem Simulation with Striped Drives")
    parser.add_argument("mountpoint", help="Mount point for the FUSE filesystem")
    parser.add_argument("--num-drives", type=int, default=4, help="Number of simulated drives")
    parser.add_argument("--drive-delay", type=float, default=0.01, help="Simulated drive delay per block (seconds)")
    parser.add_argument("--block-size", type=int, default=4096, help="Block size in bytes")
    args = parser.parse_args()
    global BLOCK_SIZE
    BLOCK_SIZE = args.block_size
    logging.info(f"Mounting filesystem at {args.mountpoint} with {args.num_drives} drives, drive delay: {args.drive_delay}s, block size: {BLOCK_SIZE} bytes")
    FUSE(SimulatedFS(args.num_drives, args.drive_delay), args.mountpoint, foreground=True, allow_other=True)

if __name__ == '__main__':
    main()
