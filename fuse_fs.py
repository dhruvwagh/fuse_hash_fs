#!/usr/bin/env python3
"""
FUSE In-Memory Filesystem with Simulated Striped Drives

It implements standard filesystem operations: getattr, readdir, mkdir, rmdir,
create, open, read, write, truncate, unlink, rename, flush, release, access, statfs, etc.

For file writes, data is stored in memory AND “dispatched” to simulated drives:
each block (based on a simple switch function) is enqueued to one of N drive queues.
Each drive’s worker thread processes a block by sleeping for a configured delay and
then updating a counter.

Upon unmount (in destroy), the system prints total bytes processed and effective throughput.

Usage:
  sudo python3 fuse_fs.py <mountpoint> --num-drives N --drive-delay D --block-size B

Example:
  sudo python3 fuse_fs.py /mnt/myfuse --num-drives 4 --drive-delay 0.01 --block-size 4096
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

# ----------------------------------------------------------------------------------
# Global parameters (overridden via command-line)
BLOCK_SIZE = 4096

# Global atomic counter for pseudo block numbers
global_block_counter = itertools.count(start=0)

# Simulated drive class
class Drive:
    """Simulated drive that processes blocks from its queue."""
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
            time.sleep(self.drive_delay)
            self.total_bytes += len(data)
            self.queue.task_done()

def init_drives(num_drives, drive_delay):
    """Initialize a dictionary of simulated drives."""
    return {i: Drive(i, drive_delay) for i in range(num_drives)}

def switch_drive(offset, num_drives):
    """
    For block-aligned writes, determine which drive handles the block:
        drive = (offset // BLOCK_SIZE) % num_drives
    """
    block_index = offset // BLOCK_SIZE
    return block_index % num_drives

# ----------------------------------------------------------------------------------
# In-memory filesystem data
fs = {}
fs_lock = threading.Lock()

def init_fs():
    """Initialize the global FS structure: create root '/', and add '/testfile'."""
    global fs
    fs = {}
    fs["/"] = {
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

def get_node(path):
    """Traverse the FS tree to find the node for `path` or raise ENOENT."""
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
    """Return (parent_node, basename) for the given path."""
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise FuseOSError(errno.EINVAL)
    basename = parts[-1]
    parent = fs["/"]
    for part in parts[:-1]:
        if parent["type"] != "dir" or part not in parent["children"]:
            raise FuseOSError(errno.ENOENT)
        parent = parent["children"][part]
    return parent, basename

# ----------------------------------------------------------------------------------
# FUSE Operations
class MemoryFS(Operations):
    def __init__(self, num_drives, drive_delay):
        init_fs()
        self.num_drives = num_drives
        self.drive_delay = drive_delay
        self.start_time = time.time()
        self.file_locks = {"/testfile": threading.Lock()}  # Lock for our testfile
        # Initialize simulated drives
        self.drives = init_drives(num_drives, drive_delay)
        logging.info(f"FS initialized with {num_drives} drives, drive delay: {drive_delay}s.")

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

    def mkdir(self, path, mode):
        logging.debug(f"mkdir({path})")
        parent, name = get_parent_and_name(path)
        with fs_lock:
            if name in parent["children"]:
                raise FuseOSError(errno.EEXIST)
            parent["children"][name] = {"type": "dir", "mode": mode, "children": {}}
        return 0

    def rmdir(self, path):
        logging.debug(f"rmdir({path})")
        parent, name = get_parent_and_name(path)
        with fs_lock:
            if name not in parent["children"]:
                raise FuseOSError(errno.ENOENT)
            node = parent["children"][name]
            if node["type"] != "dir":
                raise FuseOSError(errno.ENOTDIR)
            if node["children"]:
                raise FuseOSError(errno.ENOTEMPTY)
            del parent["children"][name]
        return 0

    def mknod(self, path, mode, dev):
        logging.debug(f"mknod({path})")
        return self.create(path, mode)

    def create(self, path, mode, fi=None):
        logging.debug(f"create({path})")
        parent, name = get_parent_and_name(path)
        with fs_lock:
            if name in parent["children"]:
                raise FuseOSError(errno.EEXIST)
            parent["children"][name] = {"type": "file", "mode": mode, "size": 0, "content": b""}
        self.file_locks[path] = threading.Lock()
        return 0

    def unlink(self, path):
        logging.debug(f"unlink({path})")
        parent, name = get_parent_and_name(path)
        with fs_lock:
            if name not in parent["children"]:
                raise FuseOSError(errno.ENOENT)
            node = parent["children"][name]
            if node["type"] != "file":
                raise FuseOSError(errno.EISDIR)
            del parent["children"][name]
        return 0

    def rename(self, old, new):
        logging.debug(f"rename({old}, {new})")
        old_parent, old_name = get_parent_and_name(old)
        new_parent, new_name = get_parent_and_name(new)
        with fs_lock:
            if old_name not in old_parent["children"]:
                raise FuseOSError(errno.ENOENT)
            node = old_parent["children"][old_name]
            new_parent["children"][new_name] = node
            del old_parent["children"][old_name]
        return 0

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
        lock = self.file_locks.get(path)
        if lock:
            lock.acquire()
        try:
            with fs_lock:
                node = get_node(path)
                if node["type"] != "file":
                    raise FuseOSError(errno.EISDIR)
                content = node["content"]
                if offset > len(content):
                    content += b'\x00' * (offset - len(content))
                new_content = content[:offset] + data
                if len(content) > offset + len(data):
                    new_content += content[offset+len(data):]
                node["content"] = new_content
                node["size"] = len(new_content)
        finally:
            if lock:
                lock.release()

        # Simulate asynchronous block processing
        block_number = next(global_block_counter)
        simulated_offset = block_number * BLOCK_SIZE
        drive_index = switch_drive(simulated_offset, self.num_drives)
        self.drives[drive_index].queue.put(data)
        return len(data)

    def truncate(self, path, length):
        logging.debug(f"truncate({path}, length={length})")
        lock = self.file_locks.get(path)
        if lock:
            lock.acquire()
        try:
            with fs_lock:
                node = get_node(path)
                if node["type"] != "file":
                    raise FuseOSError(errno.EISDIR)
                node["content"] = node["content"][:length]
                node["size"] = length
        finally:
            if lock:
                lock.release()
        return 0

    def utimens(self, path, times=None):
        logging.debug(f"utimens({path}, times={times})")
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
        print("\n=== Filesystem Destroyed ===")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Total bytes processed by simulated drives: {total_bytes}")
        print(f"Effective throughput: {throughput:.2f} MB/s")

def main():
    parser = argparse.ArgumentParser(
        description="FUSE In-Memory Filesystem with Striped Drives"
    )
    parser.add_argument("mountpoint", help="Mount point for the FUSE filesystem")
    parser.add_argument("--num-drives", type=int, default=4, help="Number of simulated drives")
    parser.add_argument("--drive-delay", type=float, default=0.01, help="Simulated processing delay per block (seconds)")
    parser.add_argument("--block-size", type=int, default=4096, help="Block size in bytes")
    args = parser.parse_args()

    global BLOCK_SIZE
    BLOCK_SIZE = args.block_size

    logging.info(
        f"Mounting filesystem at {args.mountpoint} with {args.num_drives} drives, "
        f"drive delay: {args.drive_delay}s, block size: {BLOCK_SIZE} bytes"
    )

    from fuse import FUSE
    FUSE(MemoryFS(args.num_drives, args.drive_delay), args.mountpoint, foreground=True, allow_other=True, debug= True)

if __name__ == "__main__":
    main()
