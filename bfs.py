#!/usr/bin/env python3
import os
import time
import random
import string

# -------------------------------------------------------------------------
# Hard-coded mountpoint. Change if desired.
# -------------------------------------------------------------------------
MOUNTPOINT = "/tmp/myfuse"

# -------------------------------------------------------------------------
# Constants for benchmarking
# -------------------------------------------------------------------------
TEST_FILENAME = "fuse_testfile.dat"
TOTAL_BYTES   = 32 * 1024 * 1024  # 32 MB
CHUNK_SIZE    = 4 * 1024         # 4 KB

def random_data(size_in_bytes):
    """Generate a string of random alphanumeric data of given size."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size_in_bytes))

def write_test(filepath, total_bytes, chunk_size):
    """
    Write 'total_bytes' of random data to 'filepath' in 'chunk_size' increments.
    Measure and print the write throughput.
    """
    data_chunk = random_data(chunk_size).encode('utf-8')
    iterations = total_bytes // chunk_size

    print(f"[WriteTest] Writing {total_bytes} bytes in {chunk_size}-byte chunks to '{filepath}'")
    start_time = time.time()
    with open(filepath, 'wb') as f:
        for _ in range(iterations):
            f.write(data_chunk)
    end_time = time.time()

    elapsed = end_time - start_time
    mb = total_bytes / (1024 * 1024)
    throughput = mb / elapsed if elapsed > 0 else 0
    print(f"[WriteTest] Wrote {mb:.2f} MB in {elapsed:.2f} s => {throughput:.2f} MB/s\n")

def read_test(filepath, chunk_size):
    """
    Read the entire file in 'chunk_size' increments.
    Measure and print the read throughput.
    """
    file_size = os.path.getsize(filepath)
    print(f"[ReadTest] Reading {file_size} bytes in {chunk_size}-byte chunks from '{filepath}'")

    start_time = time.time()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
    end_time = time.time()

    elapsed = end_time - start_time
    mb = file_size / (1024 * 1024)
    throughput = mb / elapsed if elapsed > 0 else 0
    print(f"[ReadTest] Read {mb:.2f} MB in {elapsed:.2f} s => {throughput:.2f} MB/s\n")

def main():
    # Construct the full path for the test file
    filepath = os.path.join(MOUNTPOINT, TEST_FILENAME)

    # Perform write test
    write_test(filepath, TOTAL_BYTES, CHUNK_SIZE)
    # Perform read test
    read_test(filepath, CHUNK_SIZE)

    # Clean up
    os.remove(filepath)
    print("[Info] Benchmark complete. Test file removed.")

if __name__ == "__main__":
    main()
