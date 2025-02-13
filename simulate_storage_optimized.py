#!/usr/bin/env python3
"""
Optimized Simulation of a Scalable Storage System with Debugging

This simulation represents a system that shows up as one logical drive.
Write operations are intercepted and distributed to multiple simulated drives
using a switch function. Each simulated drive is represented by a worker thread
that batches incoming writes into fixed-size blocks and “processes” each block
by sleeping for a specified delay (to simulate drive latency).

After running the workload for a given duration, the script waits (with debug
logging) until all drive queues are empty (or a maximum wait time is reached) and
then computes the throughput. Finally, it plots throughput (MB/s) versus the number
of drives.

Usage:
    python3 simulate_storage_optimized.py --drive-list 1,2,4,8 --duration 10 --rate 1000 --block-size 4096 --drive-delay 0.01
"""

import time
import threading
import queue
import argparse
import itertools
import matplotlib.pyplot as plt

# Global simulation parameters (defaults)
BLOCK_SIZE = 4096
OPS_RATE = 1000000          # Total write operations per second
WORKLOAD_DURATION = 10   # Duration in seconds
DRIVE_DELAY = 0.01       # Simulated processing delay per block (seconds)

# Global counter using itertools.count for a thread-safe block number generation.
global_counter = itertools.count(start=0)

class Drive:
    """Simulated drive that processes block writes from its queue."""
    def __init__(self, drive_id, drive_delay):
        self.drive_id = drive_id
        self.drive_delay = drive_delay
        self.queue = queue.Queue()
        self.total_bytes = 0
        self.worker = threading.Thread(target=self.worker_func, daemon=True)
        self.worker.start()

    def worker_func(self):
        buffer = b""
        while True:
            try:
                # Wait for a block from the queue (each block is a bytes object)
                block = self.queue.get(timeout=1)
            except queue.Empty:
                # If queue is empty and buffer has leftover data, flush it.
                if buffer:
                    self.process_block(buffer)
                    buffer = b""
                continue

            # Add the incoming block to the buffer.
            buffer += block
            # Mark the block as done (each put is matched with a task_done)
            self.queue.task_done()

            # Process complete blocks from the buffer.
            while len(buffer) >= BLOCK_SIZE:
                chunk = buffer[:BLOCK_SIZE]
                self.process_block(chunk)
                buffer = buffer[BLOCK_SIZE:]
    
    def process_block(self, chunk):
        # Simulate drive processing time per block.
        time.sleep(self.drive_delay)
        self.total_bytes += len(chunk)

# Global dictionary to hold drive instances.
drives = {}

def init_drives(num_drives, drive_delay):
    """Initialize drives dictionary with the given number of drives."""
    global drives
    drives = {i: Drive(i, drive_delay) for i in range(num_drives)}

def reset_global_counter():
    global global_counter
    global_counter = itertools.count(start=0)

def switch_drive(offset, num_drives):
    """
    Determine which drive gets a block based on the file offset.
    For block-aligned writes:
         drive = (offset // BLOCK_SIZE) % num_drives
    """
    block_index = offset // BLOCK_SIZE
    return block_index % num_drives

def generate_write(num_drives):
    """
    Simulate a write operation:
      - Get the next block number from the global counter.
      - Compute the file offset.
      - Determine the drive index using the switch function.
      - Enqueue a block of data (dummy bytes) to that drive.
    """
    block_number = next(global_counter)
    offset = block_number * BLOCK_SIZE
    drive_index = switch_drive(offset, num_drives)
    data = b'A' * BLOCK_SIZE
    drives[drive_index].queue.put(data)

def workload_generator(duration, rate, num_drives):
    """
    Generate write operations for the specified duration at the given rate.
    Each operation writes one block.
    Returns the total number of operations generated.
    """
    end_time = time.time() + duration
    op_count = 0
    interval = 1.0 / rate
    while time.time() < end_time:
        generate_write(num_drives)
        op_count += 1
        time.sleep(interval)
    return op_count

def wait_for_queues(max_wait=30):
    """
    Wait until all drive queues are empty or until max_wait seconds have passed.
    Prints debug information periodically.
    """
    start = time.time()
    while time.time() - start < max_wait:
        all_empty = all(d.queue.empty() for d in drives.values())
        if all_empty:
            return True
        # Print current queue lengths for debugging.
        queue_lengths = {d.drive_id: d.queue.qsize() for d in drives.values()}
        print(f"Waiting for queues to empty: {queue_lengths}")
        time.sleep(1)
    print("Warning: Not all queues emptied within max wait time.")
    return False

def run_simulation(num_drives, duration, rate, block_size, drive_delay):
    """
    Run the simulation for a given configuration:
      - Initialize global parameters.
      - Initialize drives.
      - Generate workload for the specified duration.
      - Wait until all drive queues are empty.
      - Compute and return throughput.
    Returns: (op_count, total_bytes, elapsed, throughput_MBps)
    """
    global BLOCK_SIZE, OPS_RATE, WORKLOAD_DURATION, DRIVE_DELAY
    BLOCK_SIZE = block_size
    OPS_RATE = rate
    WORKLOAD_DURATION = duration
    DRIVE_DELAY = drive_delay

    print(f"\nRunning simulation with {num_drives} drives:")
    print(f"Duration: {duration}s, Ops rate: {rate} ops/s, Block size: {block_size} bytes, Drive delay: {drive_delay}s")
    
    init_drives(num_drives, drive_delay)
    reset_global_counter()

    start_time = time.time()
    op_count = workload_generator(duration, rate, num_drives)
    elapsed = time.time() - start_time

    # Wait until all drive queues are empty (with a maximum wait)
    wait_for_queues(max_wait=30)

    total_bytes = sum(d.total_bytes for d in drives.values())
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0

    print(f"Total operations: {op_count}, Total bytes processed: {total_bytes}")
    print(f"Elapsed time: {elapsed:.2f} s, Throughput: {throughput:.2f} MB/s")
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Optimized Simulated Scalable Storage System with Plotting")
    parser.add_argument("--drive-list", type=str, default="1,2,4,8",
                        help="Comma-separated list of drive counts to test (e.g. 1,2,4,8)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration (in seconds) for each simulation")
    parser.add_argument("--rate", type=int, default=1000,
                        help="Write operations per second")
    parser.add_argument("--block-size", type=int, default=4096,
                        help="Block size in bytes")
    parser.add_argument("--drive-delay", type=float, default=0.01,
                        help="Simulated processing delay per block (seconds)")
    args = parser.parse_args()

    drive_list = [int(x.strip()) for x in args.drive_list.split(",")]
    throughput_results = {}

    for nd in drive_list:
        throughput = run_simulation(
            num_drives=nd,
            duration=args.duration,
            rate=args.rate,
            block_size=args.block_size,
            drive_delay=args.drive_delay
        )
        throughput_results[nd] = throughput
        # Small pause between configurations
        time.sleep(2)

    # Plot throughput vs. number of drives
    drive_nums = sorted(throughput_results.keys())
    throughputs = [throughput_results[d] for d in drive_nums]

    plt.figure(figsize=(8,6))
    plt.plot(drive_nums, throughputs, marker='o', linestyle='-')
    plt.xlabel("Number of Drives")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Simulated Storage Throughput vs. Number of Drives")
    plt.grid(True)
    plt.savefig("simulate_storage_optimized_benchmark.png")
    print("\nBenchmark complete. Graph saved as 'simulate_storage_optimized_benchmark.png'.")
    plt.show()

if __name__ == '__main__':
    main()