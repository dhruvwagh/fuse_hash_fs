#!/usr/bin/env python3
"""
Optimized Simulation of a Scalable Storage System with Fixed-Length Queues and Backpressure

This simulation represents a system that shows up as one logical drive.
Write operations are intercepted and distributed to multiple simulated drives
using a switch function. Each simulated drive is represented by a worker thread
that batches incoming writes into fixed-size blocks and "processes" each block
by sleeping for a specified delay (to simulate drive latency).

Key improvements:
1. Fixed-length queues for each drive with a configurable maximum size
2. Workload generator that respects backpressure (slows down when queues are filling up)
3. Detailed statistics tracking for queue utilization and backpressure events

After running the workload for a given duration, the script waits until all drive queues 
are empty (or a maximum wait time is reached) and then computes the throughput.
Finally, it plots throughput (MB/s) versus the number of drives.

Usage:
    python3 simulate_storage_optimized.py --drive-list 1,2,4,8 --duration 10 --rate 1000 
    --block-size 4096 --drive-delay 0.01 --queue-size 100
"""

import time
import threading
import queue
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# Global simulation parameters (defaults)
BLOCK_SIZE = 4096
OPS_RATE = 1000          # Total write operations per second
WORKLOAD_DURATION = 10   # Duration in seconds
DRIVE_DELAY = 0.01       # Simulated processing delay per block (seconds)
QUEUE_SIZE = 100         # Maximum queue size per drive
BACKOFF_TIME = 0.001     # Time to wait when facing backpressure (seconds)

# Global counter using itertools.count for a thread-safe block number generation.
global_counter = itertools.count(start=0)

# Lock for thread-safe queue puts that may fail
queue_lock = threading.Lock()

class Drive:
    """Simulated drive that processes block writes from its queue."""
    def __init__(self, drive_id, drive_delay, queue_size):
        self.drive_id = drive_id
        self.drive_delay = drive_delay
        self.queue = queue.Queue(maxsize=queue_size)  # Fixed-size queue
        self.total_bytes = 0
        self.total_blocks_processed = 0
        self.backpressure_events = 0
        self.queue_utilization = deque(maxlen=1000)  # Track queue usage over time
        self.worker = threading.Thread(target=self.worker_func, daemon=True)
        self.worker.start()
        self.start_time = time.time()

    def worker_func(self):
        buffer = b""
        while True:
            try:
                # Record queue utilization
                self.queue_utilization.append(self.queue.qsize())
                
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
            # Mark the block as done (each get is matched with a task_done)
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
        self.total_blocks_processed += 1

    def get_stats(self):
        elapsed = time.time() - self.start_time
        stats = {
            "drive_id": self.drive_id,
            "total_bytes": self.total_bytes,
            "total_blocks": self.total_blocks_processed,
            "backpressure_events": self.backpressure_events,
            "avg_queue_utilization": np.mean(self.queue_utilization) if self.queue_utilization else 0,
            "max_queue_utilization": max(self.queue_utilization) if self.queue_utilization else 0,
            "throughput_MBps": (self.total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        }
        return stats

# Global dictionary to hold drive instances.
drives = {}
total_backpressure_events = 0

def init_drives(num_drives, drive_delay, queue_size):
    """Initialize drives dictionary with the given number of drives."""
    global drives, total_backpressure_events
    drives = {i: Drive(i, drive_delay, queue_size) for i in range(num_drives)}
    total_backpressure_events = 0

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
    Simulate a write operation with backpressure handling:
      - Get the next block number from the global counter.
      - Compute the file offset.
      - Determine the drive index using the switch function.
      - Try to enqueue a block of data to that drive.
      - Return True if write succeeded, False if it experienced backpressure.
    """
    global total_backpressure_events
    
    block_number = next(global_counter)
    offset = block_number * BLOCK_SIZE
    drive_index = switch_drive(offset, num_drives)
    data = b'A' * BLOCK_SIZE
    
    # Thread-safe queue put with backpressure awareness
    with queue_lock:
        try:
            # Non-blocking put to detect if queue is full
            drives[drive_index].queue.put_nowait(data)
            return True
        except queue.Full:
            # If the queue is full, record backpressure and return False
            drives[drive_index].backpressure_events += 1
            total_backpressure_events += 1
            return False

def workload_generator(duration, rate, num_drives):
    """
    Generate write operations for the specified duration at the given rate,
    respecting backpressure from drive queues.
    Returns a tuple of (op_count, successful_ops, backpressure_events).
    """
    end_time = time.time() + duration
    op_count = 0
    successful_ops = 0
    interval = 1.0 / rate
    
    backpressure_count = 0
    last_status_time = time.time()
    status_interval = 1.0  # Print status every second
    
    while time.time() < end_time:
        # Try to generate a write
        success = generate_write(num_drives)
        op_count += 1
        
        if success:
            successful_ops += 1
            time.sleep(interval)  # Normal rate-limited sleep
        else:
            backpressure_count += 1
            time.sleep(BACKOFF_TIME)  # Shorter sleep during backpressure
        
        # Periodically print status
        current_time = time.time()
        if current_time - last_status_time >= status_interval:
            queue_sizes = {d_id: d.queue.qsize() for d_id, d in drives.items()}
            print(f"Progress: {current_time - (end_time - duration):.1f}s/{duration}s | "
                  f"Ops: {op_count} | Backpressure: {backpressure_count} | "
                  f"Queue sizes: {queue_sizes}")
            last_status_time = current_time
    
    return op_count, successful_ops, backpressure_count

def wait_for_queues(max_wait=30):
    """
    Wait until all drive queues are empty or until max_wait seconds have passed.
    Prints debug information periodically.
    """
    start = time.time()
    while time.time() - start < max_wait:
        all_empty = all(d.queue.empty() for d in drives.values())
        if all_empty:
            print("All queues empty. Processing complete.")
            return True
        
        # Print current queue lengths for debugging.
        queue_lengths = {d.drive_id: d.queue.qsize() for d in drives.values()}
        print(f"Waiting for queues to empty: {queue_lengths}")
        time.sleep(1)
    
    print("Warning: Not all queues emptied within max wait time.")
    return False

def run_simulation(num_drives, duration, rate, block_size, drive_delay, queue_size):
    """
    Run the simulation for a given configuration:
      - Initialize global parameters.
      - Initialize drives with fixed-size queues.
      - Generate workload for the specified duration with backpressure awareness.
      - Wait until all drive queues are empty.
      - Compute and return throughput and statistics.
    """
    global BLOCK_SIZE, OPS_RATE, WORKLOAD_DURATION, DRIVE_DELAY, QUEUE_SIZE
    BLOCK_SIZE = block_size
    OPS_RATE = rate
    WORKLOAD_DURATION = duration
    DRIVE_DELAY = drive_delay
    QUEUE_SIZE = queue_size

    print(f"\nRunning simulation with {num_drives} drives:")
    print(f"Duration: {duration}s, Target ops rate: {rate} ops/s, Block size: {block_size} bytes")
    print(f"Drive delay: {drive_delay}s, Queue size: {queue_size} blocks")
    
    init_drives(num_drives, drive_delay, queue_size)
    reset_global_counter()

    start_time = time.time()
    op_count, successful_ops, backpressure_events = workload_generator(duration, rate, num_drives)
    elapsed = time.time() - start_time

    # Wait until all drive queues are empty (with a maximum wait)
    wait_for_queues(max_wait=30)

    # Collect drive statistics
    drive_stats = {d_id: d.get_stats() for d_id, d in drives.items()}
    
    total_bytes = sum(d.total_bytes for d in drives.values())
    throughput = total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0
    
    # Calculate overall statistics
    backpressure_rate = backpressure_events / op_count if op_count > 0 else 0
    completion_rate = successful_ops / op_count if op_count > 0 else 0
    actual_rate = successful_ops / elapsed if elapsed > 0 else 0

    print(f"\nSimulation Results:")
    print(f"Total operations attempted: {op_count}, Successful: {successful_ops} ({completion_rate:.2%})")
    print(f"Backpressure events: {backpressure_events} ({backpressure_rate:.2%})")
    print(f"Elapsed time: {elapsed:.2f}s, Target rate: {rate} ops/s, Actual rate: {actual_rate:.2f} ops/s")
    print(f"Total bytes processed: {total_bytes}, Throughput: {throughput:.2f} MB/s")
    
    # Print per-drive statistics
    print("\nPer-drive Statistics:")
    for d_id, stats in drive_stats.items():
        print(f"Drive {d_id}: Processed {stats['total_blocks']} blocks, "
              f"Backpressure: {stats['backpressure_events']}, "
              f"Avg queue: {stats['avg_queue_utilization']:.1f}, "
              f"Max queue: {stats['max_queue_utilization']}")
    
    return {
        "throughput": throughput,
        "backpressure_rate": backpressure_rate,
        "completion_rate": completion_rate,
        "actual_rate": actual_rate,
        "drive_stats": drive_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Optimized Simulated Storage System with Fixed Queues and Backpressure")
    parser.add_argument("--drive-list", type=str, default="1,2,4,8",
                        help="Comma-separated list of drive counts to test (e.g. 1,2,4,8)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration (in seconds) for each simulation")
    parser.add_argument("--rate", type=int, default=1000,
                        help="Target write operations per second")
    parser.add_argument("--block-size", type=int, default=4096,
                        help="Block size in bytes")
    parser.add_argument("--drive-delay", type=float, default=0.01,
                        help="Simulated processing delay per block (seconds)")
    parser.add_argument("--queue-size", type=int, default=100,
                        help="Maximum queue size for each drive (in blocks)")
    parser.add_argument("--backoff-time", type=float, default=0.001,
                        help="Time to wait when experiencing backpressure (seconds)")
    args = parser.parse_args()

    global BACKOFF_TIME
    BACKOFF_TIME = args.backoff_time
    
    drive_list = [int(x.strip()) for x in args.drive_list.split(",")]
    results = {}

    # Run simulations for each drive count
    for nd in drive_list:
        results[nd] = run_simulation(
            num_drives=nd,
            duration=args.duration,
            rate=args.rate,
            block_size=args.block_size,
            drive_delay=args.drive_delay,
            queue_size=args.queue_size
        )
        # Small pause between configurations
        time.sleep(2)

    # Plot results
    plot_results(results)

def plot_results(results):
    """Create plots of the simulation results."""
    drive_nums = sorted(results.keys())
    
    # Extract data for plots
    throughputs = [results[d]["throughput"] for d in drive_nums]
    backpressure_rates = [results[d]["backpressure_rate"] * 100 for d in drive_nums]
    completion_rates = [results[d]["completion_rate"] * 100 for d in drive_nums]
    actual_rates = [results[d]["actual_rate"] for d in drive_nums]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Throughput vs Number of Drives
    axs[0, 0].plot(drive_nums, throughputs, marker='o', linestyle='-', color='blue')
    axs[0, 0].set_xlabel("Number of Drives")
    axs[0, 0].set_ylabel("Throughput (MB/s)")
    axs[0, 0].set_title("Throughput vs. Number of Drives")
    axs[0, 0].grid(True)
    
    # Plot 2: Backpressure Rate vs Number of Drives
    axs[0, 1].plot(drive_nums, backpressure_rates, marker='s', linestyle='-', color='red')
    axs[0, 1].set_xlabel("Number of Drives")
    axs[0, 1].set_ylabel("Backpressure Rate (%)")
    axs[0, 1].set_title("Backpressure Rate vs. Number of Drives")
    axs[0, 1].grid(True)
    
    # Plot 3: Completion Rate vs Number of Drives
    axs[1, 0].plot(drive_nums, completion_rates, marker='^', linestyle='-', color='green')
    axs[1, 0].set_xlabel("Number of Drives")
    axs[1, 0].set_ylabel("Completion Rate (%)")
    axs[1, 0].set_title("Operation Completion Rate vs. Number of Drives")
    axs[1, 0].grid(True)
    
    # Plot 4: Actual Rate vs Number of Drives
    axs[1, 1].plot(drive_nums, actual_rates, marker='D', linestyle='-', color='purple')
    axs[1, 1].set_xlabel("Number of Drives")
    axs[1, 1].set_ylabel("Actual Write Rate (ops/s)")
    axs[1, 1].set_title("Actual Write Rate vs. Number of Drives")
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("storage_simulation_results.png")
    print("\nResults plotted and saved as 'storage_simulation_results.png'.")
    plt.show()

if __name__ == '__main__':
    main()