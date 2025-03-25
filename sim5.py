#!/usr/bin/env python3
"""
Scalable Storage System Simulation with Realistic Workloads and Variable Drive Speeds

This modified simulation models a storage system with multiple drives having different performance
characteristics, fixed-length queues, and support for various realistic workload patterns:
- Sequential reads
- Sequential writes
- Random reads
- Random writes
- Mixed sequential read-write
- Mixed random read-write

Features:
1. Variable read/write latencies based on drive type and workload
2. Per-drive performance characteristics
3. Fixed-length queues with backpressure
4. Accurate read completion time measurement after drive processing
5. Parallel processing of I/O requests
6. Detailed performance metrics and statistics
7. Visualization of results across different workload patterns

Usage:
    python3 modified_storage_simulation.py --workloads all --drive-counts 1,2,4,8,16
    --duration 10 --rate 1000 --block-size 4096 --queue-size 100
"""

import time
import threading
import queue
import argparse
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from enum import Enum
import json

# Define operation types
class OpType(Enum):
    READ = 0
    WRITE = 1

# Drive type latency profiles (seconds)
DRIVE_LATENCY_PROFILES = {
    # Workload type: (min_read, max_read, min_write, max_write)
    "sequential_read":    (0.0005, 0.002, 0.005, 0.02),
    "sequential_write":   (0.0005, 0.002, 0.005, 0.02),
    "random_read":        (0.0005, 0.001, 0.005, 0.02),
    "random_write":       (0.0005, 0.001, 0.005, 0.02),
    "sequential_mixed":   (0.002, 0.005, 0.005, 0.03),
    "random_mixed":       (0.002, 0.005, 0.005, 0.03)
}

# Global simulation parameters (defaults)
BLOCK_SIZE = 4096
OPS_RATE = 10000          # Total operations per second
WORKLOAD_DURATION = 10   # Duration in seconds
QUEUE_SIZE = 100         # Maximum queue size per drive
BACKOFF_TIME = 0.001     # Time to wait when facing backpressure (seconds)
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB simulated file size

# Global counter for operation IDs
global_counter = itertools.count(start=0)

# Lock for thread-safe queue puts
queue_lock = threading.Lock()

class Operation:
    """Represents a read or write operation to be performed on a drive."""
    def __init__(self, op_id, op_type, offset, size=None):
        self.op_id = op_id
        self.op_type = op_type
        self.offset = offset
        self.size = size or BLOCK_SIZE
        # For write operations, include data (simulated)
        if op_type == OpType.WRITE:
            self.data = b'A' * self.size
        else:
            self.data = None
        self.timestamp = time.time()
        self.queue_time = None  # When it was dequeued
        self.completion_time = None  # When it was actually completed

class Drive:
    """Simulated drive that processes read/write operations from its queue."""
    def __init__(self, drive_id, workload_type, queue_size):
        self.drive_id = drive_id
        self.workload_type = workload_type
        self.queue = queue.Queue(maxsize=queue_size)  # Fixed-size queue
        
        # Get latency ranges for this drive based on workload type
        min_read, max_read, min_write, max_write = DRIVE_LATENCY_PROFILES[workload_type]
        
        # Assign slightly different values to each drive within the range
        # to simulate real-world variance
        variance_factor = 0.8 + (0.4 * (drive_id % 3)) / 3  # 0.8 to 1.2
        self.read_delay = min_read + (max_read - min_read) * variance_factor
        self.write_delay = min_write + (max_write - min_write) * variance_factor
        
        # Statistics
        self.total_read_bytes = 0
        self.total_write_bytes = 0
        self.total_reads = 0
        self.total_writes = 0
        self.backpressure_events = 0
        self.read_latencies = []
        self.write_latencies = []
        self.queue_utilization = deque(maxlen=1000)  # Track queue usage over time
        self.processing_rate = deque(maxlen=1000)  # Track ops processed per second
        
        # Track recent processed operations for throughput calculation
        self.recent_read_bytes = 0
        self.recent_write_bytes = 0
        self.blocks_since_last = 0
        self.last_process_time = time.time()
        self.start_time = time.time()
        
        # Create and start the worker thread
        self.worker = threading.Thread(target=self.worker_func, daemon=True, 
                                      name=f"Drive-{drive_id}-Worker")
        self.worker.start()

    def worker_func(self):
        """Worker thread function that continuously processes operations from the queue"""
        print(f"Drive {self.drive_id} worker thread started. Latencies: Read={self.read_delay:.6f}s, Write={self.write_delay:.6f}s")
        last_rate_update = time.time()
        
        while True:
            try:
                # Record queue utilization
                current_size = self.queue.qsize()
                self.queue_utilization.append(current_size)
                
                # Wait for an operation from the queue
                # Use a shorter timeout to be more responsive
                op = self.queue.get(timeout=0.1)
                
                # Record when operation was taken from queue
                op.queue_time = time.time()
                
                # Process the operation
                self.process_operation(op)
                
                # Mark the operation as done
                self.queue.task_done()
                self.blocks_since_last += 1
                
                # Calculate and record processing rate periodically
                now = time.time()
                if now - last_rate_update >= 1.0:  # Update rate every second
                    elapsed = now - last_rate_update
                    rate = self.blocks_since_last / elapsed if elapsed > 0 else 0
                    self.processing_rate.append(rate)
                    self.blocks_since_last = 0
                    # Reset recent byte counters for throughput calculation
                    self.recent_read_bytes = 0
                    self.recent_write_bytes = 0
                    last_rate_update = now
                
            except queue.Empty:
                # Brief sleep to avoid tight loop when queue is empty
                time.sleep(0.001)
                continue
    
    def process_operation(self, op):
        """Process a read or write operation with appropriate delay"""
        # Record start time for latency calculation
        start_time = time.time()
        
        # Apply the appropriate delay based on operation type
        if op.op_type == OpType.READ:
            time.sleep(self.read_delay)
            self.total_read_bytes += op.size
            self.total_reads += 1
            self.recent_read_bytes += op.size
            # For reads, we would normally return data, but we just simulate it
            
            # Record latency
            latency = time.time() - start_time
            self.read_latencies.append(latency)
            
        else:  # WRITE
            time.sleep(self.write_delay)
            self.total_write_bytes += op.size
            self.total_writes += 1
            self.recent_write_bytes += op.size
            
            # Record latency
            latency = time.time() - start_time
            self.write_latencies.append(latency)
    def get_stats(self):
        """Return comprehensive statistics for this drive"""
        elapsed = time.time() - self.start_time
        
        # Calculate average latencies
        avg_read_latency = np.mean(self.read_latencies) if self.read_latencies else 0
        avg_write_latency = np.mean(self.write_latencies) if self.write_latencies else 0
        
        # Calculate IOPS
        read_iops = self.total_reads / elapsed if elapsed > 0 else 0
        write_iops = self.total_writes / elapsed if elapsed > 0 else 0
        
        # Calculate throughput
        read_throughput = (self.total_read_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        write_throughput = (self.total_write_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        stats = {
            "drive_id": self.drive_id,
            "read_delay": self.read_delay,
            "write_delay": self.write_delay,
            "total_read_bytes": self.total_read_bytes,
            "total_write_bytes": self.total_write_bytes,
            "total_bytes": self.total_read_bytes + self.total_write_bytes,
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_ops": self.total_reads + self.total_writes,
            "backpressure_events": self.backpressure_events,
            "avg_read_latency": avg_read_latency,
            "avg_write_latency": avg_write_latency,
            "read_iops": read_iops,
            "write_iops": write_iops,
            "total_iops": read_iops + write_iops,
            "read_throughput_MBps": read_throughput,
            "write_throughput_MBps": write_throughput,
            "total_throughput_MBps": read_throughput + write_throughput,
            "avg_queue_utilization": np.mean(self.queue_utilization) if self.queue_utilization else 0,
            "max_queue_utilization": max(self.queue_utilization) if self.queue_utilization else 0
        }
        return stats
    # Global dictionary to hold drive instances
drives = {}
total_backpressure_events = 0

def init_drives(num_drives, workload_type, queue_size):
    """Initialize drives dictionary with the given number of drives."""
    global drives, total_backpressure_events
    drives = {i: Drive(i, workload_type, queue_size) for i in range(num_drives)}
    total_backpressure_events = 0

def reset_global_counter():
    """Reset the global operation counter"""
    global global_counter
    global_counter = itertools.count(start=0)

class WorkloadPattern:
    """Base class for different workload patterns"""
    def __init__(self, name, read_ratio=0.5, sequential=True):
        self.name = name
        self.read_ratio = read_ratio  # Ratio of reads (0.0 to 1.0)
        self.sequential = sequential
        self.current_position = 0
        
    def get_next_operation(self):
        """Get the next operation based on the workload pattern"""
        op_id = next(global_counter)
        
        # Determine operation type (read or write)
        op_type = OpType.READ if random.random() < self.read_ratio else OpType.WRITE
        
        # Determine offset based on sequential/random pattern
        if self.sequential:
            offset = self.current_position
            self.current_position = (self.current_position + BLOCK_SIZE) % MAX_FILE_SIZE
        else:
            # Random position aligned to block size
            offset = random.randrange(0, MAX_FILE_SIZE, BLOCK_SIZE)
        
        return Operation(op_id, op_type, offset)
    
    def reset(self):
        """Reset the workload pattern state"""
        self.current_position = 0

# Define specific workload patterns
def create_workload_pattern(pattern_name):
    """Create a workload pattern object based on the pattern name"""
    patterns = {
        "sequential_read": WorkloadPattern("Sequential Read", read_ratio=1.0, sequential=True),
        "sequential_write": WorkloadPattern("Sequential Write", read_ratio=0.0, sequential=True),
        "random_read": WorkloadPattern("Random Read", read_ratio=1.0, sequential=False),
        "random_write": WorkloadPattern("Random Write", read_ratio=0.0, sequential=False),
        "sequential_mixed": WorkloadPattern("Sequential Read-Write", read_ratio=0.5, sequential=True),
        "random_mixed": WorkloadPattern("Random Read-Write", read_ratio=0.5, sequential=False)
    }
    
    if pattern_name in patterns:
        return patterns[pattern_name]
    else:
        raise ValueError(f"Unknown workload pattern: {pattern_name}")

def switch_drive(offset, num_drives):
    """
    Determine which drive gets an operation based on the file offset.
    For block-aligned operations:
         drive = (offset // BLOCK_SIZE) % num_drives
    """
    block_index = offset // BLOCK_SIZE
    return block_index % num_drives

def dispatch_operation(operation, num_drives):
    """
    Dispatch an operation to the appropriate drive:
      - Determine the drive index using the switch function based on offset.
      - Try to enqueue the operation to that drive.
      - Return True if operation succeeded, False if it experienced backpressure.
    """
    global total_backpressure_events
    
    drive_index = switch_drive(operation.offset, num_drives)
    
    # Thread-safe queue put with backpressure awareness
    with queue_lock:
        try:
            # Non-blocking put to detect if queue is full
            drives[drive_index].queue.put_nowait(operation)
            return True
        except queue.Full:
            # If the queue is full, record backpressure and return False
            drives[drive_index].backpressure_events += 1
            total_backpressure_events += 1
            return False

def workload_generator(duration, rate, num_drives, workload_pattern):
    """
    Generate operations for the specified duration at the given rate based on the workload pattern,
    respecting backpressure from drive queues.
    
    Returns a dictionary of statistics about the workload generation.
    """
    print(f"\nRunning {workload_pattern.name} workload...")
    
    end_time = time.time() + duration
    op_count = 0
    successful_ops = 0
    read_ops = 0
    write_ops = 0
    successful_reads = 0
    successful_writes = 0
    backpressure_count = 0
    
    interval = 0
    
    last_status_time = time.time()
    status_interval = 1.0  # Print status every second
    
    # For rate calculation
    op_start_time = time.time()
    ops_in_window = 0
    rate_window = 1.0  # Calculate actual rate over 1 second window
    
    # Reset the workload pattern
    workload_pattern.reset()
    
    while time.time() < end_time:
        next_op_time = time.time() + interval
        
        # Generate the next operation based on the workload pattern
        operation = workload_pattern.get_next_operation()
        op_count += 1
        ops_in_window += 1
        
        # Track operation type
        if operation.op_type == OpType.READ:
            read_ops += 1
        else:
            write_ops += 1
        
        # Try to dispatch the operation
        success = dispatch_operation(operation, num_drives)
        
        if success:
            successful_ops += 1
            # Track successful operation type
            if operation.op_type == OpType.READ:
                successful_reads += 1
            else:
                successful_writes += 1
        else:
            backpressure_count += 1
            # Only apply backoff time if we encountered backpressure
            time.sleep(BACKOFF_TIME)
        
        # Sleep only enough to maintain the requested rate
        # This allows drive threads to process in parallel
        sleep_time = next_op_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Periodically print status including actual achieved rate
        current_time = time.time()
        if current_time - last_status_time >= status_interval:
            # Calculate actual achieved rate
            window_duration = current_time - op_start_time
            actual_rate = ops_in_window / window_duration if window_duration > 0 else 0
            ops_in_window = 0
            op_start_time = current_time
            
            # Get queue states
            queue_sizes = {d_id: d.queue.qsize() for d_id, d in drives.items()}
            queue_utilization = {d_id: d.queue.qsize() / QUEUE_SIZE for d_id, d in drives.items()}
            
            # Print progress
            elapsed = current_time - (end_time - duration)
            print(f"Progress: {elapsed:.1f}s/{duration}s | "
                  f"Ops: {op_count} (R:{read_ops}/W:{write_ops}) | Success: {successful_ops} | "
                  f"Backpressure: {backpressure_count} | Rate: {actual_rate:.1f} ops/s")
            print(f"Queue utilization: {', '.join([f'Drive {d_id}: {util:.1%}' for d_id, util in queue_utilization.items()])}")
            last_status_time = current_time
    
    # Calculate statistics
    workload_elapsed = time.time() - (end_time - duration)
    actual_rate = successful_ops / workload_elapsed if workload_elapsed > 0 else 0
    read_ratio = read_ops / op_count if op_count > 0 else 0
    successful_ratio = successful_ops / op_count if op_count > 0 else 0
    backpressure_ratio = backpressure_count / op_count if op_count > 0 else 0
    
    # Return comprehensive statistics
    stats = {
        "workload_type": workload_pattern.name,
        "total_ops": op_count,
        "successful_ops": successful_ops,
        "read_ops": read_ops,
        "write_ops": write_ops,
        "successful_reads": successful_reads,
        "successful_writes": successful_writes,
        "backpressure_events": backpressure_count,
        "workload_elapsed": workload_elapsed,
        "actual_rate": actual_rate,
        "read_ratio": read_ratio,
        "successful_ratio": successful_ratio,
        "backpressure_ratio": backpressure_ratio
    }
    
    return stats

def wait_for_queues(max_wait=30):
    """
    Wait until all drive queues are empty or until max_wait seconds have passed.
    Prints debug information periodically.
    
    Returns True if all queues emptied, False if timed out.
    """
    start = time.time()
    last_progress_time = start
    last_total_size = float('inf')
    
    while time.time() - start < max_wait:
        # Get current queue sizes
        queue_sizes = {d_id: d.queue.qsize() for d_id, d in drives.items()}
        total_size = sum(queue_sizes.values())
        
        # Check if all queues are empty
        all_empty = total_size == 0
        if all_empty:
            elapsed = time.time() - start
            print(f"All queues empty after {elapsed:.2f}s. Processing complete.")
            return True
        
        # Calculate drain rate
        current_time = time.time()
        elapsed_since_last = current_time - last_progress_time
        if elapsed_since_last >= 1.0:
            drain_rate = (last_total_size - total_size) / elapsed_since_last if elapsed_since_last > 0 else 0
            last_total_size = total_size
            last_progress_time = current_time
            
            # Estimate time to empty
            time_to_empty = total_size / drain_rate if drain_rate > 0 else float('inf')
            time_to_empty_str = f"{time_to_empty:.1f}s" if time_to_empty < float('inf') else "unknown"
            
            # Print status with drain rate and ETA
            print(f"Waiting for queues to empty - Total size: {total_size} ops, " 
                  f"Drain rate: {drain_rate:.1f} ops/s, "
                  f"ETA: {time_to_empty_str}")
            print(f"Queue sizes: {queue_sizes}")
        
        # Small sleep to avoid spinning
        time.sleep(0.1)
    
    # If we get here, we timed out
    remaining_queue_sizes = {d_id: d.queue.qsize() for d_id, d in drives.items()}
    total_remaining = sum(remaining_queue_sizes.values())
    print(f"Warning: Not all queues emptied within {max_wait}s wait time.")
    print(f"Remaining items in queues: {total_remaining} ({remaining_queue_sizes})")
    return False
def run_workload_simulation(num_drives, duration, rate, block_size, queue_size, workload_pattern_name):
    """
    Run the simulation for a specific workload pattern:
      - Initialize global parameters.
      - Initialize drives with fixed-size queues.
      - Generate workload for the specified duration with backpressure awareness.
      - Wait until all drive queues are empty.
      - Compute and return throughput and statistics.
    """
    global BLOCK_SIZE, OPS_RATE, WORKLOAD_DURATION, QUEUE_SIZE
    BLOCK_SIZE = block_size
    OPS_RATE = rate
    WORKLOAD_DURATION = duration
    QUEUE_SIZE = queue_size

    # Create the workload pattern
    workload_pattern = create_workload_pattern(workload_pattern_name)

    print(f"\nRunning {workload_pattern.name} simulation with {num_drives} drives:")
    print(f"Duration: {duration}s, Target ops rate: {rate} ops/s, Block size: {block_size} bytes")
    print(f"Queue size: {queue_size} ops")
    
    # Initialize the drives with workload-specific latencies
    init_drives(num_drives, workload_pattern_name, queue_size)
    reset_global_counter()
    
    # Print drive latency information
    for d_id, drive in drives.items():
        print(f"Drive {d_id}: Read latency={drive.read_delay:.6f}s, Write latency={drive.write_delay:.6f}s")
    
    # Calculate theoretical maximum processing rates for each drive
    max_read_rates = [1.0 / d.read_delay if d.read_delay > 0 else float('inf') for d in drives.values()]
    max_write_rates = [1.0 / d.write_delay if d.write_delay > 0 else float('inf') for d in drives.values()]
    
    # Calculate average max rates
    avg_max_read_rate = np.mean(max_read_rates)
    avg_max_write_rate = np.mean(max_write_rates)
    
    # Estimate theoretical max system throughput based on workload read ratio
    read_ratio = 1.0 if workload_pattern_name.startswith("sequential_read") or workload_pattern_name.startswith("random_read") else \
                0.0 if workload_pattern_name.startswith("sequential_write") or workload_pattern_name.startswith("random_write") else \
                0.5  # Mixed workloads
    
    # Calculate weighted average of max rates based on read/write ratio
    total_max_read_rate = sum(max_read_rates)
    total_max_write_rate = sum(max_write_rates)
    effective_max_rate = read_ratio * total_max_read_rate + (1 - read_ratio) * total_max_write_rate
    
    print(f"Theoretical max read rate: {avg_max_read_rate:.1f} ops/s per drive, {total_max_read_rate:.1f} ops/s total")
    print(f"Theoretical max write rate: {avg_max_write_rate:.1f} ops/s per drive, {total_max_write_rate:.1f} ops/s total")
    print(f"Theoretical max system throughput: {effective_max_rate:.1f} ops/s total (for {read_ratio:.0%} read workload)")
    
    if rate > effective_max_rate:
        print(f"WARNING: Requested rate ({rate} ops/s) exceeds theoretical maximum ({effective_max_rate:.1f} ops/s)")
        print(f"Expect significant backpressure and queue saturation")
    
    # Small delay to ensure all drive threads are running
    time.sleep(0.1)
    
    # Run the workload
    print(f"\nStarting {workload_pattern.name} workload for {duration} seconds...")
    start_time = time.time()
    
    # Generate the workload
    workload_stats = workload_generator(duration, rate, num_drives, workload_pattern)
    workload_end_time = time.time()
    workload_elapsed = workload_end_time - start_time
    
    print(f"\n{workload_pattern.name} workload completed in {workload_elapsed:.2f}s")
    print(f"Waiting for queues to drain...")
    
    # Wait until all drive queues are empty (with a maximum wait)
    queues_drained = wait_for_queues(max_wait=30)
    total_elapsed = time.time() - start_time
    
    # Collect drive statistics
    drive_stats = {d_id: d.get_stats() for d_id, d in drives.items()}
    
    # Calculate overall system statistics
    total_read_bytes = sum(d.total_read_bytes for d in drives.values())
    total_write_bytes = sum(d.total_write_bytes for d in drives.values())
    total_bytes = total_read_bytes + total_write_bytes
    
    total_reads = sum(d.total_reads for d in drives.values())
    total_writes = sum(d.total_writes for d in drives.values())
    
    # Calculate throughput
    read_throughput_workload = total_read_bytes / workload_elapsed / (1024 * 1024) if workload_elapsed > 0 else 0
    write_throughput_workload = total_write_bytes / workload_elapsed / (1024 * 1024) if workload_elapsed > 0 else 0
    total_throughput_workload = (total_read_bytes + total_write_bytes) / workload_elapsed / (1024 * 1024) if workload_elapsed > 0 else 0
    
    read_throughput_total = total_read_bytes / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0
    write_throughput_total = total_write_bytes / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0
    total_throughput_total = (total_read_bytes + total_write_bytes) / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0
    
    # Calculate IOPS
    read_iops_workload = total_reads / workload_elapsed if workload_elapsed > 0 else 0
    write_iops_workload = total_writes / workload_elapsed if workload_elapsed > 0 else 0
    total_iops_workload = (total_reads + total_writes) / workload_elapsed if workload_elapsed > 0 else 0
    
    read_iops_total = total_reads / total_elapsed if total_elapsed > 0 else 0
    write_iops_total = total_writes / total_elapsed if total_elapsed > 0 else 0
    total_iops_total = (total_reads + total_writes) / total_elapsed if total_elapsed > 0 else 0
    
    # Calculate average latencies across all drives
    all_read_latencies = []
    all_write_latencies = []
    
    for drive in drives.values():
        all_read_latencies.extend(drive.read_latencies)
        all_write_latencies.extend(drive.write_latencies)
    
    avg_read_latency = np.mean(all_read_latencies) if all_read_latencies else 0
    avg_write_latency = np.mean(all_write_latencies) if all_write_latencies else 0
    
    # Calculate average processing rates
    avg_processing_rates = [np.mean(d.processing_rate) if d.processing_rate else 0 
                          for d in drives.values()]
    system_processing_rate = sum(avg_processing_rates)

    # Print results
    print(f"\n{workload_pattern.name} Simulation Results:")
    print(f"Total operations attempted: {workload_stats['total_ops']}, "
          f"Successful: {workload_stats['successful_ops']} ({workload_stats['successful_ratio']:.2%})")
    print(f"Read ops: {workload_stats['read_ops']} ({workload_stats['read_ratio']:.2%}), "
          f"Write ops: {workload_stats['write_ops']} ({1-workload_stats['read_ratio']:.2%})")
    print(f"Backpressure events: {workload_stats['backpressure_events']} "
          f"({workload_stats['backpressure_ratio']:.2%})")
    print(f"Workload elapsed time: {workload_elapsed:.2f}s, Total elapsed time: {total_elapsed:.2f}s")
    
    print(f"\nPerformance Metrics:")
    print(f"Read Throughput: {read_throughput_workload:.2f} MB/s during workload, "
          f"{read_throughput_total:.2f} MB/s overall")
    print(f"Write Throughput: {write_throughput_workload:.2f} MB/s during workload, "
          f"{write_throughput_total:.2f} MB/s overall")
    print(f"Total Throughput: {total_throughput_workload:.2f} MB/s during workload, "
          f"{total_throughput_total:.2f} MB/s overall")
    
    print(f"Read IOPS: {read_iops_workload:.2f} during workload, {read_iops_total:.2f} overall")
    print(f"Write IOPS: {write_iops_workload:.2f} during workload, {write_iops_total:.2f} overall")
    print(f"Total IOPS: {total_iops_workload:.2f} during workload, {total_iops_total:.2f} overall")
    
    print(f"Average Read Latency: {avg_read_latency*1000:.2f} ms")
    print(f"Average Write Latency: {avg_write_latency*1000:.2f} ms")
    
    print(f"System processing rate: {system_processing_rate:.2f} ops/s")
    
    # Print per-drive statistics
    print("\nPer-drive Statistics:")
    for d_id, stats in drive_stats.items():
        avg_rate = np.mean(drives[d_id].processing_rate) if drives[d_id].processing_rate else 0
        print(f"Drive {d_id}: R:{stats['total_reads']}/W:{stats['total_writes']} ops, "
              f"Rate: {avg_rate:.1f} ops/s, "
              f"Backpressure: {stats['backpressure_events']}, "
              f"Avg queue: {stats['avg_queue_utilization']:.1f}, "
              f"Max queue: {stats['max_queue_utilization']}")
    
    # Prepare result dictionary
    result = {
        "workload_type": workload_pattern.name,
        "num_drives": num_drives,
        "workload_stats": workload_stats,
        "drive_stats": drive_stats,
        "system_stats": {
            "total_read_bytes": total_read_bytes,
            "total_write_bytes": total_write_bytes,
            "total_bytes": total_bytes,
            "total_reads": total_reads,
            "total_writes": total_writes,
            "total_ops": total_reads + total_writes,
            "workload_elapsed": workload_elapsed,
            "total_elapsed": total_elapsed,
            "read_throughput_workload": read_throughput_workload,
            "write_throughput_workload": write_throughput_workload,
            "total_throughput_workload": total_throughput_workload,
            "read_throughput_total": read_throughput_total,
            "write_throughput_total": write_throughput_total,
            "total_throughput_total": total_throughput_total,
            "read_iops_workload": read_iops_workload,
            "write_iops_workload": write_iops_workload,
            "total_iops_workload": total_iops_workload,
            "read_iops_total": read_iops_total,
            "write_iops_total": write_iops_total,
            "total_iops_total": total_iops_total,
            "avg_read_latency": avg_read_latency,
            "avg_write_latency": avg_write_latency,
            "system_processing_rate": system_processing_rate,
            "backpressure_events": total_backpressure_events
        }
    }
    
    return result
def plot_workload_comparison(results, drive_counts, workload_patterns):
    """Create plots comparing different workloads across drive counts"""
    # Set up the figure with multiple pages of plots
    plot_dir = "simulation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Group results by drive count
    drive_results = {}
    for drive_count in drive_counts:
        drive_results[drive_count] = {}
        for workload in workload_patterns:
            result_key = f"{workload}_{drive_count}"
            if result_key in results:
                drive_results[drive_count][workload] = results[result_key]
    
    # Plot 1: Throughput comparison across all workloads by drive count
    plt.figure(figsize=(14, 10))
    
    # Group workloads by type for better visualization
    read_workloads = [w for w in workload_patterns if w.endswith('read')]
    write_workloads = [w for w in workload_patterns if w.endswith('write')]
    mixed_workloads = [w for w in workload_patterns if w.endswith('mixed')]
    
    # Set up bar positions
    bar_width = 0.15
    x = np.arange(len(drive_counts))
    
    # Color scheme
    colors = {
        'sequential_read': 'royalblue',
        'random_read': 'lightskyblue',
        'sequential_write': 'crimson',
        'random_write': 'lightcoral',
        'sequential_mixed': 'forestgreen',
        'random_mixed': 'lightgreen'
    }
    
    # Plot read workloads throughput
    for i, workload in enumerate(read_workloads):
        values = [drive_results[d][workload]['system_stats']['total_throughput_workload'] 
                 if workload in drive_results[d] else 0 for d in drive_counts]
        plt.bar(x + (i - len(read_workloads)/2 + 0.5) * bar_width, values, 
               width=bar_width, color=colors[workload], label=f"{workload.replace('_', ' ').title()}")
    
    # Plot write workloads throughput
    for i, workload in enumerate(write_workloads):
        values = [drive_results[d][workload]['system_stats']['total_throughput_workload'] 
                 if workload in drive_results[d] else 0 for d in drive_counts]
        plt.bar(x + (i + 1) * bar_width, values, 
               width=bar_width, color=colors[workload], label=f"{workload.replace('_', ' ').title()}")
    
    # Plot mixed workloads throughput
    for i, workload in enumerate(mixed_workloads):
        values = [drive_results[d][workload]['system_stats']['total_throughput_workload'] 
                 if workload in drive_results[d] else 0 for d in drive_counts]
        plt.bar(x + (i + 3) * bar_width, values, 
               width=bar_width, color=colors[workload], label=f"{workload.replace('_', ' ').title()}")
    
    plt.xlabel('Number of Drives')
    plt.ylabel('Throughput (MB/s)')
    plt.title('Throughput Comparison Across Workloads')
    plt.xticks(x, drive_counts)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/throughput_comparison.png")
    
    # Plot 2: IOPS comparison
    plt.figure(figsize=(14, 10))
    
    for i, workload in enumerate(workload_patterns):
        values = [drive_results[d][workload]['system_stats']['total_iops_workload'] 
                 if workload in drive_results[d] else 0 for d in drive_counts]
        plt.bar(x + (i - len(workload_patterns)/2 + 0.5) * bar_width, values, 
               width=bar_width, color=colors.get(workload, f"C{i}"), 
               label=f"{workload.replace('_', ' ').title()}")
    
    plt.xlabel('Number of Drives')
    plt.ylabel('IOPS')
    plt.title('IOPS Comparison Across Workloads')
    plt.xticks(x, drive_counts)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/iops_comparison.png")
    
    # Plot 3: Latency comparison
    plt.figure(figsize=(14, 10))
    
    # Create grouped bar chart for read and write latencies
    for i, workload in enumerate(workload_patterns):
        read_latencies = [drive_results[d][workload]['system_stats']['avg_read_latency'] * 1000  # Convert to ms
                        if workload in drive_results[d] else 0 for d in drive_counts]
        write_latencies = [drive_results[d][workload]['system_stats']['avg_write_latency'] * 1000  # Convert to ms
                         if workload in drive_results[d] else 0 for d in drive_counts]
        
        # Position for this workload's group
        pos = x + (i - len(workload_patterns)/2 + 0.5) * (bar_width * 2.5)
        
        plt.bar(pos, read_latencies, width=bar_width, color='blue', 
               alpha=0.7, label=f"{workload.replace('_', ' ').title()} Read" if i == 0 else "")
        plt.bar(pos + bar_width, write_latencies, width=bar_width, color='red', 
               alpha=0.7, label=f"{workload.replace('_', ' ').title()} Write" if i == 0 else "")
    
    plt.xlabel('Number of Drives')
    plt.ylabel('Latency (ms)')
    plt.title('Average Latency Comparison')
    plt.xticks(x, drive_counts)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/latency_comparison.png")
    
    # Plot 4: Backpressure comparison
    plt.figure(figsize=(14, 10))
    
    for i, workload in enumerate(workload_patterns):
        values = [drive_results[d][workload]['workload_stats']['backpressure_ratio'] * 100
                 if workload in drive_results[d] else 0 for d in drive_counts]
        plt.bar(x + (i - len(workload_patterns)/2 + 0.5) * bar_width, values, 
               width=bar_width, color=colors.get(workload, f"C{i}"), 
               label=f"{workload.replace('_', ' ').title()}")
    
    plt.xlabel('Number of Drives')
    plt.ylabel('Backpressure Rate (%)')
    plt.title('Backpressure Comparison Across Workloads')
    plt.xticks(x, drive_counts)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/backpressure_comparison.png")
    
    # Plot 5: Queue Utilization
    plt.figure(figsize=(14, 10))
    
    for i, workload in enumerate(workload_patterns):
        values = []
        for d in drive_counts:
            if workload in drive_results[d]:
                # Average queue utilization across all drives
                drive_stats = drive_results[d][workload]['drive_stats']
                avg_util = np.mean([stats['avg_queue_utilization'] / QUEUE_SIZE * 100 
                                  for stats in drive_stats.values()])
                values.append(avg_util)
            else:
                values.append(0)
        
        plt.bar(x + (i - len(workload_patterns)/2 + 0.5) * bar_width, values, 
               width=bar_width, color=colors.get(workload, f"C{i}"), 
               label=f"{workload.replace('_', ' ').title()}")
    
    plt.xlabel('Number of Drives')
    plt.ylabel('Queue Utilization (%)')
    plt.title('Average Queue Utilization Across Workloads')
    plt.xticks(x, drive_counts)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/queue_utilization.png")
    
    print(f"\nWorkload comparison plots saved in directory: {plot_dir}/")
    # Individual workload detailed plots
    for workload in workload_patterns:
        # Skip if we don't have results for this workload
        if not any(workload in drive_results[d] for d in drive_counts):
            continue
            
        plt.figure(figsize=(14, 10))
        
        # Extract data for this workload
        drive_nums = []
        throughputs = []
        read_throughputs = []
        write_throughputs = []
        iops_values = []
        read_iops = []
        write_iops = []
        
        for d in drive_counts:
            if workload in drive_results[d]:
                stats = drive_results[d][workload]['system_stats']
                drive_nums.append(d)
                throughputs.append(stats['total_throughput_workload'])
                read_throughputs.append(stats['read_throughput_workload'])
                write_throughputs.append(stats['write_throughput_workload'])
                iops_values.append(stats['total_iops_workload'])
                read_iops.append(stats['read_iops_workload'])
                write_iops.append(stats['write_iops_workload'])
        
        plt.subplot(2, 2, 1)
        plt.plot(drive_nums, throughputs, marker='o', linestyle='-', label='Total')
        plt.plot(drive_nums, read_throughputs, marker='s', linestyle='--', label='Read')
        plt.plot(drive_nums, write_throughputs, marker='^', linestyle='-.', label='Write')
        plt.xlabel('Number of Drives')
        plt.ylabel('Throughput (MB/s)')
        plt.title(f'{workload.replace("_", " ").title()} Throughput Scaling')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(drive_nums, iops_values, marker='o', linestyle='-', label='Total')
        plt.plot(drive_nums, read_iops, marker='s', linestyle='--', label='Read')
        plt.plot(drive_nums, write_iops, marker='^', linestyle='-.', label='Write')
        plt.xlabel('Number of Drives')
        plt.ylabel('IOPS')
        plt.title(f'{workload.replace("_", " ").title()} IOPS Scaling')
        plt.grid(True)
        plt.legend()
        
        # Extract more data
        backpressure = []
        completion = []
        
        for d in drive_counts:
            if workload in drive_results[d]:
                backpressure.append(drive_results[d][workload]['workload_stats']['backpressure_ratio'] * 100)
                completion.append(drive_results[d][workload]['workload_stats']['successful_ratio'] * 100)
        
        plt.subplot(2, 2, 3)
        plt.plot(drive_nums, backpressure, marker='o', linestyle='-', color='red')
        plt.xlabel('Number of Drives')
        plt.ylabel('Backpressure Rate (%)')
        plt.title(f'{workload.replace("_", " ").title()} Backpressure')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(drive_nums, completion, marker='o', linestyle='-', color='green')
        plt.xlabel('Number of Drives')
        plt.ylabel('Completion Rate (%)')
        plt.title(f'{workload.replace("_", " ").title()} Completion Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{workload}_detailed.png")
    
    print(f"Individual workload detailed plots saved in directory: {plot_dir}/")
    plt.close('all')
def main():
    parser = argparse.ArgumentParser(description="Realistic Storage System Simulation with Variable Drive Speeds")
    parser.add_argument("--drive-counts", type=str, default="1,2,4,8,16",
                        help="Comma-separated list of drive counts to test (e.g. 1,2,4,8,16)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration (in seconds) for each simulation")
    parser.add_argument("--rate", type=int, default=1000,
                        help="Operations per second")
    parser.add_argument("--block-size", type=int, default=4096,
                        help="Block size in bytes")
    parser.add_argument("--queue-size", type=int, default=100,
                        help="Maximum queue size for each drive (in operations)")
    parser.add_argument("--backoff-time", type=float, default=0.001,
                        help="Time to wait when experiencing backpressure (seconds)")
    parser.add_argument("--workloads", type=str, default="all",
                        help="Comma-separated list of workloads to run (e.g. sequential_read,random_write) or 'all'")
    parser.add_argument("--output-dir", type=str, default="storage_results",
                        help="Directory to save results and plots")
    args = parser.parse_args()

    global BACKOFF_TIME
    BACKOFF_TIME = args.backoff_time
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse drive counts
    drive_counts = [int(x.strip()) for x in args.drive_counts.split(",")]
    
    # Define workload patterns
    all_workloads = [
        "sequential_read",
        "sequential_write",
        "random_read",
        "random_write",
        "sequential_mixed",
        "random_mixed"
    ]
    
    # Parse workload list
    if args.workloads.lower() == "all":
        workloads = all_workloads
    else:
        workloads = [w.strip() for w in args.workloads.split(",")]
        for w in workloads:
            if w not in all_workloads:
                print(f"Warning: Unknown workload '{w}'. Valid workloads are: {', '.join(all_workloads)}")
                workloads.remove(w)
    
    print(f"Running the following workloads: {', '.join(workloads)}")
    print(f"Testing with {len(drive_counts)} drive configurations: {drive_counts}")
    
    # Dictionary to store results of all simulations
    results = {}
    
    # Run simulations for each combination of drive count and workload
    for workload in workloads:
        print(f"\n{'='*80}")
        print(f"RUNNING {workload.upper()} WORKLOAD SIMULATIONS")
        print(f"{'='*80}")
        
        for nd in drive_counts:
            print(f"\n{'-'*40}")
            print(f"Testing {workload} with {nd} drives")
            print(f"{'-'*40}")
            
            # Run the simulation for this combination
            result = run_workload_simulation(
                num_drives=nd,
                duration=args.duration,
                rate=args.rate,
                block_size=args.block_size,
                queue_size=args.queue_size,
                workload_pattern_name=workload
            )
            
            # Store the result
            result_key = f"{workload}_{nd}"
            results[result_key] = result
            
            # Small pause between configurations
            time.sleep(2)
    
    # Save results to JSON file
    result_filename = os.path.join(args.output_dir, "storage_simulation_results.json")
    
    # Convert numpy values to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = {}
    for key, result in results.items():
        serializable_results[key] = json.loads(
            json.dumps(result, default=convert_to_serializable)
        )
    
    with open(result_filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {result_filename}")
    
    # Plot comparison results
    plot_workload_comparison(results, drive_counts, workloads)
    
    print("\nSimulation completed successfully!")

if __name__ == '__main__':
    main()
