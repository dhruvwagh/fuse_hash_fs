#!/bin/bash

# ===============================================
# Simple Benchmarking Tool for FUSE-Based Filesystem
# ===============================================

# Configuration Parameters
FS_EXECUTABLE="/home/parallels/Documents/framework/fuse_hash_fs/build/my_fs"                       # Path to your FUSE filesystem executable
MOUNT_POINT="/tmp/my_fs_mount"                 # Mount point directory
TEST_DIR="$MOUNT_POINT/test_dir"               # Directory within the mount point for testing
DRIVE_COUNTS=(1 2 4)                            # Array of drive counts to test
BENCHMARK_DURATION=60                           # Duration for each I/O test in seconds
RESULTS_DIR="./benchmark_results"              # Directory to store benchmark results
FS_LOG_FILE="./fs_log.txt"                      # Log file for filesystem output

# FIO Test Parameters
# Not using fio as per user request

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to perform cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$FS_PID" ]; then
        kill $FS_PID 2>/dev/null
        wait $FS_PID 2>/dev/null
    fi
    fusermount3 -u "$MOUNT_POINT" 2>/dev/null
    echo "Cleanup completed."
}

# Trap EXIT, INT, TERM signals to perform cleanup
trap cleanup EXIT INT TERM

# Function to mount the filesystem in the background
mount_fs() {
    local drives=$1
    echo "Mounting filesystem with $drives drive(s)..."

    # Run the filesystem executable in the background and redirect output to log
    $FS_EXECUTABLE -o drives=$drives "$MOUNT_POINT" -d > "$FS_LOG_FILE" 2>&1 &
    FS_PID=$!

    # Wait for the filesystem to initialize
    # Allow up to 30 seconds for the filesystem to mount
    WAIT_TIME=0
    TIMEOUT=30
    while ! mountpoint -q "$MOUNT_POINT"; do
        sleep 1
        WAIT_TIME=$((WAIT_TIME + 1))
        if [ $WAIT_TIME -ge $TIMEOUT ]; then
            echo "Error: Filesystem failed to mount within $TIMEOUT seconds."
            kill $FS_PID 2>/dev/null
            return 1
        fi
    done

    echo "Filesystem mounted with $drives drive(s)."
    return 0
}

# Function to unmount the filesystem
unmount_fs() {
    echo "Unmounting filesystem..."
    fusermount3 -u "$MOUNT_POINT"
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to unmount filesystem. It might already be unmounted."
    else
        echo "Filesystem unmounted successfully."
    fi
}

# Function to perform throughput tests using dd
throughput_test() {
    local test_name=$1
    local file_path=$2
    local bs=$3
    local count=$4
    local oflag=$5
    local iflag=$6

    echo "Running $test_name test..."
    START_TIME=$(date +%s.%N)
    dd if=/dev/zero of="$file_path" bs=$bs count=$count conv=fdatasync oflag=$oflag status=none
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    BYTES=$(echo "$bs * $count" | bc)
    BYTES_MB=$(echo "scale=2; $BYTES / 1048576" | bc)
    BANDWIDTH=$(echo "scale=2; $BYTES_MB / $DURATION" | bc)
    echo "$test_name: $BANDWIDTH MB/s over $DURATION seconds"
}

# Function to perform IOPS tests
iops_test() {
    local test_name=$1
    local file_path=$2
    local bs=$3
    local count=$4

    echo "Running $test_name test..."
    START_TIME=$(date +%s.%N)
    for i in $(seq 1 $count); do
        dd if=/dev/zero of="$file_path" bs=$bs count=1 conv=fdatasync oflag=direct status=none &
    done
    wait
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    IOPS=$(echo "scale=2; $count / $DURATION" | bc)
    echo "$test_name: $IOPS IOPS over $DURATION seconds"
}

# Main Benchmarking Loop
for drives in "${DRIVE_COUNTS[@]}"; do
    echo "=============================================="
    echo "Starting benchmark with $drives drive(s)..."
    echo "=============================================="

    # Mount the filesystem
    mount_fs "$drives"
    if [ $? -ne 0 ]; then
        echo "Skipping benchmarks for $drives drive(s) due to mount failure."
        echo ""
        continue
    fi

    # Ensure test directory exists
    mkdir -p "$TEST_DIR"

    # Define test file paths
    SEQ_WRITE_FILE="$TEST_DIR/seq_write_test"
    SEQ_READ_FILE="$SEQ_WRITE_FILE"               # Reading the same file
    IOPS_WRITE_FILE="$TEST_DIR/iops_write_test"
    IOPS_READ_FILE="$TEST_DIR/iops_read_test"

    # Perform Throughput Tests
    throughput_test "Sequential Write" "$SEQ_WRITE_FILE" "1M" "1000" "direct" ""
    throughput_test "Sequential Read" "$SEQ_READ_FILE" "1M" "1000" "" "direct"

    # Perform IOPS Tests
    iops_test "Random Write IOPS" "$IOPS_WRITE_FILE" "4k" "10000"
    iops_test "Random Read IOPS" "$IOPS_READ_FILE" "4k" "10000"

    # Remove test files
    rm -f "$SEQ_WRITE_FILE" "$IOPS_WRITE_FILE" "$IOPS_READ_FILE"

    # Unmount the filesystem
    unmount_fs

    echo "Completed benchmarks for $drives drive(s)."
    echo ""

    # Allow some time before the next test
    sleep 2
done

echo "All benchmarks completed. Results are stored in '$RESULTS_DIR'."
