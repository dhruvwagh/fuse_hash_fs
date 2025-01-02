#!/bin/bash

# ===============================================
# Benchmarking Tool for FUSE-Based Filesystem
# ===============================================

# Configuration Parameters
FS_EXECUTABLE="./my_fs"                       # Path to your FUSE filesystem executable
MOUNT_POINT="/tmp/fuse_mount"                  # Mount point directory
TEST_DIR="$MOUNT_POINT/test_dir"               # Directory within the mount point for testing
DRIVE_COUNTS=(1 2 4)                            # Array of drive counts to test
RESULTS_DIR="./benchmark_results"              # Directory to store benchmark results
LOG_FILE="./benchmark_log.txt"                 # Log file for filesystem output
SUMMARY_FILE="$RESULTS_DIR/summary.csv"         # Summary CSV file

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to perform cleanup on exit
cleanup() {
    echo "Cleaning up..."
    if mountpoint -q "$MOUNT_POINT"; then
        echo "Unmounting filesystem..."
        fusermount3 -u "$MOUNT_POINT" >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to unmount filesystem. It might already be unmounted."
        else
            echo "Filesystem unmounted successfully."
        fi
    fi

    # Kill the filesystem process if it's still running
    if pgrep -f "$FS_EXECUTABLE" >/dev/null 2>&1; then
        echo "Terminating filesystem process..."
        pkill -f "$FS_EXECUTABLE"
        sleep 2
    fi

    echo "Cleanup completed."
    exit 0
}

# Trap EXIT, INT, TERM signals to perform cleanup
trap cleanup EXIT INT TERM

# Function to mount the filesystem in the background
mount_fs() {
    local drives=$1
    echo "Mounting filesystem with $drives drive(s)..."

    # Run the filesystem executable in the background and redirect output to log
    $FS_EXECUTABLE -o drives=$drives "$MOUNT_POINT" -f -d > "$LOG_FILE" 2>&1 &
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
    fusermount3 -u "$MOUNT_POINT" >/dev/null 2>&1
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
    local direction=$5    # 'write' or 'read'

    echo "Running $test_name test..."
    START_TIME=$(date +%s.%N)
    
    if [ "$direction" == "write" ]; then
        # Write test: Write data to the file
        dd if=/dev/zero of="$file_path" bs=$bs count=$count conv=fdatasync oflag=direct status=none
    elif [ "$direction" == "read" ]; then
        # Read test: Read data from the file and discard it
        dd if="$file_path" of=/dev/null bs=$bs count=$count conv=fdatasync status=none
    else
        echo "Error: Unknown direction '$direction' for $test_name."
        return 1
    fi

    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    BYTES=$(echo "$bs * $count" | bc)
    BYTES_MB=$(echo "scale=2; $BYTES / 1048576" | bc)
    
    if [ "$direction" == "write" ]; then
        BANDWIDTH=$(echo "scale=2; $BYTES_MB / $DURATION" | bc)
        echo "$test_name: $BANDWIDTH MB/s over $DURATION seconds"
        # Append to summary
        echo "$drives,$test_name,Bandwidth,${BANDWIDTH}" >> "$SUMMARY_FILE"
    elif [ "$direction" == "read" ]; then
        BANDWIDTH=$(echo "scale=2; $BYTES_MB / $DURATION" | bc)
        echo "$test_name: $BANDWIDTH MB/s over $DURATION seconds"
        # Append to summary
        echo "$drives,$test_name,Bandwidth,${BANDWIDTH}" >> "$SUMMARY_FILE"
    fi
}

# Function to perform IOPS tests using dd
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

    # Append to summary
    echo "$drives,$test_name,IOPS,${IOPS}" >> "$SUMMARY_FILE"
}

# Function to run all tests for a given number of drives
run_tests_for_drives() {
    local drives=$1
    echo "----------------------------------------------"
    echo "Starting tests for $drives drive(s)..."
    echo "----------------------------------------------"

    # Mount the filesystem
    mount_fs "$drives"
    if [ $? -ne 0 ]; then
        echo "Skipping tests for $drives drive(s) due to mount failure."
        echo ""
        return
    fi

    # Ensure test directory exists
    mkdir -p "$TEST_DIR"

    # Define test file paths
    SEQ_WRITE_FILE="$TEST_DIR/seq_write_test"
    SEQ_READ_FILE="$SEQ_WRITE_FILE"               # Reading the same file
    IOPS_WRITE_FILE="$TEST_DIR/iops_write_test"
    IOPS_READ_FILE="$TEST_DIR/iops_read_test"

    # Perform Throughput Tests
    throughput_test "Sequential Write" "$SEQ_WRITE_FILE" "1M" "1000" "write"
    throughput_test "Sequential Read" "$SEQ_READ_FILE" "1M" "1000" "read"

    # Perform IOPS Tests
    iops_test "Random Write IOPS" "$IOPS_WRITE_FILE" "4k" "10000"
    iops_test "Random Read IOPS" "$IOPS_READ_FILE" "4k" "10000"

    # Remove test files
    rm -f "$SEQ_WRITE_FILE" "$IOPS_WRITE_FILE" "$IOPS_READ_FILE"

    # Unmount the filesystem
    unmount_fs

    echo "Completed tests for $drives drive(s)."
    echo ""
}

# Main Script Execution

# Check if necessary commands are available
if ! command_exists fusermount3; then
    echo "Error: 'fusermount3' is not installed. Please install it and rerun the script."
    exit 1
fi

if ! command_exists dd; then
    echo "Error: 'dd' is not installed. Please install it and rerun the script."
    exit 1
fi

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    mkdir -p "$MOUNT_POINT"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create mount point directory at $MOUNT_POINT."
        exit 1
    fi
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create results directory at $RESULTS_DIR."
    exit 1
fi

# Initialize Summary CSV
echo "Drives,Test Type,Metric,Value" > "$SUMMARY_FILE"

echo "Starting benchmarking tests..."
echo ""

# Iterate over each drive count and run tests
for drives in "${DRIVE_COUNTS[@]}"; do
    run_tests_for_drives "$drives"
done

echo "All benchmarks completed. Results are stored in '$RESULTS_DIR'."
