#!/bin/bash

# ===============================================
# Basic File Operations Testing Tool for FUSE-Based Filesystem
# ===============================================

# Configuration Parameters
FS_EXECUTABLE="./my_fs"                       # Path to your FUSE filesystem executable
MOUNT_POINT="/tmp/fuse_mount"                  # Mount point directory
TEST_DIR="$MOUNT_POINT/test_dir"               # Directory within the mount point for testing
DRIVE_COUNTS=(1 2 4)                            # Array of drive counts to test
RESULTS_DIR="./test_results"                   # Directory to store test results
LOG_FILE="./test_log.txt"                      # Log file for filesystem output

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

# Function to perform basic file operations
perform_file_operations() {
    local drives=$1
    echo "Performing basic file operations for $drives drive(s)..."

    # Create Test Directory
    mkdir -p "$TEST_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create test directory at $TEST_DIR."
        return 1
    fi
    echo "Directory '$TEST_DIR' created."

    # Create a Test File
    TEST_FILE="$TEST_DIR/test_file.txt"
    echo "Hello, FUSE!" > "$TEST_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create test file '$TEST_FILE'."
        return 1
    fi
    echo "File '$TEST_FILE' created and written to."

    # Read the Test File
    CONTENT=$(cat "$TEST_FILE")
    if [ "$CONTENT" != "Hello, FUSE!" ]; then
        echo "Error: Content mismatch in '$TEST_FILE'. Expected 'Hello, FUSE!', got '$CONTENT'."
        return 1
    fi
    echo "File '$TEST_FILE' read successfully with correct content."

    # Append to the Test File
    echo "Appending more data." >> "$TEST_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to append to '$TEST_FILE'."
        return 1
    fi
    echo "Data appended to '$TEST_FILE'."

    # Read the Appended Content
    APPENDED_CONTENT=$(tail -n1 "$TEST_FILE")
    if [ "$APPENDED_CONTENT" != "Appending more data." ]; then
        echo "Error: Appended content mismatch in '$TEST_FILE'."
        return 1
    fi
    echo "Appended data verified in '$TEST_FILE'."

    # Delete the Test File
    rm -f "$TEST_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to delete test file '$TEST_FILE'."
        return 1
    fi
    echo "File '$TEST_FILE' deleted successfully."

    # Remove the Test Directory
    rmdir "$TEST_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to remove test directory '$TEST_DIR'. It might not be empty."
        return 1
    fi
    echo "Directory '$TEST_DIR' removed successfully."

    echo "Basic file operations completed successfully for $drives drive(s)."
    return 0
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

    # Perform File Operations
    perform_file_operations "$drives"
    if [ $? -ne 0 ]; then
        echo "Tests failed for $drives drive(s). Check the log for details."
    else
        echo "All tests passed for $drives drive(s)."
    fi

    # Unmount the filesystem
    echo "Unmounting filesystem..."
    fusermount3 -u "$MOUNT_POINT" >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to unmount filesystem. It might already be unmounted."
    else
        echo "Filesystem unmounted successfully."
    fi

    echo ""
}

# Main Script Execution

# Check if necessary commands are available
if ! command_exists fusermount3; then
    echo "Error: 'fusermount3' is not installed. Please install it and rerun the script."
    exit 1
fi

if ! command_exists mkdir || ! command_exists rm || ! command_exists rmdir || ! command_exists echo || ! command_exists cat; then
    echo "Error: Required file operation commands are missing."
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

echo "Starting basic file operations tests..."
echo ""

# Iterate over each drive count and run tests
for drives in "${DRIVE_COUNTS[@]}"; do
    run_tests_for_drives "$drives"
done

echo "All tests completed. Check '$LOG_FILE' and '$RESULTS_DIR' for details."
