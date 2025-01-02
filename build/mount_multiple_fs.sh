#!/bin/bash

# ==============================
# Configuration Parameters
# ==============================

# Path to your FUSE filesystem executable
FS_EXECUTABLE="./my_fs"

# Base directory for mount points
BASE_MOUNT_DIR="/tmp"

# Array of drive counts to mount
DRIVE_COUNTS=(1 2 3 5 4 8 12 16)

# Log directory
LOG_DIR="./mount_logs"

# ==============================
# Utility Functions
# ==============================

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to check if a directory is already a mount point
is_mounted() {
    mountpoint -q "$1"
}

# Function to mount the filesystem
mount_fs() {
    local drives=$1
    local mount_dir=$2
    local log_file=$3

    # Check if already mounted
    if is_mounted "$mount_dir"; then
        log_message "Mount point '$mount_dir' is already mounted. Skipping."
        return
    fi

    # Create mount directory if it doesn't exist
    if [ ! -d "$mount_dir" ]; then
        mkdir -p "$mount_dir"
        log_message "Created mount directory '$mount_dir'."
    fi

    # Start mounting in the background and redirect output to log file
    log_message "Mounting with $drives drive(s) on '$mount_dir'. Logging to '$log_file'."
    $FS_EXECUTABLE -o drives="$drives" "$mount_dir" -f -d > "$log_file" 2>&1 &

    # Capture PID of the mount process
    local pid=$!
    echo "$pid"
}

# Function to unmount all mounted filesystems created by this script
unmount_all() {
    log_message "Starting unmount of all mounted filesystems."

    for drives in "${DRIVE_COUNTS[@]}"; do
        local mount_dir="${BASE_MOUNT_DIR}/fuse_mount${drives}"
        if is_mounted "$mount_dir"; then
            log_message "Unmounting '$mount_dir'."
            fusermount3 -u "$mount_dir"
            if [ $? -eq 0 ]; then
                log_message "Successfully unmounted '$mount_dir'."
            else
                log_message "Failed to unmount '$mount_dir'. Check manually."
            fi
        else
            log_message "Mount point '$mount_dir' is not mounted. Skipping."
        fi
    done

    log_message "Unmounting process completed."
}

# ==============================
# Main Script Logic
# ==============================

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if the filesystem executable exists
if [ ! -x "$FS_EXECUTABLE" ]; then
    log_message "Error: Filesystem executable '$FS_EXECUTABLE' not found or not executable."
    exit 1
fi

# Parse command-line arguments
if [ "$1" == "unmount" ]; then
    unmount_all
    exit 0
fi

# Iterate over each drive count and mount the filesystem
for drives in "${DRIVE_COUNTS[@]}"; do
    mount_dir="${BASE_MOUNT_DIR}/fuse_mount${drives}"
    log_file="${LOG_DIR}/mount_${drives}_drives.log"

    # Mount the filesystem and capture the PID
    pid=$(mount_fs "$drives" "$mount_dir" "$log_file")

    # Optionally, you can store PIDs for later use
    echo "$pid" >> "${LOG_DIR}/mount_pids.txt"
done

log_message "Mounting of all specified filesystems initiated."
log_message "Check '$LOG_DIR' for individual mount logs."
