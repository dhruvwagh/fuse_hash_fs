#!/usr/bin/env bash
#
# run_benchmark.sh
#
# Iterates over 1,2,4,8,16 drives, measures throughput by
# performing parallel writes and reads on the mounted filesystem.
#

DRIVE_COUNTS=(1 2 4 8 16)
MNT=/tmp/fuse_mount
EXE=./my_fs  # path to the compiled fuse binary
FILE_SIZE_MB=10
PARALLEL_JOBS=4

if [ ! -f "$EXE" ]; then
  echo "ERROR: $EXE not found. Build it first."
  exit 1
fi

mkdir -p $MNT

for drives in "${DRIVE_COUNTS[@]}"; do
  echo "==========================================="
  echo "Benchmark: DRIVES=$drives"
  echo "==========================================="

  # Start FUSE in background
  $EXE -o drives=$drives $MNT -f -d &
  FUSE_PID=$!

  # Wait a moment for the FS to mount
  sleep 2

  # Confirm mount
  if [ ! "$(ls -A $MNT 2>/dev/null)" ]; then
    echo "Filesystem mounted (possibly empty root)."
  fi

  # Start timer
  start_time=$(date +%s)

  # Example: parallel writes
  echo "Writing $PARALLEL_JOBS x ${FILE_SIZE_MB}MB files..."
  for i in $(seq 1 $PARALLEL_JOBS); do
    dd if=/dev/zero of=$MNT/file_$i bs=1M count=$FILE_SIZE_MB oflag=direct status=none &
  done
  wait

  # parallel reads
  echo "Reading those files back..."
  for i in $(seq 1 $PARALLEL_JOBS); do
    dd if=$MNT/file_$i of=/dev/null bs=1M status=none &
  done
  wait

  end_time=$(date +%s)
  elapsed=$(( end_time - start_time ))

  total_data_mb=$(( FILE_SIZE_MB * PARALLEL_JOBS ))
  if [ $elapsed -gt 0 ]; then
    throughput=$(( total_data_mb / elapsed ))
  else
    throughput=$(( total_data_mb ))
  fi

  echo "Total data: $total_data_mb MB in $elapsed seconds"
  echo "Throughput: $throughput MB/s"

  # Clean up test files
  rm -f $MNT/file_*

  # Unmount
  fusermount3 -u $MNT
  kill -9 $FUSE_PID 2>/dev/null
  sleep 1

done

echo "Benchmark complete."
