#!/usr/bin/env bash

echo "===== 1) MKDIR ====="
mkdir testdir
ls -l

echo
echo "===== 2) CREATE + WRITE ====="
echo "Hello from FUSE" > testdir/file
ls -l testdir
cat testdir/file

echo
echo "===== 3) READ + GETATTR ====="
echo "Contents of file:"
cat testdir/file
stat testdir/file

echo
echo "===== 4) ACCESS (test -r) ====="
if [ -r testdir/file ]; then
  echo "testdir/file is readable"
else
  echo "testdir/file is not readable"
fi

echo
echo "===== 5) STATFS (df) ====="
df -h .

echo
echo "===== 6) RENAME ====="
mv testdir/file testdir/file_renamed
ls -l testdir

echo
echo "===== 7) UNLINK (remove) ====="
rm testdir/file_renamed
ls -l testdir

echo
echo "===== 8) RMDIR ====="
rmdir testdir
ls -l

echo
echo "===== 9) FLUSH & RELEASE test ====="
# Flush & release happen automatically on close, but we can do a system-wide sync:
sync
echo "Flushed (sync called)."

echo
echo "===== ALL DONE! ====="
