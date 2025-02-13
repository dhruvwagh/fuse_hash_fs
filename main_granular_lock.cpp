#define FUSE_USE_VERSION 35

#include <fuse3/fuse_lowlevel.h>
#include <fuse3/fuse_opt.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <future>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <array>

// For CPU affinity (thread pinning)
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/syscall.h>

// -----------------------------------------------------------------------------
// Custom FUSE option struct
struct options {
    int drives;
};
static struct options g_opts = {4};

static struct fuse_opt option_spec[] = {
    {"drives=%d", offsetof(struct options, drives), 0},
    FUSE_OPT_END
};

static constexpr int SSD_READ_LATENCY_US  = 1;
static constexpr int SSD_WRITE_LATENCY_US = 5;
static int NUM_DRIVES = 4;

static constexpr size_t BLOCK_SIZE = 4096;   
static constexpr size_t MAX_BLOCKS_PER_DRIVE = 64;     
static constexpr size_t MAX_BYTES_PER_DRIVE = MAX_BLOCKS_PER_DRIVE * BLOCK_SIZE;

// -----------------------------------------------------------------------------
// Forward declarations
enum class RequestOp {
    LOOKUP, MKDIR, CREATE, UNLINK, RMDIR, RENAME,
    WRITE, READ, GETATTR, FLUSH, RELEASE, READDIR,
    ACCESS, STATFS
};
// Forward declarations for FUSE operations (add this after RequestOp enum)
class FsRequest;  // Forward declare FsRequest first

// Forward declarations of all process functions
static void processLookup(FsRequest &r);
static void processMkdir(FsRequest &r);
static void processCreate(FsRequest &r);
static void processUnlink(FsRequest &r);
static void processRmdir(FsRequest &r);
static void processRename(FsRequest &r);
static void processWrite(FsRequest &r);
static void processRead(FsRequest &r);
static void processGetattr(FsRequest &r);
static void processFlush(FsRequest &r);
static void processRelease(FsRequest &r);
static void processReaddir(FsRequest &r);
static void processAccess(FsRequest &r);
static void processStatfs(FsRequest &r);
// -----------------------------------------------------------------------------
// Improved Locking System
class InodeLockManager {
    struct LockPair {
        std::shared_mutex rw_lock;
        std::atomic<uint32_t> active_readers{0};
    };
    std::vector<LockPair> locks;
    static constexpr size_t NUM_LOCKS = 64;  // Power of 2 for better distribution

public:
    InodeLockManager() : locks(NUM_LOCKS) {}

    std::shared_lock<std::shared_mutex> getReadLock(fuse_ino_t ino) {
        size_t idx = ino % NUM_LOCKS;
        auto& lock_pair = locks[idx];
        lock_pair.active_readers++;
        return std::shared_lock<std::shared_mutex>(lock_pair.rw_lock);
    }

    std::unique_lock<std::shared_mutex> getWriteLock(fuse_ino_t ino) {
        size_t idx = ino % NUM_LOCKS;
        return std::unique_lock<std::shared_mutex>(locks[idx].rw_lock);
    }

    void releaseReadLock(fuse_ino_t ino) {
        locks[ino % NUM_LOCKS].active_readers--;
    }
};

static InodeLockManager g_lockManager;

// -----------------------------------------------------------------------------
// Stats structures with atomic variables
struct LatencyStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_us{0};
    std::atomic<uint64_t> max_us{0};
    std::atomic<uint64_t> min_us{std::numeric_limits<uint64_t>::max()};

    LatencyStats() = default;

    // No copy
    LatencyStats(const LatencyStats&) = delete;
    LatencyStats& operator=(const LatencyStats&) = delete;

    // Move constructor
    LatencyStats(LatencyStats&& other) noexcept {
        count.store(other.count.load());
        total_us.store(other.total_us.load());
        max_us.store(other.max_us.load());
        min_us.store(other.min_us.load());
    }
    
    // Move assignment
    LatencyStats& operator=(LatencyStats&& other) noexcept {
        if (this != &other) {
            count.store(other.count.load());
            total_us.store(other.total_us.load());
            max_us.store(other.max_us.load());
            min_us.store(other.min_us.load());
        }
        return *this;
    }

    void record(uint64_t latency_us) {
        count.fetch_add(1, std::memory_order_relaxed);
        total_us.fetch_add(latency_us, std::memory_order_relaxed);

        uint64_t old_max = max_us.load(std::memory_order_relaxed);
        while(latency_us > old_max &&
              !max_us.compare_exchange_weak(old_max, latency_us, std::memory_order_relaxed)) {}

        uint64_t old_min = min_us.load(std::memory_order_relaxed);
        while(latency_us < old_min &&
              !min_us.compare_exchange_weak(old_min, latency_us, std::memory_order_relaxed)) {}
    }
};

struct DriveStats {
    std::atomic<uint64_t> read_ops{0};
    std::atomic<uint64_t> write_ops{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> queue_length{0};
    std::atomic<uint64_t> queue_wait_time_us{0};
    LatencyStats operation_latency;
    std::atomic<uint64_t> active_time_us{0};

    DriveStats() = default;

    // No copy
    DriveStats(const DriveStats&) = delete;
    DriveStats& operator=(const DriveStats&) = delete;

    // Move constructor
    DriveStats(DriveStats&& other) noexcept {
        read_ops.store(other.read_ops.load());
        write_ops.store(other.write_ops.load());
        bytes_read.store(other.bytes_read.load());
        bytes_written.store(other.bytes_written.load());
        queue_length.store(other.queue_length.load());
        queue_wait_time_us.store(other.queue_wait_time_us.load());
        operation_latency = std::move(other.operation_latency);
        active_time_us.store(other.active_time_us.load());
    }

    // Move assignment
    DriveStats& operator=(DriveStats&& other) noexcept {
        if (this != &other) {
            read_ops.store(other.read_ops.load());
            write_ops.store(other.write_ops.load());
            bytes_read.store(other.bytes_read.load());
            bytes_written.store(other.bytes_written.load());
            queue_length.store(other.queue_length.load());
            queue_wait_time_us.store(other.queue_wait_time_us.load());
            operation_latency = std::move(other.operation_latency);
            active_time_us.store(other.active_time_us.load());
        }
        return *this;
    }
};

struct ThreadStats {
    std::atomic<uint64_t> processed_requests{0};
    std::atomic<uint64_t> active_time_us{0};
    std::atomic<uint64_t> idle_time_us{0};
    std::chrono::steady_clock::time_point last_activity;

    ThreadStats() = default;

    // No copy
    ThreadStats(const ThreadStats&) = delete;
    ThreadStats& operator=(const ThreadStats&) = delete;

    // Move
    ThreadStats(ThreadStats&& other) noexcept {
        processed_requests.store(other.processed_requests.load());
        active_time_us.store(other.active_time_us.load());
        idle_time_us.store(other.idle_time_us.load());
        last_activity = other.last_activity;
    }

    ThreadStats& operator=(ThreadStats&& other) noexcept {
        if (this != &other) {
            processed_requests.store(other.processed_requests.load());
            active_time_us.store(other.active_time_us.load());
            idle_time_us.store(other.idle_time_us.load());
            last_activity = other.last_activity;
        }
        return *this;
    }
};

struct LockStats {
    std::atomic<uint64_t> contentions{0};
    std::atomic<uint64_t> total_wait_time_us{0};
    std::atomic<uint64_t> max_wait_time_us{0};

    LockStats() = default;

    // No copy
    LockStats(const LockStats&) = delete;
    LockStats& operator=(const LockStats&) = delete;

    // Move
    LockStats(LockStats&& other) noexcept {
        contentions.store(other.contentions.load());
        total_wait_time_us.store(other.total_wait_time_us.load());
        max_wait_time_us.store(other.max_wait_time_us.load());
    }

    LockStats& operator=(LockStats&& other) noexcept {
        if (this != &other) {
            contentions.store(other.contentions.load());
            total_wait_time_us.store(other.total_wait_time_us.load());
            max_wait_time_us.store(other.max_wait_time_us.load());
        }
        return *this;
    }

    void record_contention(uint64_t wait_time_us) {
        contentions.fetch_add(1, std::memory_order_relaxed);
        total_wait_time_us.fetch_add(wait_time_us, std::memory_order_relaxed);

        uint64_t old_max = max_wait_time_us.load(std::memory_order_relaxed);
        while(wait_time_us > old_max &&
              !max_wait_time_us.compare_exchange_weak(old_max, wait_time_us, std::memory_order_relaxed)) {}
    }
};

// -----------------------------------------------------------------------------
// Global vectors
static std::vector<DriveStats>  g_drive_stats;
static std::vector<ThreadStats> g_thread_stats;
static std::vector<LockStats>   g_lock_stats;

static std::mutex g_stats_mutex;
static std::ofstream g_stats_file;
static std::chrono::steady_clock::time_point g_start_time;

static std::thread g_stats_thread;
static std::atomic<bool> g_stop_stats{false};

static std::mutex* g_queueMutex = nullptr;
static std::condition_variable* g_queueCond = nullptr;
static std::condition_variable* g_queueCondNotFull = nullptr;
static std::thread* g_workers = nullptr;
static std::atomic<bool> g_stopThreads{false};

// forward declare FsRequest
class FsRequest;
static std::vector<FsRequest>* g_queues = nullptr;
static std::vector<size_t> g_queueCapacity;

// -----------------------------------------------------------------------------
// Minimal "inode" structures
struct FsNode {
    bool isDir;
    mode_t mode;
    std::vector<uint8_t> fileData;
    std::map<std::string, fuse_ino_t> children;
};

static std::mutex g_inodeCreationMutex;
static std::atomic<fuse_ino_t> g_nextIno{2}; 
static std::vector<std::shared_ptr<FsNode>> g_inodeTable;

static fuse_ino_t createInode(bool isDir, mode_t mode)
{
    std::lock_guard<std::mutex> lk(g_inodeCreationMutex);
    fuse_ino_t ino = g_nextIno.fetch_add(1, std::memory_order_relaxed);
    auto node = std::make_shared<FsNode>();
    node->isDir = isDir;
    node->mode = mode;
    if((size_t)ino >= g_inodeTable.size()) {
        g_inodeTable.resize(ino + 1);
    }
    g_inodeTable[ino] = node;
    return ino;
}

static std::shared_ptr<FsNode> getNode(fuse_ino_t ino)
{
    if(ino < g_inodeTable.size()) {
        return g_inodeTable[ino];
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Pin to CPU
void pinThreadToCore(std::thread &thr, int coreId)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);

    pthread_t native = thr.native_handle();
    int rc = pthread_setaffinity_np(native, sizeof(cpu_set_t), &cpuset);
    if(rc != 0) {
        std::cerr << "[Warning] Could not pin to core " << coreId
                  << ": " << strerror(errno) << std::endl;
    } else {
        std::cout << "[Info] Worker pinned to core " << coreId << std::endl;
    }
}

// -----------------------------------------------------------------------------
// SsdModel
class SsdModel {
public:
    int drive_id;
    explicit SsdModel(int id) : drive_id(id) {}

    void readLatency(size_t /*bytes*/) {
        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_READ_LATENCY_US));
        auto end = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        g_drive_stats[drive_id].active_time_us.fetch_add(dur, std::memory_order_relaxed);
    }

    void writeLatency(size_t /*bytes*/) {
        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_WRITE_LATENCY_US));
        auto end = std::chrono::steady_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        g_drive_stats[drive_id].active_time_us.fetch_add(dur, std::memory_order_relaxed);
    }
};

static std::vector<std::unique_ptr<SsdModel>> g_ssdModels;

// Continuing from previous section...

// -----------------------------------------------------------------------------
// Hash helpers
static uint64_t fnv1a_hash_offset(fuse_ino_t ino, off_t offset)
{
    std::string combined = std::to_string(ino) + "-" + std::to_string(offset / BLOCK_SIZE);
    uint64_t hash = 1469598103934665603ULL;
    for (char c : combined) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t fnv1a_hash_string(const std::string &s)
{
    uint64_t hash = 1469598103934665603ULL;
    for (char c : s) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// -----------------------------------------------------------------------------
// RequestMonitor
struct RequestMonitor {
    std::chrono::steady_clock::time_point enqueue_time;
    std::chrono::steady_clock::time_point start_time;
    int assigned_drive;
    RequestOp op;
    size_t size;

    void record_start() {
        start_time = std::chrono::steady_clock::now();
    }

    void record_completion() {
        auto now = std::chrono::steady_clock::now();
        auto queue_time = std::chrono::duration_cast<std::chrono::microseconds>(start_time - enqueue_time).count();
        auto proc_time = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count();

        g_drive_stats[assigned_drive].queue_wait_time_us.fetch_add(queue_time, std::memory_order_relaxed);
        g_drive_stats[assigned_drive].operation_latency.record(proc_time);
        g_drive_stats[assigned_drive].active_time_us.fetch_add(proc_time, std::memory_order_relaxed);

        if(op==RequestOp::READ) {
            g_drive_stats[assigned_drive].read_ops.fetch_add(1, std::memory_order_relaxed);
        } else if(op==RequestOp::WRITE) {
            g_drive_stats[assigned_drive].write_ops.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

// -----------------------------------------------------------------------------
// FsRequest
class FsRequest {
public:
    RequestOp op;
    fuse_req_t req;
    fuse_ino_t parent;
    std::string name;
    fuse_ino_t ino;
    fuse_ino_t newparent;
    std::string newname;
    size_t size;
    off_t offset;
    mode_t mode;
    std::vector<uint8_t> data;
    unsigned int rename_flags;
    struct fuse_file_info fi;
    int access_mask;
    std::promise<void> completion_promise;
    RequestMonitor monitor;

    FsRequest() = default;
    FsRequest(FsRequest&&) = default;
    FsRequest& operator=(FsRequest&&) = default;
    FsRequest(const FsRequest&) = delete;
    FsRequest& operator=(const FsRequest&) = delete;
};

// -----------------------------------------------------------------------------
// Worker
static void workerThreadFunc(int driveIndex)
{
    auto &stats = g_thread_stats[driveIndex];
    stats.last_activity = std::chrono::steady_clock::now();
    auto &ssd = *(g_ssdModels[driveIndex]);

    while(!g_stopThreads.load()) {
        std::vector<FsRequest> localReqs;
        {
            auto wait_start = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> lk(g_queueMutex[driveIndex]);
            g_queueCond[driveIndex].wait(lk, [driveIndex]{
                return (!g_queues[driveIndex].empty() || g_stopThreads.load());
            });
            auto wait_end = std::chrono::steady_clock::now();
            uint64_t idle_us = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();
            stats.idle_time_us.fetch_add(idle_us, std::memory_order_relaxed);

            if(g_stopThreads.load()) break;

            localReqs = std::move(g_queues[driveIndex]);
            g_queues[driveIndex].clear();
        }

        for(auto &req : localReqs) {
            auto proc_start = std::chrono::steady_clock::now();
            req.monitor.record_start();

            size_t used = 0;
            if(req.op==RequestOp::WRITE || req.op==RequestOp::READ) {
                size_t d = req.data.size();
                size_t blocks = (d+BLOCK_SIZE-1)/BLOCK_SIZE;
                used = blocks*BLOCK_SIZE;
            }

            try {
                switch(req.op) {
                    case RequestOp::LOOKUP:   processLookup(req); break;
                    case RequestOp::MKDIR:    processMkdir(req);  break;
                    case RequestOp::CREATE:   processCreate(req); break;
                    case RequestOp::UNLINK:   processUnlink(req); break;
                    case RequestOp::RMDIR:    processRmdir(req);  break;
                    case RequestOp::RENAME:   processRename(req); break;
                    case RequestOp::WRITE:
                        ssd.writeLatency(req.data.size());
                        processWrite(req);
                        break;
                    case RequestOp::READ:
                        ssd.readLatency(req.size);
                        processRead(req);
                        break;
                    case RequestOp::GETATTR:  processGetattr(req); break;
                    case RequestOp::FLUSH:    processFlush(req);   break;
                    case RequestOp::RELEASE:  processRelease(req); break;
                    case RequestOp::READDIR:  processReaddir(req); break;
                    case RequestOp::ACCESS:   processAccess(req);  break;
                    case RequestOp::STATFS:   processStatfs(req);  break;
                }
                req.monitor.record_completion();
                stats.processed_requests.fetch_add(1, std::memory_order_relaxed);

            } catch(std::exception &e) {
                std::lock_guard<std::mutex> g(g_stats_mutex);
                g_stats_file<<"[Error] Worker "<<driveIndex<<" exception: "<< e.what()<<"\n";
                fuse_reply_err(req.req, EIO);
            }

            auto proc_end = std::chrono::steady_clock::now();
            uint64_t proc_us = std::chrono::duration_cast<std::chrono::microseconds>(proc_end - proc_start).count();
            stats.active_time_us.fetch_add(proc_us, std::memory_order_relaxed);
            stats.last_activity = proc_end;

            g_drive_stats[driveIndex].queue_length.fetch_sub(1, std::memory_order_relaxed);

            if(used>0) {
                std::unique_lock<std::mutex> lk(g_queueMutex[driveIndex]);
                g_queueCapacity[driveIndex] += used;
                g_queueCondNotFull[driveIndex].notify_all();
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Enqueue
static void enqueueRequest(FsRequest &&r, const std::string &uniqueKey, bool offsetBased)
{
    uint64_t h = offsetBased ? fnv1a_hash_offset(r.ino, r.offset)
                            : fnv1a_hash_string(uniqueKey);
    int qIdx = h % NUM_DRIVES;
    r.monitor.enqueue_time = std::chrono::steady_clock::now();
    r.monitor.assigned_drive = qIdx;
    r.monitor.op = r.op;
    r.monitor.size = r.data.size();

    size_t needed = 0;
    if(r.op==RequestOp::WRITE || r.op==RequestOp::READ) {
        size_t ds = r.data.size();
        size_t blocks = (ds+BLOCK_SIZE-1)/BLOCK_SIZE;
        needed = blocks*BLOCK_SIZE;
    }

    {
        std::lock_guard<std::mutex> gg(g_stats_mutex);
        g_stats_file << "[Distribution] Op=" << (int)r.op
                     << " Drive=" << qIdx
                     << " Size=" << needed
                     << " Key=" << uniqueKey
                     << std::endl;
    }

    auto waitStart = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(g_queueMutex[qIdx]);
        g_queueCondNotFull[qIdx].wait(lk, [qIdx,needed]{
            return (g_queueCapacity[qIdx]>=needed) || g_stopThreads.load();
        });
        if(g_stopThreads.load()) return;

        uint64_t w = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - waitStart).count();
        g_drive_stats[qIdx].queue_wait_time_us.fetch_add(w, std::memory_order_relaxed);

        g_queues[qIdx].push_back(std::move(r));
        g_queueCapacity[qIdx] -= needed;
        g_drive_stats[qIdx].queue_length.fetch_add(1, std::memory_order_relaxed);
    }
    g_queueCond[qIdx].notify_one();
}
// -----------------------------------------------------------------------------
// Implementation of processX calls with granular locking
static void processLookup(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.parent);
    auto parentNode = getNode(r.parent);
    if(!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto it = parentNode->children.find(r.name);
    if(it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    fuse_ino_t child = it->second;
    auto childReadLock = g_lockManager.getReadLock(child);
    auto cnode = getNode(child);
    if(!cnode) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    struct fuse_entry_param e;
    memset(&e, 0, sizeof(e));
    e.ino = child;
    e.attr.st_ino = child;
    e.generation = 1;
    e.attr_timeout = 1.0;
    e.entry_timeout = 1.0;
    if(cnode->isDir) {
        e.attr.st_mode = S_IFDIR | cnode->mode;
        e.attr.st_nlink = 2 + cnode->children.size();
    } else {
        e.attr.st_mode = S_IFREG | cnode->mode;
        e.attr.st_size = cnode->fileData.size();
        e.attr.st_nlink = 1;
    }
    fuse_reply_entry(r.req, &e);
}

static void processMkdir(FsRequest &r) {
    auto writeLock = g_lockManager.getWriteLock(r.parent);
    auto parentNode = getNode(r.parent);
    if(!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    if(parentNode->children.find(r.name) != parentNode->children.end()) {
        fuse_reply_err(r.req, EEXIST);
        return;
    }
    
    fuse_ino_t newIno = createInode(true, r.mode);
    parentNode->children[r.name] = newIno;
    
    struct fuse_entry_param e;
    memset(&e, 0, sizeof(e));
    e.ino = newIno;
    e.attr.st_mode = S_IFDIR | r.mode;
    e.attr.st_ino = newIno;
    e.attr.st_nlink = 2;
    e.attr_timeout = 1.0;
    e.entry_timeout = 1.0;
    fuse_reply_entry(r.req, &e);
}

static void processCreate(FsRequest &r) {
    auto writeLock = g_lockManager.getWriteLock(r.parent);
    auto parentNode = getNode(r.parent);
    if(!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    if(parentNode->children.find(r.name) != parentNode->children.end()) {
        fuse_reply_err(r.req, EEXIST);
        return;
    }
    
    fuse_ino_t newIno = createInode(false, r.mode);
    parentNode->children[r.name] = newIno;

    struct fuse_entry_param e;
    memset(&e, 0, sizeof(e));
    e.ino = newIno;
    e.attr.st_mode = S_IFREG | r.mode;
    e.attr.st_ino = newIno;
    e.attr.st_nlink = 1;
    e.attr_timeout = 1.0;
    e.entry_timeout = 1.0;

    struct fuse_file_info fi;
    memset(&fi, 0, sizeof(fi));
    fi.fh = newIno;
    fuse_reply_create(r.req, &e, &fi);
}

static void processUnlink(FsRequest &r) {
    auto writeLock = g_lockManager.getWriteLock(r.parent);
    auto parentNode = getNode(r.parent);
    if(!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto it = parentNode->children.find(r.name);
    if(it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto childWriteLock = g_lockManager.getWriteLock(it->second);
    auto child = getNode(it->second);
    if(!child || child->isDir) {
        fuse_reply_err(r.req, EISDIR);
        return;
    }
    
    parentNode->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processRmdir(FsRequest &r) {
    auto writeLock = g_lockManager.getWriteLock(r.parent);
    auto parentNode = getNode(r.parent);
    if(!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto it = parentNode->children.find(r.name);
    if(it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto childWriteLock = g_lockManager.getWriteLock(it->second);
    auto child = getNode(it->second);
    if(!child || !child->isDir) {
        fuse_reply_err(r.req, ENOTDIR);
        return;
    }
    if(!child->children.empty()) {
        fuse_reply_err(r.req, ENOTEMPTY);
        return;
    }
    
    parentNode->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processRename(FsRequest &r) {
    // Lock both parent directories to prevent deadlocks
    fuse_ino_t first = std::min(r.parent, r.newparent);
    fuse_ino_t second = std::max(r.parent, r.newparent);
    
    auto lock1 = g_lockManager.getWriteLock(first);
    auto lock2 = (first != second) ? g_lockManager.getWriteLock(second) : std::unique_lock<std::shared_mutex>();
    
    auto oldP = getNode(r.parent);
    auto newP = getNode(r.newparent);
    if(!oldP || !oldP->isDir || !newP || !newP->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    auto it = oldP->children.find(r.name);
    if(it == oldP->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    fuse_ino_t oldIno = it->second;
    newP->children[r.newname] = oldIno;
    oldP->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processWrite(FsRequest &r) {
    auto writeLock = g_lockManager.getWriteLock(r.ino);
    auto node = getNode(r.ino);
    if(!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    size_t endPos = r.offset + r.data.size();
    if(endPos > node->fileData.size()) {
        node->fileData.resize(endPos);
    }
    std::memcpy(&node->fileData[r.offset], r.data.data(), r.data.size());
    fuse_reply_write(r.req, r.data.size());
}

static void processRead(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    auto node = getNode(r.ino);
    if(!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    size_t filesize = node->fileData.size();
    if((size_t)r.offset >= filesize) {
        fuse_reply_buf(r.req, nullptr, 0);
        return;
    }
    
    size_t toRead = r.size;
    if(r.offset + toRead > filesize) {
        toRead = filesize - r.offset;
    }
    fuse_reply_buf(r.req, (const char*)(node->fileData.data() + r.offset), toRead);
}

static void processGetattr(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    auto node = getNode(r.ino);
    if(!node) {
        if(r.ino == FUSE_ROOT_ID) {
            struct stat st{};
            st.st_ino = FUSE_ROOT_ID;
            st.st_mode = S_IFDIR | 0755;
            st.st_nlink = 2;
            st.st_uid = getuid();
            st.st_gid = getgid();
            fuse_reply_attr(r.req, &st, 1.0);
            return;
        }
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    
    struct stat st{};
    st.st_ino = r.ino;
    if(node->isDir) {
        st.st_mode = S_IFDIR | node->mode;
        st.st_nlink = 2 + node->children.size();
    } else {
        st.st_mode = S_IFREG | node->mode;
        st.st_size = node->fileData.size();
        st.st_nlink = 1;
    }
    st.st_uid = getuid();
    st.st_gid = getgid();
    fuse_reply_attr(r.req, &st, 1.0);
}

static void processFlush(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processRelease(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processReaddir(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    auto node = getNode(r.ino);
    if(!node || !node->isDir) {
        fuse_reply_err(r.req, ENOTDIR);
        return;
    }
    
    char* buf = (char*)malloc(r.size);
    if(!buf) {
        fuse_reply_err(r.req, ENOMEM);
        return;
    }
    
    size_t bpos = 0;
    auto addEntry = [&](fuse_ino_t e_ino, const char *n) {
        struct stat st{};
        st.st_ino = e_ino;
        auto en = getNode(e_ino);
        if(en) {
            st.st_mode = en->isDir ? (S_IFDIR|en->mode) : (S_IFREG|en->mode);
        }
        size_t esz = fuse_add_direntry(r.req, buf+bpos, r.size-bpos, n, &st, r.offset+1);
        if(esz>0 && bpos+esz <= r.size) {
            bpos += esz;
        }
    };
    
    if(r.offset == 0) {
        addEntry(r.ino, ".");
        addEntry(FUSE_ROOT_ID, "..");
        for(auto &ch: node->children) {
            addEntry(ch.second, ch.first.c_str());
        }
    }
    
    fuse_reply_buf(r.req, buf, bpos);
    free(buf);
}

static void processAccess(FsRequest &r) {
    auto readLock = g_lockManager.getReadLock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processStatfs(FsRequest &r) {
    struct statvfs st{};
    st.f_bsize = 4096;
    st.f_frsize = 4096;
    st.f_blocks = 1024*1024;
    st.f_bfree = 1024*1024;
    st.f_bavail = 1024*1024;
    st.f_files = 100000;
    st.f_ffree = 100000;
    st.f_favail = 100000;
    st.f_fsid = 1234;
    st.f_flag = 0;
    st.f_namemax = 255;
    fuse_reply_statfs(r.req, &st);
}
// -----------------------------------------------------------------------------
// FUSE low-level ops
static void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op = RequestOp::LOOKUP;
    r.req = req;
    r.parent = parent;
    r.name = name;
    enqueueRequest(std::move(r), std::to_string(parent)+"/"+name, false);
}

static void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode)
{
    FsRequest r;
    r.op = RequestOp::MKDIR;
    r.req = req;
    r.parent = parent;
    r.name = name;
    r.mode = mode;
    enqueueRequest(std::move(r), std::to_string(parent)+"/"+name, false);
}

static void ll_create(fuse_req_t req, fuse_ino_t parent, const char *name,
                      mode_t mode, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::CREATE;
    r.req = req;
    r.parent = parent;
    r.name = name;
    r.mode = mode;
    enqueueRequest(std::move(r), std::to_string(parent)+"/"+name, false);
}

static void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op = RequestOp::UNLINK;
    r.req = req;
    r.parent = parent;
    r.name = name;
    enqueueRequest(std::move(r), std::to_string(parent)+"/"+name, false);
}

static void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op = RequestOp::RMDIR;
    r.req = req;
    r.parent = parent;
    r.name = name;
    enqueueRequest(std::move(r), std::to_string(parent)+"/"+name, false);
}

static void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
                      fuse_ino_t newparent, const char *newname, unsigned int flags)
{
    FsRequest r;
    r.op = RequestOp::RENAME;
    r.req = req;
    r.parent = parent;
    r.name = name;
    r.newparent = newparent;
    r.newname = newname;
    r.rename_flags = flags;
    std::string key = std::to_string(parent)+"/"+name+"->"+
                     std::to_string(newparent)+"/"+newname;
    enqueueRequest(std::move(r), key, false);
}

static void ll_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    fi->fh = ino;
    fuse_reply_open(req, fi);
}

static void ll_write(fuse_req_t req, fuse_ino_t ino, const char *buf,
                     size_t size, off_t off, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::WRITE;
    r.req = req;
    r.ino = ino;
    r.offset = off;
    r.data.assign(buf, buf+size);
    enqueueRequest(std::move(r), "", true);
}

static void ll_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                    struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::READ;
    r.req = req;
    r.ino = ino;
    r.size = size;
    r.offset = off;
    enqueueRequest(std::move(r), "", true);
}

static void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::GETATTR;
    r.req = req;
    r.ino = ino;
    enqueueRequest(std::move(r), std::to_string(ino), false);
}

static void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::FLUSH;
    r.req = req;
    r.ino = ino;
    enqueueRequest(std::move(r), std::to_string(ino), false);
}

static void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::RELEASE;
    r.req = req;
    r.ino = ino;
    enqueueRequest(std::move(r), std::to_string(ino), false);
}

static void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                       struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op = RequestOp::READDIR;
    r.req = req;
    r.ino = ino;
    r.size = size;
    r.offset = off;
    enqueueRequest(std::move(r), std::to_string(ino)+"/readdir", false);
}

static void ll_access(fuse_req_t req, fuse_ino_t ino, int mask)
{
    FsRequest r;
    r.op = RequestOp::ACCESS;
    r.req = req;
    r.ino = ino;
    r.access_mask = mask;
    enqueueRequest(std::move(r), std::to_string(ino), false);
}

static void ll_statfs(fuse_req_t req, fuse_ino_t ino)
{
    FsRequest r;
    r.op = RequestOp::STATFS;
    r.req = req;
    enqueueRequest(std::move(r), "statfs", false);
}

static void ll_init(void* userdata, struct fuse_conn_info* conn)
{
    (void)userdata;
    (void)conn;
    std::cout << "[Info] FUSE init callback\n";
    g_stats_file << "[Info] FUSE filesystem initialized\n";
}

// -----------------------------------------------------------------------------
// Stats printing
static void print_distribution_stats()
{
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    auto now = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_time).count();

    g_stats_file << "\n=== Distribution Report (Uptime: " << uptime << "s) ===\n";
    for(int i=0; i<NUM_DRIVES; i++) {
        auto &st = g_drive_stats[i];
        auto &tst = g_thread_stats[i];

        uint64_t tot_ops = st.read_ops.load() + st.write_ops.load();
        double avg_wait = 0.0;
        if(tot_ops > 0) {
            avg_wait = double(st.queue_wait_time_us.load()) / double(tot_ops);
        }
        
        uint64_t count_ops = st.operation_latency.count.load();
        double avg_lat = 0.0;
        if(count_ops > 0) {
            avg_lat = double(st.operation_latency.total_us.load()) / double(count_ops);
        }
        
        double utilization = 0.0;
        if(uptime > 0) {
            utilization = 100.0 * double(st.active_time_us.load()) / double(uptime*1000000);
        }

        uint64_t active = tst.active_time_us.load();
        uint64_t idle = tst.idle_time_us.load();
        double thr_util = 0.0;
        if(active+idle > 0) {
            thr_util = 100.0 * active/double(active+idle);
        }
        
        g_stats_file 
            << "\nDrive " << i << ":\n"
            << "  Reads:  " << st.read_ops.load()
            << " (" << st.bytes_read.load()/1024/1024 << " MB)\n"
            << "  Writes: " << st.write_ops.load()
            << " (" << st.bytes_written.load()/1024/1024 << " MB)\n"
            << "  Queue:  length=" << st.queue_length.load()
            << ", avg_wait=" << avg_wait << "us\n"
            << "  Latency: avg=" << avg_lat << "us, max="
            << st.operation_latency.max_us.load() << "us\n"
            << "  Drive Utilization: " << utilization << "%\n"
            << "  Thread Utilization: " << thr_util << "%\n";
    }

    g_stats_file << "\n=== Lock Contention Statistics ===\n";
    for(size_t i=0; i<g_lock_stats.size(); i++) {
        auto &ls = g_lock_stats[i];
        if(ls.contentions.load() > 0) {
            double avg_w = double(ls.total_wait_time_us.load()) / double(ls.contentions.load());
            g_stats_file << "Lock " << i << ": "
                        << ls.contentions.load() << " contentions, "
                        << "Avg wait=" << avg_w << "us, "
                        << "Max wait=" << ls.max_wait_time_us.load() << "us\n";
        }
    }
    g_stats_file << std::endl;
    g_stats_file.flush();
}

// -----------------------------------------------------------------------------
// main
int main(int argc, char* argv[])
{
    g_start_time = std::chrono::steady_clock::now();
    std::cout << "[Info] Starting monitored FUSE filesystem\n";
    g_stats_file.open("fuse_monitor.log");
    if(!g_stats_file.is_open()) {
        std::cerr << "[Error] Could not open fuse_monitor.log\n";
        return 1;
    }

    // parse fuse args
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
    struct fuse_cmdline_opts cmdline_opts;
    memset(&cmdline_opts, 0, sizeof(cmdline_opts));

    if(fuse_parse_cmdline(&args, &cmdline_opts) != 0) {
        fuse_opt_free_args(&args);
        return 1;
    }
    if(fuse_opt_parse(&args, &g_opts, option_spec, nullptr) == -1) {
        std::cerr << "[Error] Failed to parse -o drives=NN.\n";
        fuse_opt_free_args(&args);
        return 1;
    }
    if(g_opts.drives < 1) g_opts.drives = 1;
    if(g_opts.drives > 16) g_opts.drives = 16;
    NUM_DRIVES = g_opts.drives;

    std::cout << "[Info] Using NUM_DRIVES=" << NUM_DRIVES << "\n";
    g_stats_file << "[Config] drives=" << NUM_DRIVES << "\n";

    // fuse ops
    static struct fuse_lowlevel_ops ll_ops;
    memset(&ll_ops, 0, sizeof(ll_ops));
    ll_ops.lookup   = ll_lookup;
    ll_ops.mkdir    = ll_mkdir;
    ll_ops.create   = ll_create;
    ll_ops.unlink   = ll_unlink;
    ll_ops.rmdir    = ll_rmdir;
    ll_ops.rename   = ll_rename;
    ll_ops.open     = ll_open;
    ll_ops.write    = ll_write;
    ll_ops.read     = ll_read;
    ll_ops.getattr  = ll_getattr;
    ll_ops.flush    = ll_flush;
    ll_ops.release  = ll_release;
    ll_ops.readdir  = ll_readdir;
    ll_ops.access   = ll_access;
    ll_ops.statfs   = ll_statfs;
    ll_ops.init     = ll_init;

    // create session
    struct fuse_session* se = fuse_session_new(&args, &ll_ops, sizeof(ll_ops), nullptr);
    if(!se) {
        fuse_opt_free_args(&args);
        return 1;
    }
    if(fuse_set_signal_handlers(se) != 0) {
        fuse_session_destroy(se);
        fuse_opt_free_args(&args);
        return 1;
    }
    if(fuse_session_mount(se, cmdline_opts.mountpoint) != 0) {
        fuse_session_destroy(se);
        fuse_opt_free_args(&args);
        return 1;
    }

    // init root
    g_inodeTable.resize(2);
    auto root = std::make_shared<FsNode>();
    root->isDir = true;
    root->mode = 0755;
    g_inodeTable[FUSE_ROOT_ID] = root;

    // init vectors
    g_drive_stats.resize(NUM_DRIVES);
    g_thread_stats.resize(NUM_DRIVES);
    g_lock_stats.resize(NUM_DRIVES+50);

    g_queueMutex = new std::mutex[NUM_DRIVES];
    g_queueCond = new std::condition_variable[NUM_DRIVES];
    g_queueCondNotFull = new std::condition_variable[NUM_DRIVES];
    g_workers = new std::thread[NUM_DRIVES];
    g_queues = new std::vector<FsRequest>[NUM_DRIVES];
    g_queueCapacity.resize(NUM_DRIVES, MAX_BYTES_PER_DRIVE);

    // create ssd models
    for(int i=0; i<NUM_DRIVES; i++) {
        g_ssdModels.push_back(std::make_unique<SsdModel>(i));
    }

    // spawn workers
    for(int i=0; i<NUM_DRIVES; i++) {
        g_workers[i] = std::thread(workerThreadFunc, i);
        int coreId = i % std::thread::hardware_concurrency();
        pinThreadToCore(g_workers[i], coreId);
    }

    // stats thread
    g_stats_thread = std::thread([]{
        while(!g_stopThreads.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            print_distribution_stats();
        }
    });

    g_stats_file << "[Info] FUSE filesystem ready\n";
    std::cout << "[Info] Low-level FUSE FS started.\n";

    // run fuse main loop
    fuse_session_loop(se);

    // cleanup
    g_stopThreads = true;
    if(g_stats_thread.joinable()) {
        g_stats_thread.join();
    }

    g_stats_file << "\n=== Final Statistics ===\n";
    print_distribution_stats();
    auto end = std::chrono::steady_clock::now();
    auto total_runtime = std::chrono::duration_cast<std::chrono::seconds>(end - g_start_time).count();
    g_stats_file << "Total runtime: " << total_runtime << " seconds\n";

    fuse_session_unmount(se);
    fuse_remove_signal_handlers(se);
    fuse_session_destroy(se);
    fuse_opt_free_args(&args);

    for(int i=0; i<NUM_DRIVES; i++) {
        g_queueCond[i].notify_all();
        g_queueCondNotFull[i].notify_all();
    }
    for(int i=0; i<NUM_DRIVES; i++) {
        if(g_workers[i].joinable()) {
            g_workers[i].join();
        }
    }

    delete[] g_queueMutex;
    delete[] g_queueCond;
    delete[] g_queueCondNotFull;
    delete[] g_workers;
    delete[] g_queues;

    g_stats_file << "[Info] Filesystem unmounted and cleaned up.\n";
    g_stats_file.close();
    std::cout << "[Info] Exiting. Check fuse_monitor.log for performance data.\n";
    return 0;
}