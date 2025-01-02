#define FUSE_USE_VERSION 35

#include <fuse3/fuse_lowlevel.h>
#include <fuse3/fuse_opt.h>    // For fuse_opt_parse
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
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <future>
#include <unistd.h>

// -----------------------------------------------------------------------------
// SSD-like latencies (adjust as needed)
static const int SSD_READ_LATENCY_US  = 1;  // microseconds
static const int SSD_WRITE_LATENCY_US = 1;  // microseconds

// Default number of drives
static int NUM_DRIVES = 4;

// -----------------------------------------------------------------------------
// Custom options
struct options {
    int drives;
};
static struct options g_opts = {4};

static struct fuse_opt option_spec[] = {
    {"drives=%d", offsetof(struct options, drives), 0},
    FUSE_OPT_END
};

// -----------------------------------------------------------------------------
// Increase queue capacity: 4KB blocks × 64 = 256KB per drive
static constexpr size_t BLOCK_SIZE           = 4096;    
static constexpr size_t MAX_BLOCKS_PER_DRIVE = 64;      
static constexpr size_t MAX_BYTES_PER_DRIVE  = MAX_BLOCKS_PER_DRIVE * BLOCK_SIZE; 

// -----------------------------------------------------------------------------
// SSD Model
struct SsdModel {
    void readLatency() {
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_READ_LATENCY_US));
    }
    void writeLatency() {
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_WRITE_LATENCY_US));
    }
};
static std::vector<std::unique_ptr<SsdModel>> g_ssdModels;

// -----------------------------------------------------------------------------
// Inodes and Synchronization
struct InodeLock {
    std::mutex mutex;
    std::atomic<int> ref_count{0};
};

static std::vector<std::unique_ptr<InodeLock>> g_inodeLocks;
static std::mutex g_inodeLocksMutex;

// Forward declare getInodeLock before it's used in InodeLockGuard
static InodeLock* getInodeLock(fuse_ino_t ino);

class InodeLockGuard {
    InodeLock* lock;
public:
    InodeLockGuard(fuse_ino_t ino) {
        lock = getInodeLock(ino);
        lock->mutex.lock();
        lock->ref_count++;
    }
    ~InodeLockGuard() {
        lock->ref_count--;
        lock->mutex.unlock();
    }
};

static InodeLock* getInodeLock(fuse_ino_t ino) {
    std::lock_guard<std::mutex> lock(g_inodeLocksMutex);
    if (ino >= g_inodeLocks.size()) {
        g_inodeLocks.resize(ino + 1);
    }
    if (!g_inodeLocks[ino]) {
        g_inodeLocks[ino] = std::make_unique<InodeLock>();
    }
    return g_inodeLocks[ino].get();
}

// -----------------------------------------------------------------------------
// Request Operation Types
enum class RequestOp {
    LOOKUP,
    MKDIR,
    CREATE,
    UNLINK,
    RMDIR,
    RENAME,
    WRITE,
    READ,
    GETATTR,
    FLUSH,
    RELEASE,
    READDIR,
    ACCESS,
    STATFS
};

// -----------------------------------------------------------------------------
// FsRequest Structure
struct FsRequest {
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
    std::promise<void> completion_promise;  // Direct member

    FsRequest() = default;
    FsRequest(FsRequest&&) = default;
    FsRequest& operator=(FsRequest&&) = default;
    
    FsRequest(const FsRequest&) = delete;
    FsRequest& operator=(const FsRequest&) = delete;
};

// -----------------------------------------------------------------------------
// Forward declarations of process functions
static void processLookup(FsRequest& r);
static void processMkdir(FsRequest& r);
static void processCreate(FsRequest& r);
static void processUnlink(FsRequest& r);
static void processRmdir(FsRequest& r);
static void processRename(FsRequest& r);
static void processWrite(FsRequest& r);
static void processRead(FsRequest& r);
static void processGetattr(FsRequest& r);
static void processFlush(FsRequest& r);
static void processRelease(FsRequest& r);
static void processReaddir(FsRequest& r);
static void processAccess(FsRequest& r);
static void processStatfs(FsRequest& r);

// -----------------------------------------------------------------------------
// Forward declarations of FUSE callbacks
static void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name);
static void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode);
static void ll_create(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode, struct fuse_file_info *fi);
static void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name);
static void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name);
static void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
                      fuse_ino_t newparent, const char *newname, unsigned int flags);
static void ll_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void ll_write(fuse_req_t req, fuse_ino_t ino, const char *buf,
                     size_t size, off_t off, struct fuse_file_info *fi);
static void ll_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                    struct fuse_file_info *fi);
static void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi);
static void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                       struct fuse_file_info *fi);
static void ll_access(fuse_req_t req, fuse_ino_t ino, int mask);
static void ll_statfs(fuse_req_t req, fuse_ino_t ino);

// -----------------------------------------------------------------------------
// Concurrency structures
static std::mutex* g_queueMutex           = nullptr;
static std::condition_variable* g_queueCond = nullptr;        
static std::condition_variable* g_queueCondNotFull = nullptr; 

static std::thread* g_workers     = nullptr;
static std::atomic<bool> g_stopThreads{false};
static std::vector<FsRequest>* g_queues = nullptr;  
static std::vector<size_t> g_queueCapacity;          

// -----------------------------------------------------------------------------
// Inode structures
struct FsNode {
    bool isDir;
    mode_t mode;
    std::vector<uint8_t> fileData;
    std::map<std::string, fuse_ino_t> children; // For directories
};

static std::vector<std::shared_ptr<FsNode>> g_inodeTable;
static std::mutex g_inodeTableMutex;
static fuse_ino_t g_nextIno = 2; // inode 1 is root

// -----------------------------------------------------------------------------
// Inode management
static fuse_ino_t createInode(bool isDir, mode_t mode) {
    std::lock_guard<std::mutex> lock(g_inodeTableMutex);
    fuse_ino_t ino = g_nextIno++;
    auto node = std::make_shared<FsNode>();
    node->isDir = isDir;
    node->mode  = mode;
    g_inodeTable.resize(std::max<size_t>(g_inodeTable.size(), ino + 1));
    g_inodeTable[ino] = node;
    return ino;
}

static std::shared_ptr<FsNode> getNode(fuse_ino_t ino) {
    std::lock_guard<std::mutex> lock(g_inodeTableMutex);
    if (ino < g_inodeTable.size() && g_inodeTable[ino]) {
        return g_inodeTable[ino];
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Offset-based hash function for read/write
static uint64_t fnv1a_hash_offset(fuse_ino_t ino, off_t offset) {
    std::string combined = std::to_string(ino) + "-" + std::to_string(offset / BLOCK_SIZE);
    uint64_t hash = 1469598103934665603ULL;
    for (char c : combined) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Basic FNV-1a if no offset used
static uint64_t fnv1a_hash_string(const std::string &s) {
    uint64_t hash = 1469598103934665603ULL;
    for (char c : s) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// -----------------------------------------------------------------------------
// Bounded enqueueRequest
static void enqueueRequest(FsRequest&& r, const std::string &uniqueKey, bool offsetBased = false) {
    uint64_t h;
    if (!offsetBased) {
        // Use basic string hashing
        h = fnv1a_hash_string(uniqueKey);
    } else {
        // offset-based for read/write
        h = fnv1a_hash_offset(r.ino, r.offset);
    }
    int queueIdx = h % NUM_DRIVES;

    size_t bytesNeeded = 0;
    if (r.op == RequestOp::WRITE || r.op == RequestOp::READ) {
        size_t dataSize = r.data.size();
        size_t blocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bytesNeeded = blocks * BLOCK_SIZE;
    }

    {
        std::unique_lock<std::mutex> lock(g_queueMutex[queueIdx]);
        g_queueCondNotFull[queueIdx].wait(lock, [queueIdx, bytesNeeded] {
            return (g_queueCapacity[queueIdx] >= bytesNeeded) || g_stopThreads.load();
        });
        if (g_stopThreads.load()) {
            return;
        }
        g_queues[queueIdx].push_back(std::move(r));
        g_queueCapacity[queueIdx] -= bytesNeeded;
    }
    g_queueCond[queueIdx].notify_one();
}

// -----------------------------------------------------------------------------
// Helper
static void fillEntryParam(fuse_entry_param &e, fuse_ino_t ino, bool isDir) {
    memset(&e, 0, sizeof(e));
    e.ino           = ino;
    e.generation    = 1;
    e.attr_timeout  = 1.0;
    e.entry_timeout = 1.0;
    
    auto node = getNode(ino);
    if (!node) return;
    
    e.attr.st_ino = ino;
    if (isDir) {
        e.attr.st_mode = S_IFDIR | node->mode;
        e.attr.st_nlink = 2 + node->children.size();
    } else {
        e.attr.st_mode = S_IFREG | node->mode;
        e.attr.st_size = node->fileData.size();
        e.attr.st_nlink = 1;
    }
    e.attr.st_uid = getuid();
    e.attr.st_gid = getgid();
}

// -----------------------------------------------------------------------------
// Worker
static void workerThreadFunc(int driveIndex) {
    auto &ssd = *(g_ssdModels[driveIndex]);

    while (!g_stopThreads.load()) {
        std::vector<FsRequest> localRequests;
        {
            std::unique_lock<std::mutex> lock(g_queueMutex[driveIndex]);
            g_queueCond[driveIndex].wait(lock, [driveIndex] {
                return (!g_queues[driveIndex].empty() || g_stopThreads.load());
            });
            if (g_stopThreads.load()) break;

            localRequests = std::move(g_queues[driveIndex]);
            g_queues[driveIndex].clear();
        }

        for (auto &req : localRequests) {
            size_t bytesUsed = 0;
            if (req.op == RequestOp::WRITE || req.op == RequestOp::READ) {
                size_t dataSize = req.data.size();
                size_t blocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
                bytesUsed = blocks * BLOCK_SIZE;
            }

            try {
                switch (req.op) {
                    case RequestOp::LOOKUP:   processLookup(req);   break;
                    case RequestOp::MKDIR:    processMkdir(req);    break;
                    case RequestOp::CREATE:   processCreate(req);   break;
                    case RequestOp::UNLINK:   processUnlink(req);   break;
                    case RequestOp::RMDIR:    processRmdir(req);    break;
                    case RequestOp::RENAME:   processRename(req);   break;
                    case RequestOp::WRITE:
                        ssd.writeLatency();
                        processWrite(req);
                        break;
                    case RequestOp::READ:
                        ssd.readLatency();
                        processRead(req);
                        break;
                    case RequestOp::GETATTR:  processGetattr(req);  break;
                    case RequestOp::FLUSH:    processFlush(req);    break;
                    case RequestOp::RELEASE:  processRelease(req);  break;
                    case RequestOp::READDIR:  processReaddir(req);  break;
                    case RequestOp::ACCESS:   processAccess(req);   break;
                    case RequestOp::STATFS:   processStatfs(req);   break;
                }
            } catch (const std::exception &e) {
                std::cerr << "[Error] exception: " << e.what() << "\n";
                fuse_reply_err(req.req, EIO);
            }

            req.completion_promise.set_value();

            if (bytesUsed > 0) {
                std::unique_lock<std::mutex> lock(g_queueMutex[driveIndex]);
                g_queueCapacity[driveIndex] += bytesUsed;
                g_queueCondNotFull[driveIndex].notify_all();
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Process functions
static void processLookup(FsRequest &r) {
    InodeLockGuard lock(r.parent);
    auto parentNode = getNode(r.parent);
    if (!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto it = parentNode->children.find(r.name);
    if (it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    fuse_ino_t childIno = it->second;
    auto childNode = getNode(childIno);
    if (!childNode) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    fuse_entry_param e;
    fillEntryParam(e, childIno, childNode->isDir);
    fuse_reply_entry(r.req, &e);
}

static void processMkdir(FsRequest &r) {
    InodeLockGuard lock(r.parent);
    auto parentNode = getNode(r.parent);
    if (!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    if (parentNode->children.find(r.name) != parentNode->children.end()) {
        fuse_reply_err(r.req, EEXIST);
        return;
    }
    fuse_ino_t newIno = createInode(true, r.mode);
    parentNode->children[r.name] = newIno;
    fuse_entry_param e;
    fillEntryParam(e, newIno, true);
    fuse_reply_entry(r.req, &e);
}

static void processCreate(FsRequest &r) {
    InodeLockGuard lock(r.parent);
    auto parentNode = getNode(r.parent);
    if (!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    if (parentNode->children.find(r.name) != parentNode->children.end()) {
        fuse_reply_err(r.req, EEXIST);
        return;
    }
    fuse_ino_t newIno = createInode(false, r.mode);
    parentNode->children[r.name] = newIno;

    fuse_entry_param e;
    fillEntryParam(e, newIno, false);
    struct fuse_file_info fi;
    memset(&fi, 0, sizeof(fi));
    fi.fh = newIno;
    fuse_reply_create(r.req, &e, &fi);
}

static void processUnlink(FsRequest &r) {
    InodeLockGuard lock(r.parent);
    auto parentNode = getNode(r.parent);
    if (!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto it = parentNode->children.find(r.name);
    if (it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto childNode = getNode(it->second);
    if (!childNode || childNode->isDir) {
        fuse_reply_err(r.req, EISDIR);
        return;
    }
    parentNode->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processRmdir(FsRequest &r) {
    InodeLockGuard lock(r.parent);
    auto parentNode = getNode(r.parent);
    if (!parentNode || !parentNode->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto it = parentNode->children.find(r.name);
    if (it == parentNode->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto childNode = getNode(it->second);
    if (!childNode || !childNode->isDir) {
        fuse_reply_err(r.req, ENOTDIR);
        return;
    }
    if (!childNode->children.empty()) {
        fuse_reply_err(r.req, ENOTEMPTY);
        return;
    }
    parentNode->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processRename(FsRequest &r) {
    InodeLockGuard oldParentLock(r.parent);
    InodeLockGuard newParentLock(r.newparent);
    auto oldParent = getNode(r.parent);
    if (!oldParent || !oldParent->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto it = oldParent->children.find(r.name);
    if (it == oldParent->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    fuse_ino_t oldIno = it->second;
    auto newParent = getNode(r.newparent);
    if (!newParent || !newParent->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto existing = newParent->children.find(r.newname);
    if (existing != newParent->children.end()) {
        auto existingNode = getNode(existing->second);
        if (existingNode && existingNode->isDir && !existingNode->children.empty()) {
            fuse_reply_err(r.req, ENOTEMPTY);
            return;
        }
        newParent->children.erase(existing);
    }
    newParent->children[r.newname] = oldIno;
    oldParent->children.erase(it);
    fuse_reply_err(r.req, 0);
}

static void processWrite(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    size_t endPos = r.offset + r.data.size();
    if (endPos > node->fileData.size()) {
        node->fileData.resize(endPos);
    }
    std::memcpy(&node->fileData[r.offset], r.data.data(), r.data.size());
    fuse_reply_write(r.req, r.data.size());
}

static void processRead(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    size_t filesize = node->fileData.size();
    if ((size_t)r.offset < filesize) {
        size_t bytesToRead = r.size;
        if (r.offset + bytesToRead > filesize) {
            bytesToRead = filesize - r.offset;
        }
        fuse_reply_buf(r.req, (const char*)(node->fileData.data() + r.offset), bytesToRead);
    } else {
        fuse_reply_buf(r.req, nullptr, 0);
    }
}

static void processGetattr(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    auto node = getNode(r.ino);
    if (!node) {
        if (r.ino == FUSE_ROOT_ID) {
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
    if (node->isDir) {
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
    InodeLockGuard lock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processRelease(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processReaddir(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    auto node = getNode(r.ino);
    if (!node || !node->isDir) {
        fuse_reply_err(r.req, ENOTDIR);
        return;
    }
    char *buf = (char*)malloc(r.size);
    if (!buf) {
        fuse_reply_err(r.req, ENOMEM);
        return;
    }
    size_t bpos = 0;
    auto addDirEntry = [&](fuse_ino_t e_ino, const char *name) {
        struct stat st{};
        st.st_ino = e_ino;
        auto en = getNode(e_ino);
        if (en) {
            st.st_mode = en->isDir ? (S_IFDIR|en->mode) : (S_IFREG|en->mode);
        }
        size_t entsize = fuse_add_direntry(r.req, buf + bpos, r.size - bpos,
                                           name, &st, r.offset+1);
        if (entsize > 0 && bpos + entsize <= r.size) {
            bpos += entsize;
        }
    };
    if (r.offset == 0) {
        addDirEntry(r.ino, ".");
        addDirEntry(FUSE_ROOT_ID, "..");
        for (const auto &ch : node->children) {
            addDirEntry(ch.second, ch.first.c_str());
        }
    }
    fuse_reply_buf(r.req, buf, bpos);
    free(buf);
}

static void processAccess(FsRequest &r) {
    InodeLockGuard lock(r.ino);
    fuse_reply_err(r.req, 0);
}

static void processStatfs(FsRequest &r) {
    struct statvfs st{};
    st.f_bsize   = 4096;
    st.f_frsize  = 4096;
    st.f_blocks  = 1024 * 1024;
    st.f_bfree   = 1024 * 1024;
    st.f_bavail  = 1024 * 1024;
    st.f_files   = 100000;
    st.f_ffree   = 100000;
    st.f_favail  = 100000;
    st.f_fsid    = 1234;
    st.f_flag    = 0;
    st.f_namemax = 255;
    fuse_reply_statfs(r.req, &st);
}

// -----------------------------------------------------------------------------
// FUSE Callbacks
static void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::LOOKUP;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(parent) + "/" + name);
    fut.wait();
}

static void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode) {
    FsRequest r;
    r.op     = RequestOp::MKDIR;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    r.mode   = mode;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(parent) + "/" + name);
    fut.wait();
}

static void ll_create(fuse_req_t req, fuse_ino_t parent, const char *name,
                      mode_t mode, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op     = RequestOp::CREATE;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    r.mode   = mode;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(parent) + "/" + name);
    fut.wait();
}

static void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::UNLINK;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(parent) + "/" + name);
    fut.wait();
}

static void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::RMDIR;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(parent) + "/" + name);
    fut.wait();
}

static void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
                      fuse_ino_t newparent, const char *newname, unsigned int flags) {
    FsRequest r;
    r.op          = RequestOp::RENAME;
    r.req         = req;
    r.parent      = parent;
    r.name        = name;
    r.newparent   = newparent;
    r.newname     = newname;
    r.rename_flags= flags;
    std::string key = std::to_string(parent) + "/" + name + "->" +
                      std::to_string(newparent) + "/" + newname;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), key);
    fut.wait();
}

static void ll_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    fi->fh = ino;
    fuse_reply_open(req, fi);
}

static void ll_write(fuse_req_t req, fuse_ino_t ino, const char *buf,
                     size_t size, off_t off, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op     = RequestOp::WRITE;
    r.req    = req;
    r.ino    = ino;
    r.offset = off;
    r.data.assign(buf, buf + size);
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), "", true /*offsetBased*/);
    fut.wait();
}

static void ll_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                    struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op     = RequestOp::READ;
    r.req    = req;
    r.ino    = ino;
    r.size   = size;
    r.offset = off;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), "", true /*offsetBased*/);
    fut.wait();
}

static void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::GETATTR;
    r.req = req;
    r.ino = ino;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(ino));
    fut.wait();
}

static void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::FLUSH;
    r.req = req;
    r.ino = ino;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(ino));
    fut.wait();
}

static void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::RELEASE;
    r.req = req;
    r.ino = ino;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(ino));
    fut.wait();
}

static void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                       struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op     = RequestOp::READDIR;
    r.req    = req;
    r.ino    = ino;
    r.size   = size;
    r.offset = off;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(ino) + "/readdir");
    fut.wait();
}

static void ll_access(fuse_req_t req, fuse_ino_t ino, int mask) {
    FsRequest r;
    r.op     = RequestOp::ACCESS;
    r.req    = req;
    r.ino    = ino;
    r.access_mask = mask;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), std::to_string(ino));
    fut.wait();
}

static void ll_statfs(fuse_req_t req, fuse_ino_t ino) {
    FsRequest r;
    r.op  = RequestOp::STATFS;
    r.req = req;
    auto fut = r.completion_promise.get_future();
    enqueueRequest(std::move(r), "statfs");
    fut.wait();
}

// Root directory lookup handler
static void ll_lookup_root(fuse_req_t req, fuse_ino_t parent, const char *name) {
    if (parent == FUSE_ROOT_ID && strcmp(name, ".") == 0) {
        fuse_entry_param e{};
        e.ino = FUSE_ROOT_ID;
        e.attr.st_mode = S_IFDIR | 0755;
        e.attr.st_ino  = FUSE_ROOT_ID;
        e.attr.st_nlink = 2;
        e.attr_timeout = 1.0;
        e.entry_timeout = 1.0;
        fuse_reply_entry(req, &e);
        return;
    }
    ll_lookup(req, parent, name);
}

// Optional init callback
static void ll_init(void *userdata, struct fuse_conn_info *conn) {
    (void)userdata;
    (void)conn;
    std::cout << "[Info] FUSE init callback\n";
}

int main(int argc, char *argv[]) {
    std::cout << "[Info] Starting main()\n";
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
    struct fuse_cmdline_opts cmdline_opts;
    
    if (fuse_parse_cmdline(&args, &cmdline_opts) != 0) {
        fuse_opt_free_args(&args);
        return 1;
    }
    
    if (fuse_opt_parse(&args, &g_opts, option_spec, nullptr) == -1) {
        std::cerr << "[Error] Failed to parse custom -o drives=NN option.\n";
        fuse_opt_free_args(&args);
        return 1;
    }
    
    if (g_opts.drives < 1) g_opts.drives = 1;
    if (g_opts.drives > 16) g_opts.drives = 16;
    NUM_DRIVES = g_opts.drives;
    std::cout << "[Info] Using NUM_DRIVES=" << NUM_DRIVES << std::endl;

    static struct fuse_lowlevel_ops ll_ops;
    memset(&ll_ops, 0, sizeof(ll_ops));
    ll_ops.lookup   = ll_lookup_root;
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

    struct fuse_session *se = fuse_session_new(&args, &ll_ops, sizeof(ll_ops), nullptr);
    if (!se) {
        fuse_opt_free_args(&args);
        return 1;
    }
    if (fuse_set_signal_handlers(se) != 0) {
        fuse_session_destroy(se);
        fuse_opt_free_args(&args);
        return 1;
    }
    if (fuse_session_mount(se, cmdline_opts.mountpoint) != 0) {
        fuse_session_destroy(se);
        fuse_opt_free_args(&args);
        return 1;
    }

    // Initialize root inode
    {
        std::lock_guard<std::mutex> lk(g_inodeTableMutex);
        g_inodeTable.resize(2);  // Space for root inode
        auto rootNode = std::make_shared<FsNode>();
        rootNode->isDir = true;
        rootNode->mode  = 0755;
        g_inodeTable[FUSE_ROOT_ID] = rootNode;
    }

    // Initialize concurrency structures
    g_queueMutex      = new std::mutex[NUM_DRIVES];
    g_queueCond       = new std::condition_variable[NUM_DRIVES];
    g_queueCondNotFull= new std::condition_variable[NUM_DRIVES];
    g_workers         = new std::thread[NUM_DRIVES];
    g_queues          = new std::vector<FsRequest>[NUM_DRIVES];
    g_queueCapacity.resize(NUM_DRIVES, MAX_BYTES_PER_DRIVE);

    // Create SSD models
    for (int i=0; i<NUM_DRIVES; i++) {
        g_ssdModels.push_back(std::make_unique<SsdModel>());
    }

    // Start worker threads
    for (int i=0; i<NUM_DRIVES; i++) {
        g_workers[i] = std::thread(workerThreadFunc, i);
    }

    std::cout << "[Info] Low-level FUSE FS started.\n";
    fuse_session_loop(se);

    // Cleanup
    fuse_session_unmount(se);
    fuse_remove_signal_handlers(se);
    fuse_session_destroy(se);
    fuse_opt_free_args(&args);

    g_stopThreads.store(true);
    for (int i=0; i<NUM_DRIVES; i++) {
        g_queueCond[i].notify_all();
        g_queueCondNotFull[i].notify_all();
    }
    for (int i=0; i<NUM_DRIVES; i++) {
        if (g_workers[i].joinable()) {
            g_workers[i].join();
        }
    }

    delete[] g_queueMutex;
    delete[] g_queueCond;
    delete[] g_queueCondNotFull;
    delete[] g_workers;
    delete[] g_queues;

    std::cout << "[Info] Exiting.\n";
    return 0;
}
