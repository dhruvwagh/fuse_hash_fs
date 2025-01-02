#define FUSE_USE_VERSION 35
#define DEBUG_LOG(msg, ...) do { printf("[DEBUG] " msg "\n", ##__VA_ARGS__); } while(0)

#include <fuse3/fuse_lowlevel.h>
#include <fuse3/fuse_opt.h>    // For fuse_opt_parse
#include <fcntl.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>

// Add these two reference counting functions that match FUSE's expected callbacks
static inline void fuse_req_ref(fuse_req_t req) {
    fuse_req_interrupt_func(req, NULL, NULL);
}

static inline void fuse_req_unref(fuse_req_t req) {
    (void)req;  // Unused in this simple implementation
}
// SSD-like latencies (microseconds)
static const int SSD_READ_LATENCY_US  = 100;  
static const int SSD_WRITE_LATENCY_US = 500;  

// Default number of drives if user doesn't specify
static int NUM_DRIVES = 4;

// -----------------------------------------------------------------------------
// We'll parse a custom option "-o drives=N" using fuse_opt_parse
// -----------------------------------------------------------------------------
struct options {
    int drives;
};
static struct options g_opts = {4};

static struct fuse_opt option_spec[] = {
    {"drives=%d", offsetof(struct options, drives), 0},
    FUSE_OPT_END
};

// -----------------------------------------------------------------------------
// A small SSD model that inserts read/write latencies
// -----------------------------------------------------------------------------
struct SsdModel {
    void readLatency() {
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_READ_LATENCY_US));
    }
    void writeLatency() {
        std::this_thread::sleep_for(std::chrono::microseconds(SSD_WRITE_LATENCY_US));
    }
};

// We'll have one SsdModel per "drive"
static std::vector<std::unique_ptr<SsdModel>> g_ssdModels;

// -----------------------------------------------------------------------------
// Multi-drive concurrency structures
// -----------------------------------------------------------------------------
static std::mutex* g_queueMutex        = nullptr;
static std::condition_variable* g_queueCond = nullptr;
static std::thread* g_workers          = nullptr;
static std::atomic<bool> g_stopThreads{false};

// Forward declaration of FsRequest
struct FsRequest;
static std::vector<FsRequest>* g_queues = nullptr;  // array[NUM_DRIVES] of vectors

// -----------------------------------------------------------------------------
// Inode + File/Dir Data Structures
// -----------------------------------------------------------------------------
struct FsNode {
    bool isDir;
    mode_t mode;
    std::vector<uint8_t> fileData; 
    std::map<std::string, fuse_ino_t> children; // For directories
};

// Global inode table
static std::vector<std::shared_ptr<FsNode>> g_inodeTable;
static std::mutex g_inodeTableMutex;
static fuse_ino_t g_nextIno = 2; // inode 1 is root

static fuse_ino_t createInode(bool isDir, mode_t mode) {
    std::lock_guard<std::mutex> lock(g_inodeTableMutex);
    fuse_ino_t ino = g_nextIno++;
    auto node = std::make_shared<FsNode>();
    node->isDir = isDir;
    node->mode  = mode;
    g_inodeTable.resize(std::max<size_t>(g_inodeTable.size(), ino+1));
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
// FNV-1a hash function for distributing requests
// -----------------------------------------------------------------------------
static uint64_t fnv1a_hash(const std::string &str) {
    uint64_t hash = 1469598103934665603ULL;
    for (char c : str) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// -----------------------------------------------------------------------------
// RequestOp + FsRequest
// -----------------------------------------------------------------------------
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
    ACCESS,   // Make sure this is here
    STATFS    // Make sure this is here

};

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
};

// We'll define worker logic next
static void processLookup(const FsRequest &r);
static void processMkdir(const FsRequest &r);
static void processCreate(const FsRequest &r);
static void processUnlink(const FsRequest &r);
static void processRmdir(const FsRequest &r);
static void processRename(const FsRequest &r);
static void processWrite(const FsRequest &r);
static void processRead(const FsRequest &r);
static void processGetattr(const FsRequest &r);
static void processFlush(const FsRequest &r);
static void processRelease(const FsRequest &r);
static void processReaddir(const FsRequest &r);
// Add these to your existing forward declarations
static void processAccess(const FsRequest &r);
static void processStatfs(const FsRequest &r);

// -----------------------------------------------------------------------------
static void workerThreadFunc(int driveIndex) {
    printf("[DEBUG] Worker %d starting\n", driveIndex);
    auto &ssd = *(g_ssdModels[driveIndex]);
    
    while (!g_stopThreads.load()) {
        std::vector<FsRequest> localRequests;
        {
            std::unique_lock<std::mutex> lock(g_queueMutex[driveIndex]);
            g_queueCond[driveIndex].wait(lock, [driveIndex]{
                return (!g_queues[driveIndex].empty() || g_stopThreads.load());
            });
            if (g_stopThreads.load()) break;
            localRequests.swap(g_queues[driveIndex]);
        }
        
        printf("[DEBUG] Worker %d processing %zu requests\n", driveIndex, localRequests.size());
        
        for (auto &req : localRequests) {
            printf("[DEBUG] Processing op=%d\n", static_cast<int>(req.op));
            try {
                switch (req.op) {
                    case RequestOp::LOOKUP:
                        printf("[DEBUG] LOOKUP request\n");
                        processLookup(req);
                        break;
                    case RequestOp::MKDIR:
                        printf("[DEBUG] MKDIR request\n");
                        processMkdir(req);
                        break;
                    case RequestOp::CREATE:
                        printf("[DEBUG] CREATE request\n");
                        processCreate(req);
                        break;
                    case RequestOp::UNLINK:
                        printf("[DEBUG] UNLINK request\n");
                        processUnlink(req);
                        break;
                    case RequestOp::RMDIR:
                        printf("[DEBUG] RMDIR request\n");
                        processRmdir(req);
                        break;
                    case RequestOp::RENAME:
                        printf("[DEBUG] RENAME request\n");
                        processRename(req);
                        break;
                    case RequestOp::WRITE:
                        printf("[DEBUG] WRITE request\n");
                        processWrite(req);
                        break;
                    case RequestOp::READ:
                        printf("[DEBUG] READ request\n");
                        processRead(req);
                        break;
                    case RequestOp::GETATTR:
                        printf("[DEBUG] GETATTR request\n");
                        processGetattr(req);
                        break;
                    case RequestOp::FLUSH:
                        printf("[DEBUG] FLUSH request\n");
                        processFlush(req);
                        break;
                    case RequestOp::RELEASE:
                        printf("[DEBUG] RELEASE request\n");
                        processRelease(req);
                        break;
                    case RequestOp::READDIR:
                        printf("[DEBUG] READDIR request\n");
                        processReaddir(req);
                        break;
                    case RequestOp::ACCESS:
                        printf("[DEBUG] ACCESS request\n");
                        processAccess(req);
                        break;
                    case RequestOp::STATFS:
                        printf("[DEBUG] STATFS request\n");
                        processStatfs(req);
                        break;
                    default:
                        printf("[DEBUG] Unknown operation %d\n", static_cast<int>(req.op));
                        fuse_reply_err(req.req, ENOSYS);
                        break;
                }
            } catch (const std::exception &e) {
                printf("[ERROR] Exception processing request: %s\n", e.what());
                fuse_reply_err(req.req, EIO);
            } catch (...) {
                printf("[ERROR] Unknown exception processing request\n");
                fuse_reply_err(req.req, EIO);
            }
        }
    }
    printf("[DEBUG] Worker %d exiting\n", driveIndex);
}

// -----------------------------------------------------------------------------
static void enqueueRequest(const FsRequest &r, const std::string &uniqueKey) {
    uint64_t h = fnv1a_hash(uniqueKey);
    int queueIdx = h % NUM_DRIVES;
    DEBUG_LOG("Enqueueing request op=%d to queue %d", static_cast<int>(r.op), queueIdx);
    {
        std::lock_guard<std::mutex> lock(g_queueMutex[queueIdx]);
        g_queues[queueIdx].push_back(r);
    }
    g_queueCond[queueIdx].notify_one();
}

// -----------------------------------------------------------------------------
// Helpers: fill fuse_entry_param
// -----------------------------------------------------------------------------
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
    e.attr.st_uid = 0;
    e.attr.st_gid = 0;
}

// -----------------------------------------------------------------------------
// Worker logic: each operation
// -----------------------------------------------------------------------------
static void processLookup(const FsRequest &r) {
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

static void processMkdir(const FsRequest &r) {
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

static void processCreate(const FsRequest &r) {
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

static void processUnlink(const FsRequest &r) {
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

static void processRmdir(const FsRequest &r) {
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

static void processRename(const FsRequest &r) {
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
    // Overwrite existing
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

static void processWrite(const FsRequest &r) {
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    // Simulate SSD write latency
    int driveIndex = (int)(fnv1a_hash(std::to_string(r.ino)) % NUM_DRIVES);
    g_ssdModels[driveIndex]->writeLatency();

    size_t endPos = r.offset + r.data.size();
    if (endPos > node->fileData.size()) {
        node->fileData.resize(endPos);
    }
    std::memcpy(&node->fileData[r.offset], r.data.data(), r.data.size());
    fuse_reply_write(r.req, r.data.size());
}

static void processRead(const FsRequest &r) {
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    // Simulate SSD read latency
    int driveIndex = (int)(fnv1a_hash(std::to_string(r.ino)) % NUM_DRIVES);
    g_ssdModels[driveIndex]->readLatency();

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

static void processGetattr(const FsRequest &r) {
    auto node = getNode(r.ino);
    if (!node) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    struct stat st;
    memset(&st, 0, sizeof(st));
    st.st_ino = r.ino;
    if (node->isDir) {
        st.st_mode = S_IFDIR | node->mode;
        st.st_nlink = 2 + node->children.size();
    } else {
        st.st_mode = S_IFREG | node->mode;
        st.st_size = node->fileData.size();
        st.st_nlink = 1;
    }
    st.st_uid = 0;
    st.st_gid = 0;
    fuse_reply_attr(r.req, &st, 1.0);
}

static void processFlush(const FsRequest &r) {
    fuse_reply_err(r.req, 0);
}

static void processRelease(const FsRequest &r) {
    fuse_reply_err(r.req, 0);
}

static void processReaddir(const FsRequest &r) {
    auto node = getNode(r.ino);
    if (!node || !node->isDir) {
        fuse_reply_err(r.req, ENOTDIR);
        return;
    }
    // Build the direntries
    char *buf = (char*)malloc(r.size);
    if (!buf) {
        fuse_reply_err(r.req, ENOMEM);
        return;
    }
    size_t bpos = 0;
    auto addDirEntry = [&](fuse_ino_t e_ino, const char *name){
        struct stat st;
        memset(&st, 0, sizeof(st));
        st.st_ino = e_ino;
        auto en = getNode(e_ino);
        if (en) {
            st.st_mode = en->isDir ? (S_IFDIR|en->mode) : (S_IFREG|en->mode);
        }
        size_t entsize = fuse_add_direntry(r.req, buf + bpos, r.size - bpos, name, &st, r.offset+1);
        if (entsize > 0 && bpos + entsize <= r.size) {
            bpos += entsize;
        }
    };
    if (r.offset == 0) {
        addDirEntry(r.ino, ".");
        addDirEntry(1, "..");
        for (auto &ch : node->children) {
            addDirEntry(ch.second, ch.first.c_str());
        }
    }
    fuse_reply_buf(r.req, buf, bpos);
    free(buf);
}
static void processAccess(const FsRequest &r) {
    // Always allow access for now
    fuse_reply_err(r.req, 0);
}

static void processStatfs(const FsRequest &r) {
    struct statvfs st{};
    st.f_bsize   = 4096;
    st.f_frsize  = 4096;
    st.f_blocks  = 1024*1024;
    st.f_bfree   = 1024*1024;
    st.f_bavail  = 1024*1024;
    st.f_files   = 100000;
    st.f_ffree   = 100000;
    st.f_favail  = 100000;
    st.f_fsid    = 1234;
    st.f_flag    = 0;
    st.f_namemax = 255;
    fuse_reply_statfs(r.req, &st);
}

// -----------------------------------------------------------------------------
// Low-level FUSE callbacks
// -----------------------------------------------------------------------------
static void ll_lookup_root(fuse_req_t req, fuse_ino_t parent, const char *name);

static void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::LOOKUP;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
}

static void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode) {
    FsRequest r;
    r.op     = RequestOp::MKDIR;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    r.mode   = mode;
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
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
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
}

static void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::UNLINK;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
}

static void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name) {
    FsRequest r;
    r.op     = RequestOp::RMDIR;
    r.req    = req;
    r.parent = parent;
    r.name   = name;
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
}

static void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
                      fuse_ino_t newparent, const char *newname, unsigned int flags) {
    FsRequest r;
    r.op        = RequestOp::RENAME;
    r.req       = req;
    r.parent    = parent;
    r.name      = name;
    r.newparent = newparent;
    r.newname   = newname;
    r.rename_flags = flags;
    enqueueRequest(r, std::to_string(parent)+"/"+r.name);
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
    r.data.assign(buf, buf+size);
    enqueueRequest(r, std::to_string(ino));
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
    enqueueRequest(r, std::to_string(ino));
}

static void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::GETATTR;
    r.req = req;
    r.ino = ino;
    enqueueRequest(r, std::to_string(ino));
}

static void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::FLUSH;
    r.req = req;
    r.ino = ino;
    enqueueRequest(r, std::to_string(ino));
}

static void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op  = RequestOp::RELEASE;
    r.req = req;
    r.ino = ino;
    enqueueRequest(r, std::to_string(ino));
}

static void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size,
                       off_t off, struct fuse_file_info *fi) {
    (void)fi;
    FsRequest r;
    r.op     = RequestOp::READDIR;
    r.req    = req;
    r.ino    = ino;
    r.size   = size;
    r.offset = off;
    enqueueRequest(r, std::to_string(ino)+"/readdir");
}

// Intercept root "." lookups
static void ll_lookup_root(fuse_req_t req, fuse_ino_t parent, const char *name) {
    if (parent == FUSE_ROOT_ID && strcmp(name, ".") == 0) {
        // synthetic
        fuse_entry_param e;
        memset(&e, 0, sizeof(e));
        e.ino = 1; 
        e.attr.st_mode = S_IFDIR | 0755;
        e.attr.st_ino  = 1;
        e.attr.st_nlink = 2;
        e.attr_timeout  = 1.0;
        e.entry_timeout = 1.0;
        fuse_reply_entry(req, &e);
    } else {
        ll_lookup(req, parent, name);
    }
}
static void ll_access(fuse_req_t req, fuse_ino_t ino, int mask) {
    FsRequest r;
    r.op          = RequestOp::ACCESS;
    r.req         = req;
    r.ino         = ino;
    r.access_mask = mask;
    enqueueRequest(r, std::to_string(ino));
}

static void ll_statfs(fuse_req_t req, fuse_ino_t ino) {
    FsRequest r;
    r.op  = RequestOp::STATFS;
    r.req = req;
    enqueueRequest(r, "statfs");
}

// Low-level ops struct
static struct fuse_lowlevel_ops ll_ops;

int main(int argc, char *argv[])
{
    // 1. Parse standard FUSE args AND our custom -o drives=NN
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
    struct fuse_cmdline_opts cmdline_opts;
    if (fuse_parse_cmdline(&args, &cmdline_opts) != 0) {
        fuse_opt_free_args(&args);
        return 1;
    }

    // 2. Parse custom option: drives=NN
    if (fuse_opt_parse(&args, &g_opts, option_spec, nullptr) == -1) {
        std::cerr << "Failed to parse custom -o drives=NN option.\n";
        fuse_opt_free_args(&args);
        return 1;
    }
    if (g_opts.drives < 1) g_opts.drives = 1;
    if (g_opts.drives > 16) g_opts.drives = 16;
    NUM_DRIVES = g_opts.drives;  // set global
    std::cout << "[Info] Using NUM_DRIVES=" << NUM_DRIVES << std::endl;

    // Create fuse session
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

    // Initialize root inode=1
    {
        std::lock_guard<std::mutex> lk(g_inodeTableMutex);
        g_inodeTable.resize(2);
        auto rootNode = std::make_shared<FsNode>();
        rootNode->isDir = true;
        rootNode->mode  = 0755;
        g_inodeTable[1] = rootNode;
    }

    // Allocate concurrency structures
    g_queueMutex = new std::mutex[NUM_DRIVES];
    g_queueCond  = new std::condition_variable[NUM_DRIVES];
    g_workers    = new std::thread[NUM_DRIVES];
    g_queues     = new std::vector<FsRequest>[NUM_DRIVES];

    // Create SSD models
    for (int i=0; i<NUM_DRIVES; i++) {
        g_ssdModels.push_back(std::make_unique<SsdModel>());
    }

    // Start worker threads
    for (int i=0; i<NUM_DRIVES; i++) {
        g_workers[i] = std::thread(workerThreadFunc, i);
    }

    // Fill ll_ops
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
    ll_ops.access   = ll_access;    // Add this
    ll_ops.statfs   = ll_statfs;  
    DEBUG_LOG("FUSE operations initialized");
    std::cout << "[Info] Low-level FUSE FS started.\n";
    fuse_session_loop(se);

    fuse_session_unmount(se);
    fuse_remove_signal_handlers(se);
    fuse_session_destroy(se);
    fuse_opt_free_args(&args);

    // shutdown threads
    g_stopThreads.store(true);
    for (int i=0; i<NUM_DRIVES; i++) {
        g_queueCond[i].notify_all();
    }
    for (int i=0; i<NUM_DRIVES; i++) {
        if (g_workers[i].joinable()) {
            g_workers[i].join();
        }
    }

    delete[] g_queueMutex;
    delete[] g_queueCond;
    delete[] g_workers;
    delete[] g_queues;

    std::cout << "[Info] Exiting.\n";
    return 0;
}
