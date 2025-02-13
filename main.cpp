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
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <unistd.h>

static constexpr size_t BLOCK_SIZE = 4096;
static constexpr size_t QUEUE_CAPACITY_BYTES = 64 * 1024; // 64KB per drive queue

// -----------------------------------------------------------------------------
// FNV1a helper for (ino, offset) to pick a drive
static uint64_t fnv1a_hash_offset(fuse_ino_t ino, off_t offset)
{
    // We hash the block index (offset / BLOCK_SIZE) plus the inode
    const off_t blockIndex = offset / BLOCK_SIZE;
    std::string combined = std::to_string(ino) + "-" + std::to_string(blockIndex);
    uint64_t hash = 1469598103934665603ULL;
    for (char c : combined) {
        hash ^= (unsigned char)c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// -----------------------------------------------------------------------------
// FUSE option struct
struct options {
    int drives;
};
static struct options g_opts = {4};

static struct fuse_opt option_spec[] = {
    {"drives=%d", offsetof(struct options, drives), 0},
    FUSE_OPT_END
};

static int NUM_DRIVES = 4;

// -----------------------------------------------------------------------------
// Basic FsNode: "striped" file data
struct FsNode {
    bool isDir;
    mode_t mode;

    // For directories
    std::map<std::string, fuse_ino_t> children;

    // For files: stripes[driveIndex] is a map<offset, blockData>
    std::vector<std::map<off_t, std::vector<uint8_t>>> stripes; 
    size_t fileSize;  // max offset written (like "logical" file size)

    FsNode(bool is_dir, mode_t m)
        : isDir(is_dir), mode(m), stripes(NUM_DRIVES), fileSize(0) {}
};

// Global inode table
static std::mutex g_inodeMutex;
static std::atomic<fuse_ino_t> g_nextIno{2};
static std::vector<std::shared_ptr<FsNode>> g_inodeTable;

static fuse_ino_t createInode(bool isDir, mode_t mode)
{
    std::lock_guard<std::mutex> lk(g_inodeMutex);
    fuse_ino_t ino = g_nextIno.fetch_add(1, std::memory_order_relaxed);
    if ((size_t)ino >= g_inodeTable.size()) {
        g_inodeTable.resize(ino + 1);
    }
    g_inodeTable[ino] = std::make_shared<FsNode>(isDir, mode);
    return ino;
}
static std::shared_ptr<FsNode> getNode(fuse_ino_t ino)
{
    if (ino < g_inodeTable.size()) {
        return g_inodeTable[ino];
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// We define an enum for operation
enum class RequestOp {
    LOOKUP, MKDIR, CREATE, UNLINK, RMDIR, RENAME,
    WRITE, READ, GETATTR, FLUSH, RELEASE, READDIR,
    ACCESS, STATFS
};

// The request structure, stored in the queue
struct FsRequest {
    RequestOp  op;
    fuse_req_t req;

    // Common fields
    fuse_ino_t parent;
    std::string name;
    fuse_ino_t ino;
    fuse_ino_t newparent;
    std::string newname;
    size_t size;
    off_t offset;
    mode_t mode;
    std::vector<uint8_t> data;  // The data for READ/WRITE
    unsigned int rename_flags;
    struct fuse_file_info fi;
    int access_mask;
};

// -----------------------------------------------------------------------------
// We store each drive's queue, capacity control, worker thread, etc.
static std::vector<FsRequest>* g_queues = nullptr;   // array of queues, one per drive
static std::mutex* g_queueMutex = nullptr;           // array of mutexes
static std::condition_variable* g_queueCond = nullptr;       // signal "not empty"
static std::condition_variable* g_queueCondNotFull = nullptr; // signal "space available"
static std::vector<size_t> g_queueFreeBytes;          // how many bytes free in queue
static std::thread* g_workers = nullptr;
static std::atomic<bool> g_stopThreads{false};

// -----------------------------------------------------------------------------
// Forward declarations of the "processX" worker functions
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
// Worker thread: continuously pop from queue, handle requests
static void workerThreadFunc(int driveIndex)
{
    while (!g_stopThreads.load()) {
        FsRequest r;
        {
            std::unique_lock<std::mutex> lk(g_queueMutex[driveIndex]);
            // Wait for something in the queue (or stop)
            g_queueCond[driveIndex].wait(lk, [driveIndex]{
                return !g_queues[driveIndex].empty() || g_stopThreads.load();
            });
            if (g_stopThreads.load()) {
                break;
            }
            // Pop front
            r = std::move(g_queues[driveIndex].front());
            g_queues[driveIndex].erase(g_queues[driveIndex].begin());

            // Free up capacity
            size_t usedBytes = r.data.size();
            g_queueFreeBytes[driveIndex] += usedBytes;
            // Notify any waiting producers that space is available
            g_queueCondNotFull[driveIndex].notify_one();
        }

        // Now actually process
        try {
            switch(r.op) {
                case RequestOp::LOOKUP:   processLookup(r);   break;
                case RequestOp::MKDIR:    processMkdir(r);    break;
                case RequestOp::CREATE:   processCreate(r);   break;
                case RequestOp::UNLINK:   processUnlink(r);   break;
                case RequestOp::RMDIR:    processRmdir(r);    break;
                case RequestOp::RENAME:   processRename(r);   break;
                case RequestOp::WRITE:    processWrite(r);    break;
                case RequestOp::READ:     processRead(r);     break;
                case RequestOp::GETATTR:  processGetattr(r);  break;
                case RequestOp::FLUSH:    processFlush(r);    break;
                case RequestOp::RELEASE:  processRelease(r);  break;
                case RequestOp::READDIR:  processReaddir(r);  break;
                case RequestOp::ACCESS:   processAccess(r);   break;
                case RequestOp::STATFS:   processStatfs(r);   break;
            }
        } catch(...) {
            fuse_reply_err(r.req, EIO);
        }
    }
}

// -----------------------------------------------------------------------------
// Enqueue with capacity check
static void enqueueRequest(FsRequest &&r, int driveIndex)
{
    size_t needed = r.data.size(); // number of bytes to store in queue

    // Acquire lock
    std::unique_lock<std::mutex> lk(g_queueMutex[driveIndex]);
    // Wait until there's enough free capacity or we are stopping
    g_queueCondNotFull[driveIndex].wait(lk, [driveIndex, needed]{
        return (g_queueFreeBytes[driveIndex] >= needed) || g_stopThreads.load();
    });
    if (g_stopThreads.load()) {
        fuse_reply_err(r.req, ESHUTDOWN);
        return;
    }
    // Insert at the back
    g_queues[driveIndex].push_back(std::move(r));
    // Reduce capacity
    g_queueFreeBytes[driveIndex] -= needed;

    // Notify the consumer
    g_queueCond[driveIndex].notify_one();
}

// -----------------------------------------------------------------------------
// Decide which drive queue gets a request
static int computeDriveIndexForWrite(fuse_ino_t ino, off_t off)
{
    uint64_t h = fnv1a_hash_offset(ino, off);
    return h % NUM_DRIVES;
}
static int computeDriveIndexForRead(fuse_ino_t ino, off_t off)
{
    // same logic
    uint64_t h = fnv1a_hash_offset(ino, off);
    return h % NUM_DRIVES;
}
// For metadata ops, you can do something simpler (like modulo by parent or inode)
static int computeDriveIndexDefault(fuse_ino_t x)
{
    // e.g. just do (x % NUM_DRIVES)
    return x % NUM_DRIVES;
}

// -----------------------------------------------------------------------------
// “process” functions -- these run in the worker threads
static std::mutex g_fsLock; // single global lock for all FS ops, for simplicity

static void processLookup(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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
    fuse_ino_t child = it->second;
    auto cnode = getNode(child);
    if (!cnode) {
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
    if (cnode->isDir) {
        e.attr.st_mode = S_IFDIR | cnode->mode;
        e.attr.st_nlink = 2 + cnode->children.size();
    } else {
        e.attr.st_mode = S_IFREG | cnode->mode;
        e.attr.st_size = cnode->fileSize;
        e.attr.st_nlink = 1;
    }
    fuse_reply_entry(r.req, &e);
}
static void processMkdir(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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

    struct fuse_entry_param e;
    memset(&e, 0, sizeof(e));
    e.ino = newIno;
    e.attr.st_ino = newIno;
    e.attr.st_mode = S_IFDIR | r.mode;
    e.attr.st_nlink = 2;
    e.attr_timeout = 1.0;
    e.entry_timeout = 1.0;
    fuse_reply_entry(r.req, &e);
}
static void processCreate(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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

    struct fuse_entry_param e;
    memset(&e, 0, sizeof(e));
    e.ino = newIno;
    e.attr.st_ino = newIno;
    e.attr.st_mode = S_IFREG | r.mode;
    e.attr.st_nlink = 1;
    e.attr_timeout = 1.0;
    e.entry_timeout = 1.0;

    struct fuse_file_info fi;
    memset(&fi, 0, sizeof(fi));
    fi.fh = newIno;  // store inode in fh
    fuse_reply_create(r.req, &e, &fi);
}
static void processUnlink(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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
    // If the child is a directory, it's an error
    auto child = getNode(it->second);
    if (!child || child->isDir) {
        fuse_reply_err(r.req, EISDIR);
        return;
    }
    parentNode->children.erase(it);
    fuse_reply_err(r.req, 0);
}
static void processRmdir(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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
    std::lock_guard<std::mutex> guard(g_fsLock);
    auto pold = getNode(r.parent);
    auto pnew = getNode(r.newparent);
    if (!pold || !pold->isDir || !pnew || !pnew->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    auto it = pold->children.find(r.name);
    if (it == pold->children.end()) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    fuse_ino_t oldIno = it->second;
    pnew->children[r.newname] = oldIno;
    pold->children.erase(it);
    fuse_reply_err(r.req, 0);
}
// The “striped write”
static void processWrite(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    // offset -> which drive
    int driveIndex = computeDriveIndexForWrite(r.ino, r.offset);
    // Overwrite stripes[driveIndex][offset] with r.data
    node->stripes[driveIndex][r.offset] = r.data;

    // Update fileSize if needed
    size_t endPos = r.offset + r.data.size();
    if (endPos > node->fileSize) {
        node->fileSize = endPos;
    }
    fuse_reply_write(r.req, r.data.size());
}
// The “striped read”
static void processRead(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
    auto node = getNode(r.ino);
    if (!node || node->isDir) {
        fuse_reply_err(r.req, ENOENT);
        return;
    }
    if ((size_t)r.offset >= node->fileSize) {
        fuse_reply_buf(r.req, nullptr, 0);
        return;
    }
    // clamp read size if reading past EOF
    size_t maxAvail = node->fileSize - r.offset;
    if (r.size > maxAvail) {
        r.size = maxAvail;
    }
    int driveIndex = computeDriveIndexForRead(r.ino, r.offset);
    auto &m = node->stripes[driveIndex];
    auto it = m.find(r.offset);
    if (it == m.end()) {
        // no data => read 0
        fuse_reply_buf(r.req, nullptr, 0);
        return;
    }
    const auto &blk = it->second;
    size_t actual = std::min(r.size, blk.size());
    fuse_reply_buf(r.req, (const char*)blk.data(), actual);
}
static void processGetattr(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
    if (r.ino == FUSE_ROOT_ID && r.ino >= g_inodeTable.size()) {
        // in theory you might handle root specially
        struct stat st{};
        st.st_ino = FUSE_ROOT_ID;
        st.st_mode = S_IFDIR | 0755;
        st.st_nlink = 2;
        fuse_reply_attr(r.req, &st, 1.0);
        return;
    }
    auto node = getNode(r.ino);
    if (!node) {
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
        st.st_size = node->fileSize;
        st.st_nlink = 1;
    }
    st.st_uid = getuid();
    st.st_gid = getgid();
    fuse_reply_attr(r.req, &st, 1.0);
}
static void processFlush(FsRequest &r) {
    fuse_reply_err(r.req, 0);
}
static void processRelease(FsRequest &r) {
    fuse_reply_err(r.req, 0);
}
static void processReaddir(FsRequest &r) {
    std::lock_guard<std::mutex> guard(g_fsLock);
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
    auto addEntry = [&](fuse_ino_t e_ino, const char *n) {
        struct stat st{};
        st.st_ino = e_ino;
        auto en = getNode(e_ino);
        if (en) {
            st.st_mode = en->isDir ? (S_IFDIR | en->mode) : (S_IFREG | en->mode);
        }
        size_t esz = fuse_add_direntry(r.req, buf + bpos, r.size - bpos, n, &st, r.offset + 1);
        if (esz > 0 && bpos + esz <= r.size) {
            bpos += esz;
        }
    };
    if (r.offset == 0) {
        addEntry(r.ino, ".");
        addEntry(FUSE_ROOT_ID, "..");
        for (auto &kv : node->children) {
            addEntry(kv.second, kv.first.c_str());
        }
    }
    fuse_reply_buf(r.req, buf, bpos);
    free(buf);
}
static void processAccess(FsRequest &r) {
    fuse_reply_err(r.req, 0);
}
static void processStatfs(FsRequest &r) {
    struct statvfs st{};
    st.f_bsize = 4096;
    st.f_frsize = 4096;
    st.f_blocks = 1024*1024;
    st.f_bfree  = 1024*1024;
    st.f_bavail = 1024*1024;
    st.f_files  = 100000;
    st.f_ffree  = 100000;
    st.f_favail = 100000;
    st.f_fsid   = 1234;
    st.f_flag   = 0;
    st.f_namemax= 255;
    fuse_reply_statfs(r.req, &st);
}

// -----------------------------------------------------------------------------
// FUSE low-level ops => They create FsRequest, pick drive, call enqueue
static void ll_lookup(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op = RequestOp::LOOKUP;
    r.req= req;
    r.parent= parent;
    r.name= name;
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_mkdir(fuse_req_t req, fuse_ino_t parent, const char *name, mode_t mode)
{
    FsRequest r;
    r.op = RequestOp::MKDIR;
    r.req= req;
    r.parent= parent;
    r.name= name;
    r.mode= mode;
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_create(fuse_req_t req, fuse_ino_t parent, const char *name,
                      mode_t mode, struct fuse_file_info *fi)
{
    FsRequest r;
    r.op= RequestOp::CREATE;
    r.req= req;
    r.parent= parent;
    r.name= name;
    r.mode= mode;
    r.fi= *fi;
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_unlink(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op= RequestOp::UNLINK;
    r.req= req;
    r.parent= parent;
    r.name= name;
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_rmdir(fuse_req_t req, fuse_ino_t parent, const char *name)
{
    FsRequest r;
    r.op= RequestOp::RMDIR;
    r.req= req;
    r.parent= parent;
    r.name= name;
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_rename(fuse_req_t req, fuse_ino_t parent, const char *name,
                      fuse_ino_t newparent, const char *newname, unsigned int flags)
{
    FsRequest r;
    r.op = RequestOp::RENAME;
    r.req= req;
    r.parent= parent;
    r.name= name;
    r.newparent= newparent;
    r.newname= newname;
    r.rename_flags= flags;
    // you might do something more advanced: pick one queue or broadcast
    int drive = computeDriveIndexDefault(parent);
    enqueueRequest(std::move(r), drive);
}
static void ll_open(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    // typically just set fi->fh = ino
    fi->fh = ino;
    fuse_reply_open(req, fi);
}
static void ll_write(fuse_req_t req, fuse_ino_t ino, const char *buf,
                     size_t size, off_t off, struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op= RequestOp::WRITE;
    r.req= req;
    r.ino= ino;
    r.offset= off;
    r.data.assign(buf, buf + size);
    // pick the drive via hashing offset
    int drive = computeDriveIndexForWrite(ino, off);
    enqueueRequest(std::move(r), drive);
}
static void ll_read(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                    struct fuse_file_info *fi)
{
    (void)fi;
    FsRequest r;
    r.op= RequestOp::READ;
    r.req= req;
    r.ino= ino;
    r.size= size;
    r.offset= off;
    int drive = computeDriveIndexForRead(ino, off);
    enqueueRequest(std::move(r), drive);
}
static void ll_getattr(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    FsRequest r;
    r.op= RequestOp::GETATTR;
    r.req= req;
    r.ino= ino;
    if (fi) r.fi = *fi;
    int drive = computeDriveIndexDefault(ino);
    enqueueRequest(std::move(r), drive);
}
static void ll_flush(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    FsRequest r;
    r.op= RequestOp::FLUSH;
    r.req= req;
    r.ino= ino;
    if (fi) r.fi = *fi;
    int drive = computeDriveIndexDefault(ino);
    enqueueRequest(std::move(r), drive);
}
static void ll_release(fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi)
{
    FsRequest r;
    r.op= RequestOp::RELEASE;
    r.req= req;
    r.ino= ino;
    if (fi) r.fi = *fi;
    int drive = computeDriveIndexDefault(ino);
    enqueueRequest(std::move(r), drive);
}
static void ll_readdir(fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
                       struct fuse_file_info *fi)
{
    FsRequest r;
    r.op= RequestOp::READDIR;
    r.req= req;
    r.ino= ino;
    r.size= size;
    r.offset= off;
    if (fi) r.fi = *fi;
    int drive = computeDriveIndexDefault(ino);
    enqueueRequest(std::move(r), drive);
}
static void ll_access(fuse_req_t req, fuse_ino_t ino, int mask)
{
    FsRequest r;
    r.op= RequestOp::ACCESS;
    r.req= req;
    r.ino= ino;
    r.access_mask= mask;
    int drive = computeDriveIndexDefault(ino);
    enqueueRequest(std::move(r), drive);
}
static void ll_statfs(fuse_req_t req, fuse_ino_t ino)
{
    FsRequest r;
    r.op= RequestOp::STATFS;
    r.req= req;
    r.ino= ino;
    // just always drive 0, or something
    enqueueRequest(std::move(r), 0);
}

static void ll_init(void* /*userdata*/, struct fuse_conn_info* /*conn*/)
{
    std::cout << "[Info] FUSE init callback\n";
}

// -----------------------------------------------------------------------------
// main
int main(int argc, char** argv)
{
    std::cout << "[Info] Starting queued & striped FUSE FS\n";

    // parse fuse args
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
    struct fuse_cmdline_opts cmdline_opts;
    memset(&cmdline_opts, 0, sizeof(cmdline_opts));
    if (fuse_parse_cmdline(&args, &cmdline_opts) != 0) {
        fuse_opt_free_args(&args);
        return 1;
    }
    if (fuse_opt_parse(&args, &g_opts, option_spec, nullptr) == -1) {
        std::cerr << "[Error] parse -o drives=NN\n";
        fuse_opt_free_args(&args);
        return 1;
    }
    if (g_opts.drives < 1) g_opts.drives = 1;
    if (g_opts.drives > 16) g_opts.drives = 16;
    NUM_DRIVES = g_opts.drives;
    std::cout << "[Info] Using NUM_DRIVES=" << NUM_DRIVES << "\n";

    // set up fuse ops
    static struct fuse_lowlevel_ops ll_ops{};
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

    // create FUSE session
    struct fuse_session* se = fuse_session_new(&args, &ll_ops, sizeof(ll_ops), nullptr);
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

    // Prepare root inode
    g_inodeTable.resize(2);
    auto root = std::make_shared<FsNode>(true, 0755);
    g_inodeTable[FUSE_ROOT_ID] = root;

    // Prepare queues & worker threads
    g_queues            = new std::vector<FsRequest>[NUM_DRIVES];
    g_queueMutex        = new std::mutex[NUM_DRIVES];
    g_queueCond         = new std::condition_variable[NUM_DRIVES];
    g_queueCondNotFull  = new std::condition_variable[NUM_DRIVES];
    g_queueFreeBytes.resize(NUM_DRIVES, QUEUE_CAPACITY_BYTES);
    g_workers = new std::thread[NUM_DRIVES];

    for (int i = 0; i < NUM_DRIVES; i++) {
        // Start a worker that handles the queue for drive i
        g_workers[i] = std::thread(workerThreadFunc, i);
    }

    std::cout << "[Info] FUSE FS mounted at " << cmdline_opts.mountpoint << "\n";
    fuse_session_loop(se);

    // Cleanup
    g_stopThreads = true;
    for (int i = 0; i < NUM_DRIVES; i++) {
        {
            std::lock_guard<std::mutex> lk(g_queueMutex[i]);
            // notify workers so they can exit
            g_queueCond[i].notify_all();
            g_queueCondNotFull[i].notify_all();
        }
    }
    for (int i = 0; i < NUM_DRIVES; i++) {
        if (g_workers[i].joinable()) {
            g_workers[i].join();
        }
    }

    fuse_session_unmount(se);
    fuse_remove_signal_handlers(se);
    fuse_session_destroy(se);
    fuse_opt_free_args(&args);

    delete[] g_queues;
    delete[] g_queueMutex;
    delete[] g_queueCond;
    delete[] g_queueCondNotFull;
    delete[] g_workers;

    std::cout << "[Info] Filesystem unmounted\n";
    return 0;
}
