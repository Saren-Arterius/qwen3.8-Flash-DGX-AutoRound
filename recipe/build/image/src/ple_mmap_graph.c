/* ple_mmap_graph.c — CUDA-graph-capturable local SSD PLE row source (magi-v3-public).
 *
 * Same host-func pattern as ple_rdma_graph.c, but rows come from mmap'd
 * local safetensors shards instead of RDMA READs: the GPU copies n-gram ids
 * into a pinned host buffer, ple_mg_host_fn (recorded into the graph with
 * cuLaunchHostFunc) memcpy's those rows from the page cache into a pinned
 * row buffer, and the GPU copies the rows out. No Python on the hot path,
 * no .cpu() during capture — fully capturable in FULL and PIECEWISE graphs.
 *
 * No retry logic (local files don't flap); a missing file fails at init.
 * Out-of-range ids (padding, warm-up garbage) become zero rows.
 * Build: gcc -O2 -Wall -shared -fPIC ple_mmap_graph.c -o libple_mmap_graph.so
 * (no libverbs needed).
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct ple_mcall ple_mcall;
typedef struct {
    uint8_t *rows;
    int64_t *ids;
    uint8_t **shard_base;
    int nshards;
    int shard_size;
    int rb;
    uint64_t total;
    void **maps;
    size_t *map_lens;
    int nmaps;
    char hello[512];
    long reads;
    long cap;
    int nthreads;
    long chunk;
    pthread_t *tids;
    struct ple_mwork *work;
    ple_mcall *calls;
} ple_mg;
struct ple_mcall { ple_mg *g; long n; ple_mcall *next; };

struct ple_mwork { ple_mg *g; long lo, hi; };

/* Thread worker: memcpy one row range. Pure page-cache reads, no syscalls. */
static void *ple_mg_worker(void *arg)
{
    struct ple_mwork *w = arg;
    ple_mg *g = w->g;
    int rb = g->rb, ss = g->shard_size;
    uint64_t total = g->total;
    for (long i = w->lo; i < w->hi; i++) {
        int64_t id = g->ids[i];
        uint8_t *dst = g->rows + (size_t)i * rb;
        if (id < 0 || (uint64_t)id >= total) {
            memset(dst, 0, rb);
        } else {
            long shard = (long)((uint64_t)id / (uint64_t)ss);
            long local = (long)((uint64_t)id % (uint64_t)ss);
            if (shard < 0 || shard >= g->nshards) {
                memset(dst, 0, rb);
            } else {
                memcpy(dst, g->shard_base[shard] + (size_t)local * rb, rb);
            }
        }
    }
    return NULL;
}

/* Serial fallback (small batches: thread spawn costs more than it saves). */
static void ple_mg_lookup_serial(ple_mg *g, long n)
{
    struct ple_mwork w = { g, 0, n };
    ple_mg_worker(&w);
}

ple_mg *ple_mg_new(void *rows, void *ids, long cap, int row_bytes,
                   uint64_t total, int shard_size, int nshards,
                   const char **paths, const int64_t *offsets,
                   double weight_scale, int prewarm)
{
    ple_mg *g = calloc(1, sizeof *g);
    if (!g) return NULL;
    g->rows = rows; g->ids = ids; g->cap = cap;
    g->rb = row_bytes; g->total = total;
    g->shard_size = shard_size; g->nshards = nshards;
    g->shard_base = calloc(nshards, sizeof *g->shard_base);
    g->maps = calloc(nshards, sizeof *g->maps);
    g->map_lens = calloc(nshards, sizeof *g->map_lens);
    if (!g->shard_base || !g->maps || !g->map_lens) return NULL;

    /* mmap each unique file once; shards in the same file share it */
    char **uniq = calloc(nshards, sizeof *uniq);
    void **ubase = calloc(nshards, sizeof *ubase);
    int nuniq = 0;
    for (int j = 0; j < nshards; j++) {
        const char *p = paths[j];
        int k = -1;
        for (int u = 0; u < nuniq; u++)
            if (!strcmp(uniq[u], p)) { k = u; break; }
        if (k < 0) {
            int fd = open(p, O_RDONLY);
            if (fd < 0) { perror("ple_mmap_graph: open"); return NULL; }
            struct stat st;
            if (fstat(fd, &st)) { perror("ple_mmap_graph: fstat"); close(fd); return NULL; }
            void *base = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) { perror("ple_mmap_graph: mmap"); close(fd); return NULL; }
            close(fd); /* mapping survives close */
            uniq[nuniq] = strdup(p);
            ubase[nuniq] = base;
            g->maps[nuniq] = base;
            g->map_lens[nuniq] = st.st_size;
            k = nuniq++;
        }
        g->shard_base[j] = (uint8_t *)ubase[k] + offsets[j];
    }
    for (int u = 0; u < nuniq; u++) free(uniq[u]);
    free(uniq); free(ubase);
    g->nmaps = nuniq;
    /* Reuse the mmap knobs: WORKERS = threads, CHUNK = min rows/thread. */
    const char *we = getenv("VLLM_PLE_MMAP_WORKERS");
    long tn = we ? atol(we) : 16;
    if (tn < 1) tn = 1;
    if (tn > 64) tn = 64;
    g->nthreads = (int)tn;
    const char *ce = getenv("VLLM_PLE_MMAP_CHUNK");
    long ck = ce ? atol(ce) : 2048;
    if (ck < 1) ck = 1;
    g->chunk = ck;
    g->tids = calloc(tn, sizeof *g->tids);
    g->work = calloc(tn, sizeof *g->work);

    if (prewarm) {
#ifdef MADV_POPULATE_READ
        for (int u = 0; u < nuniq; u++)
            madvise(g->maps[u], g->map_lens[u], MADV_POPULATE_READ);
#else
        for (int u = 0; u < nuniq; u++) {
            volatile uint8_t *b = g->maps[u];
            size_t len = g->map_lens[u];
            long ps = sysconf(_SC_PAGESIZE);
            for (size_t o = 0; o < len; o += ps) (void)b[o];
        }
#endif
    }

    snprintf(g->hello, sizeof g->hello,
             "{\"rows\": %llu, \"row_bytes\": %d, \"weight_scale\": %.10g}\n",
             (unsigned long long)total, row_bytes, weight_scale);
    return g;
}

const char *ple_mg_hello(ple_mg *g) { return g->hello; }
long ple_mg_reads(ple_mg *g) { return g->reads; }

/* One callback argument per batch size (graphs bake the pointer in). */
void *ple_mg_call(ple_mg *g, long n)
{
    for (ple_mcall *c = g->calls; c; c = c->next)
        if (c->n == n) return c;
    if (n > g->cap) return NULL;
    ple_mcall *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->g = g; c->n = n; c->next = g->calls; g->calls = c;
    return c;
}

/* CUDA host function: runs on the driver's callback thread, stream-ordered.
 * Pure memcpy from page cache — no syscalls, no Python, no GIL. */
void ple_mg_host_fn(void *arg)
{
    ple_mcall *c = arg;
    ple_mg *g = c->g;
    long n = c->n;
    /* Small batches stay serial: spawn costs more than it saves. */
    long nthr = g->nthreads;
    long by_chunk = (n + g->chunk - 1) / g->chunk;
    if (by_chunk < nthr) nthr = by_chunk;
    if (nthr <= 1) {
        ple_mg_lookup_serial(g, n);
    } else {
        long per = (n + nthr - 1) / nthr;
        int spawned[64] = {0};
        for (int t = 0; t < nthr; t++) {
            long lo = t * per;
            if (lo >= n) break;
            long hi = lo + per > n ? n : lo + per;
            g->work[t].g = g;
            g->work[t].lo = lo;
            g->work[t].hi = hi;
            /* ponytail: thread handles/args are preallocated, one spawn per chunk. */
            if (pthread_create(&g->tids[t], NULL, ple_mg_worker, &g->work[t])) {
                struct ple_mwork w = { g, lo, hi };
                ple_mg_worker(&w);  /* fallback: this chunk inline */
            } else {
                spawned[t] = 1;
            }
        }
        for (int t = 0; t < nthr; t++)
            if (spawned[t])
                pthread_join(g->tids[t], NULL);
    }
    g->reads++;
}
