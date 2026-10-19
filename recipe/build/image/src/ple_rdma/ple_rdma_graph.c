/* ple_rdma_graph.c — CUDA-graph-capturable PLE row source (vLLM 0.32 adapter).
 *
 * The GPU copies n-gram ids into a pinned host buffer, ple_g_host_fn (recorded
 * into the graph with cuLaunchHostFunc) READs those rows from wtako into a
 * pinned MR buffer, and the GPU copies the rows out. No Python on the hot path.
 *
 * Failure semantics match vllm_ple_rdma.py: a failed READ never raises. The
 * dead QP/socket is closed, the client redials every 2 s until the server is
 * back, and the READ is retried; the CUDA stream (and so the engine) stalls
 * meanwhile. Out-of-range ids (padding, warm-up garbage) become zero rows.
 * Build: gcc -O2 -Wall -shared -fPIC ple_rdma_graph.c -libverbs -o libple_rdma_graph.so
 */
#include "ple_rdma.c"
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

typedef struct ple_call ple_call;
typedef struct {
    char host[128], port[16];
    ple_dev *dev;
    ple_qp *qp;
    int sock;
    struct ibv_mr *mr;
    uint32_t lkey, rkey;
    uint8_t *rows;
    int64_t *ids;
    uint64_t *idu;
    unsigned char *bad;
    long cap;
    int rb;
    uint64_t raddr, total;
    char hello[8192];
    long reads, reconnects;
    ple_call *calls;
} ple_g;
struct ple_call { ple_g *g; long n; ple_call *next; };

static void logf_(const char *m, long a)
{
    time_t t = time(NULL);
    char ts[32];
    strftime(ts, sizeof ts, "%F %T", localtime(&t));
    fprintf(stderr, "%s ple_rdma_graph: %s %ld\n", ts, m, a);
    fflush(stderr);
}

static int jnum(const char *s, const char *key, unsigned long long *out)
{
    char k[64];
    snprintf(k, sizeof k, "\"%s\"", key);
    const char *p = strstr(s, k);
    if (!p || !(p = strchr(p + strlen(k), ':'))) return -1;
    *out = strtoull(p + 1, NULL, 10);
    return 0;
}

static int jgid(const char *s, uint8_t out[16])
{
    const char *p = strstr(s, "\"gid\"");
    if (!p || !(p = strchr(p + 5, ':')) || !(p = strchr(p, '"'))) return -1;
    p++;
    for (int i = 0; i < 16; i++) {
        unsigned v;
        if (sscanf(p + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
    }
    return 0;
}

static void g_close(ple_g *g)
{
    if (g->qp) { ple_qp_destroy(g->qp); g->qp = NULL; }
    if (g->sock >= 0) { close(g->sock); g->sock = -1; }
}

static int g_connect_once(ple_g *g)
{
    struct addrinfo hints = { .ai_family = AF_INET, .ai_socktype = SOCK_STREAM }, *ai;
    if (getaddrinfo(g->host, g->port, &hints, &ai)) return -1;
    int s = socket(ai->ai_family, ai->ai_socktype, 0);
    int one = 1, two = 2, five = 5, three = 3;
    struct timeval tv = { .tv_sec = 15 }, none = { 0 };
    setsockopt(s, IPPROTO_TCP, TCP_SYNCNT, &two, sizeof two);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &five, sizeof five);
    setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &five, sizeof five);
    setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &three, sizeof three);
    int rc = connect(s, ai->ai_addr, ai->ai_addrlen);
    freeaddrinfo(ai);
    if (rc) { close(s); return -2; }
    size_t len = 0;
    while (len == 0 || g->hello[len - 1] != '\n') {
        if (len + 1 >= sizeof g->hello) { close(s); return -3; }
        ssize_t got = recv(s, g->hello + len, sizeof g->hello - 1 - len, 0);
        if (got <= 0) { close(s); return -4; }
        len += got;
    }
    g->hello[len] = 0;
    unsigned long long addr, rkey, rb, rows, qpn;
    uint8_t rgid[16];
    if (jnum(g->hello, "addr", &addr) || jnum(g->hello, "rkey", &rkey) ||
        jnum(g->hello, "row_bytes", &rb) || jnum(g->hello, "rows", &rows) ||
        jnum(g->hello, "qpn", &qpn) || jgid(g->hello, rgid)) { close(s); return -5; }
    if ((int)rb != g->rb || rows < g->total) { close(s); return -6; }
    ple_qp *q = ple_qp_create(g->dev);
    if (!q) { close(s); return -7; }
    uint8_t lgid[16];
    uint32_t lqpn;
    ple_qp_local(q, lgid, &lqpn);
    char msg[128];
    int m = snprintf(msg, sizeof msg, "{\"gid\": \"");
    for (int i = 0; i < 16; i++) m += snprintf(msg + m, sizeof msg - m, "%02x", lgid[i]);
    m += snprintf(msg + m, sizeof msg - m, "\", \"qpn\": %u}\n", lqpn);
    if (send(s, msg, m, MSG_NOSIGNAL) != m || ple_qp_connect(q, rgid, (uint32_t)qpn)) {
        ple_qp_destroy(q); close(s); return -8;
    }
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &none, sizeof none);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &none, sizeof none);
    g->sock = s;  /* kept open = session alive on the server */
    g->qp = q;
    g->raddr = addr;
    g->rkey = (uint32_t)rkey;
    return 0;
}

static void g_connect_forever(ple_g *g)
{
    for (long attempt = 1;; attempt++) {
        int rc = g_connect_once(g);
        if (!rc) {
            if (attempt > 1) logf_("connected after retries:", attempt - 1);
            return;
        }
        logf_("connect failed (rc), retrying in 2s, stalling (no fallback), rc", rc);
        sleep(2);
    }
}

ple_g *ple_g_new(const char *host, const char *port, const char *dev_name, int gid_index,
                 void *rows, void *ids, long cap, int row_bytes, uint64_t total)
{
    ple_g *g = calloc(1, sizeof *g);
    snprintf(g->host, sizeof g->host, "%s", host);
    snprintf(g->port, sizeof g->port, "%s", port);
    g->sock = -1;
    g->rows = rows; g->ids = ids; g->cap = cap; g->rb = row_bytes; g->total = total;
    g->idu = malloc(cap * sizeof *g->idu);
    g->bad = malloc(cap);
    if (!(g->dev = ple_dev_open(dev_name, 1, gid_index))) return NULL;
    if (!(g->mr = ple_mr_reg(g->dev, rows, (size_t)cap * row_bytes, 0))) return NULL;
    g->lkey = ple_mr_lkey(g->mr);
    g_connect_forever(g);
    return g;
}

const char *ple_g_hello(ple_g *g) { return g->hello; }
long ple_g_reads(ple_g *g) { return g->reads; }
long ple_g_reconnects(ple_g *g) { return g->reconnects; }

/* One callback argument per batch size (graphs bake the pointer in). */
void *ple_g_call(ple_g *g, long n)
{
    for (ple_call *c = g->calls; c; c = c->next)
        if (c->n == n) return c;
    if (n > g->cap) return NULL;
    ple_call *c = calloc(1, sizeof *c);
    c->g = g; c->n = n; c->next = g->calls; g->calls = c;
    return c;
}

/* CUDA host function: runs on the driver's callback thread, in stream order. */
void ple_g_host_fn(void *arg)
{
    ple_call *c = arg;
    ple_g *g = c->g;
    long n = c->n, nbad = 0;
    for (long i = 0; i < n; i++) {
        int64_t id = g->ids[i];
        int b = id < 0 || (uint64_t)id >= g->total;
        g->bad[i] = b;
        g->idu[i] = b ? 0 : (uint64_t)id;
        nbad += b;
    }
    for (long attempt = 0;; attempt++) {
        if (g->qp && !ple_read_rows(g->qp, g->lkey, g->rows, g->raddr, g->rkey, g->idu, n, g->rb)) {
            if (attempt) logf_("READ recovered after retries:", attempt);
            break;
        }
        logf_("READ failed, reconnecting in 2s, stalling (no fallback), attempt", attempt + 1);
        g->reconnects++;
        g_close(g);
        sleep(2);
        g_connect_forever(g);
    }
    if (nbad)
        for (long i = 0; i < n; i++)
            if (g->bad[i]) memset(g->rows + i * g->rb, 0, g->rb);
    g->reads++;
}
