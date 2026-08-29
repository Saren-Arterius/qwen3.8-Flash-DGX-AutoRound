/* ple_rdma.c — minimal one-sided RDMA READ helper for the PLE table.
 *
 * Shared by the wtako server (registers the table MR, then idles) and the
 * magi client (posts batched READs of 160-byte rows). RC QPs, RoCE v2,
 * manual out-of-band QP exchange (no rdma_cm). Build:
 *   gcc -O2 -Wall -shared -fPIC ple_rdma.c -libverbs -o libple_rdma.so
 */
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK 256 /* WQEs posted per doorbell; QP depth is 2*CHUNK */

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    int port;
    int gid_index;
    union ibv_gid gid;
    enum ibv_mtu mtu;
} ple_dev;

typedef struct {
    ple_dev *dev;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
} ple_qp;

static void err(const char *what) { fprintf(stderr, "ple_rdma: %s failed\n", what); }

ple_dev *ple_dev_open(const char *name, int port, int gid_index)
{
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) { err("ibv_get_device_list"); return NULL; }
    struct ibv_device *want = NULL;
    for (int i = 0; i < n; i++)
        if (!strcmp(ibv_get_device_name(list[i]), name)) want = list[i];
    if (!want) { err("device lookup"); ibv_free_device_list(list); return NULL; }

    ple_dev *d = calloc(1, sizeof(*d));
    d->ctx = ibv_open_device(want);
    ibv_free_device_list(list);
    if (!d->ctx) { err("ibv_open_device"); free(d); return NULL; }
    d->pd = ibv_alloc_pd(d->ctx);
    if (!d->pd) { err("ibv_alloc_pd"); return NULL; }
    d->port = port;
    d->gid_index = gid_index;
    if (ibv_query_gid(d->ctx, port, gid_index, &d->gid)) { err("ibv_query_gid"); return NULL; }
    struct ibv_port_attr pa;
    if (ibv_query_port(d->ctx, port, &pa)) { err("ibv_query_port"); return NULL; }
    d->mtu = pa.active_mtu;
    return d;
}

struct ibv_mr *ple_mr_reg(ple_dev *d, void *addr, size_t len, int remote_read)
{
    int access = IBV_ACCESS_LOCAL_WRITE;
    if (remote_read) access |= IBV_ACCESS_REMOTE_READ;
    struct ibv_mr *mr = ibv_reg_mr(d->pd, addr, len, access);
    if (!mr) err("ibv_reg_mr");
    return mr;
}

uint32_t ple_mr_rkey(struct ibv_mr *mr) { return mr->rkey; }
uint32_t ple_mr_lkey(struct ibv_mr *mr) { return mr->lkey; }
uint64_t ple_mr_addr(struct ibv_mr *mr) { return (uint64_t)(uintptr_t)mr->addr; }

ple_qp *ple_qp_create(ple_dev *d)
{
    ple_qp *q = calloc(1, sizeof(*q));
    q->dev = d;
    q->cq = ibv_create_cq(d->ctx, 2 * CHUNK, NULL, NULL, 0);
    if (!q->cq) { err("ibv_create_cq"); return NULL; }
    struct ibv_qp_init_attr ia = {
        .send_cq = q->cq,
        .recv_cq = q->cq,
        .cap = { .max_send_wr = 2 * CHUNK, .max_recv_wr = 1,
                 .max_send_sge = 1, .max_recv_sge = 1 },
        .qp_type = IBV_QPT_RC,
        .sq_sig_all = 0,
    };
    q->qp = ibv_create_qp(d->pd, &ia);
    if (!q->qp) { err("ibv_create_qp"); return NULL; }
    struct ibv_qp_attr a = {
        .qp_state = IBV_QPS_INIT,
        .pkey_index = 0,
        .port_num = d->port,
        .qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE,
    };
    if (ibv_modify_qp(q->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                      IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        err("modify INIT");
        return NULL;
    }
    return q;
}

int ple_qp_local(ple_qp *q, uint8_t gid_out[16], uint32_t *qpn)
{
    memcpy(gid_out, q->dev->gid.raw, 16);
    *qpn = q->qp->qp_num;
    return 0;
}

int ple_qp_connect(ple_qp *q, const uint8_t rgid[16], uint32_t rqpn)
{
    struct ibv_qp_attr a = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = q->dev->mtu,
        .dest_qp_num = rqpn,
        .rq_psn = 0,
        .max_dest_rd_atomic = 16,
        .min_rnr_timer = 12,
        .ah_attr = {
            .is_global = 1,
            .port_num = q->dev->port,
            .grh = { .sgid_index = q->dev->gid_index, .hop_limit = 64 },
        },
    };
    memcpy(a.ah_attr.grh.dgid.raw, rgid, 16);
    if (ibv_modify_qp(q->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        err("modify RTR");
        return -1;
    }
    struct ibv_qp_attr b = {
        .qp_state = IBV_QPS_RTS,
        .timeout = 12,
        .retry_cnt = 7,
        .rnr_retry = 7,
        .sq_psn = 0,
        .max_rd_atomic = 16,
    };
    if (ibv_modify_qp(q->qp, &b, IBV_QP_STATE | IBV_QP_TIMEOUT |
                      IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                      IBV_QP_MAX_QP_RD_ATOMIC)) {
        err("modify RTS");
        return -1;
    }
    return 0;
}

/* Read n rows (ids[i] * row_bytes offsets from raddr_base) into
 * laddr_base + i*row_bytes. Synchronous; chunks of CHUNK WQEs, one signaled
 * completion per chunk. Returns 0 or negative errno-ish. */
int ple_read_rows(ple_qp *q, uint32_t lkey, void *laddr_base,
                  uint64_t raddr_base, uint32_t rkey,
                  const uint64_t *ids, long n, int row_bytes)
{
    static __thread struct ibv_sge sge[CHUNK];
    static __thread struct ibv_send_wr wr[CHUNK];
    for (long done = 0; done < n; done += CHUNK) {
        long m = n - done < CHUNK ? n - done : CHUNK;
        for (long i = 0; i < m; i++) {
            sge[i].addr = (uint64_t)(uintptr_t)laddr_base + (done + i) * row_bytes;
            sge[i].length = row_bytes;
            sge[i].lkey = lkey;
            wr[i].wr_id = done + i;
            wr[i].next = i + 1 < m ? &wr[i + 1] : NULL;
            wr[i].sg_list = &sge[i];
            wr[i].num_sge = 1;
            wr[i].opcode = IBV_WR_RDMA_READ;
            wr[i].send_flags = i + 1 < m ? 0 : IBV_SEND_SIGNALED;
            wr[i].wr.rdma.remote_addr = raddr_base + ids[done + i] * (uint64_t)row_bytes;
            wr[i].wr.rdma.rkey = rkey;
        }
        struct ibv_send_wr *bad = NULL;
        if (ibv_post_send(q->qp, &wr[0], &bad)) { err("ibv_post_send"); return -1; }
        struct ibv_wc wc;
        int got;
        while ((got = ibv_poll_cq(q->cq, 1, &wc)) == 0)
            ;
        if (got < 0 || wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ple_rdma: wc status %d (%s)\n", wc.status,
                    got > 0 ? ibv_wc_status_str(wc.status) : "poll error");
            return -2;
        }
    }
    return 0;
}

void ple_qp_destroy(ple_qp *q)
{
    if (!q) return;
    if (q->qp) ibv_destroy_qp(q->qp);
    if (q->cq) ibv_destroy_cq(q->cq);
    free(q);
}

void ple_mr_dereg(struct ibv_mr *mr) { if (mr) ibv_dereg_mr(mr); }

void ple_dev_close(ple_dev *d)
{
    if (!d) return;
    if (d->pd) ibv_dealloc_pd(d->pd);
    if (d->ctx) ibv_close_device(d->ctx);
    free(d);
}
