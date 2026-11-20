/* Real refactor optimization derived from the bundled KLU refactor routine.
 * KLU by Timothy A. Davis; see the KLU terms in the project COPYING file.
 * The elimination order and each row's arithmetic order are preserved.
 */
#include "ngspice/config.h"
#include "ng_klu_parallel.h"

#ifdef NGSPICE_SYSTEM_KLU
/* The fast path depends on the bundled KLU storage layout. Do not interpret
 * storage owned by an arbitrary installed SuiteSparse version. */
int ng_klu_refactor(NgKluPlan **handle, int *ap, int *ai, double *ax,
                    klu_symbolic *s, klu_numeric *num, klu_common *common, int threads)
{
    (void)handle; (void)threads;
    return klu_refactor(ap, ai, ax, s, num, common);
}
void ng_klu_plan_free(NgKluPlan **handle) { *handle = NULL; }
#else
#include "SuiteSparse/klu_internal.h"
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define NG_KLU_AVX2 1
#pragma GCC optimize ("fp-contract=off")
#endif

typedef struct { int row; double value; } Pair;
typedef struct { int offset, row, length; } Run;
typedef struct {
    int start, size, oldcol, llen, ulen, *li, *ui, first_run, runs, long_values;
    double *lx, *ux, cost;
} Column;
typedef struct { int begin, end, parallel; } Stage;

struct ng_klu_plan {
    klu_numeric *numeric;
    klu_symbolic *symbolic;
    int n, maxblock, threads, eligible, stages, parallel_stages, vectorized;
    Column *column;
    Run *run;
    Stage *stage;
    int *order;
    double *workspace;
    double work;
    unsigned long calls, parallel_calls, fallbacks;
    int max_workers;
};

static int pair_compare(const void *a, const void *b)
{
    const Pair *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

void ng_klu_plan_free(NgKluPlan **handle)
{
    NgKluPlan *p = *handle;
    if (!p) return;
    if (p->calls)
        fprintf(stderr, "KLU optimized summary: calls=%lu parallel_calls=%lu fallbacks=%lu max_workers=%d simd_bits=%d\n",
                p->calls, p->parallel_calls, p->fallbacks, p->max_workers, p->vectorized);
    free(p->column); free(p->run); free(p->stage); free(p->order); free(p->workspace);
    free(p); *handle = NULL;
}

#ifdef NG_KLU_AVX2
__attribute__((target("avx2,no-fma")))
static void dense_avx2(double *x, const double *a, double factor, int n)
{
    int i = 0;
    __m256d f = _mm256_set1_pd(factor);
    for (; i + 4 <= n; i += 4) {
        __m256d y = _mm256_loadu_pd(x+i), v = _mm256_loadu_pd(a+i);
        _mm256_storeu_pd(x+i, _mm256_sub_pd(y, _mm256_mul_pd(v, f)));
    }
    for (; i < n; ++i) x[i] -= a[i] * factor;
}

__attribute__((target("avx512f,no-fma")))
static void dense_avx512(double *x, const double *a, double factor, int n)
{
    int i = 0;
    __m512d f = _mm512_set1_pd(factor);
    for (; i + 8 <= n; i += 8) {
        __m512d y = _mm512_loadu_pd(x+i), v = _mm512_loadu_pd(a+i);
        _mm512_storeu_pd(x+i, _mm512_sub_pd(y, _mm512_mul_pd(v, f)));
    }
    for (; i < n; ++i) x[i] -= a[i] * factor;
}

__attribute__((target("avx512f,no-fma")))
static void sparse_avx512(double *x, const double *a, const int *rows, double factor, int n)
{
    int i = 0;
    __m512d f = _mm512_set1_pd(factor);
    for (; i + 8 <= n; i += 8) {
        __m256i indices = _mm256_loadu_si256((const __m256i *)(rows+i));
        __m512d y = _mm512_i32gather_pd(indices, x, 8), v = _mm512_loadu_pd(a+i);
        _mm512_i32scatter_pd(x, indices, _mm512_sub_pd(y, _mm512_mul_pd(v, f)), 8);
    }
    for (; i < n; ++i) x[rows[i]] -= a[i] * factor;
}

__attribute__((target("avx2,no-fma")))
static void sparse_avx2(double *x, const double *a, const int *rows, double factor, int n)
{
    int i = 0;
    double result[4];
    __m256d f = _mm256_set1_pd(factor);
    for (; i + 4 <= n; i += 4) {
        __m128i indices = _mm_loadu_si128((const __m128i *)(rows+i));
        __m256d y = _mm256_i32gather_pd(x, indices, 8), v = _mm256_loadu_pd(a+i);
        _mm256_storeu_pd(result, _mm256_sub_pd(y, _mm256_mul_pd(v, f)));
        x[rows[i]]=result[0]; x[rows[i+1]]=result[1];
        x[rows[i+2]]=result[2]; x[rows[i+3]]=result[3];
    }
    for (; i < n; ++i) x[rows[i]] -= a[i] * factor;
}
#endif

static void update_column(const NgKluPlan *plan, const Column *column,
                          double *x, double factor)
{
    int i, j;
    if (column->llen < 16 || !plan->vectorized) {
        for (i = 0; i < column->llen; ++i) x[column->li[i]] -= column->lx[i] * factor;
        return;
    }
#ifdef NG_KLU_AVX2
    if (column->long_values < column->llen*0.8) {
        if (plan->vectorized == 512)
            sparse_avx512(x, column->lx, column->li, factor, column->llen);
        else sparse_avx2(x, column->lx, column->li, factor, column->llen);
        return;
    }
#endif
    for (i = column->first_run; i < column->first_run + column->runs; ++i) {
        const Run *r = plan->run+i;
#ifdef NG_KLU_AVX2
        if (plan->vectorized && r->length >= 8) {
            if (plan->vectorized == 512)
                dense_avx512(x+r->row, column->lx+r->offset, factor, r->length);
            else dense_avx2(x+r->row, column->lx+r->offset, factor, r->length);
            continue;
        }
#endif
        for (j = 0; j < r->length; ++j)
            x[r->row+j] -= column->lx[r->offset+j] * factor;
    }
}

static NgKluPlan *make_plan(int *ap, klu_symbolic *s, klu_numeric *num, int threads)
{
    int n = s->n, b, c, k1, k, j, l, count = 0, depth = 0, offset = 0, nruns = 0;
    int *level = NULL, *width = NULL, *cursor = NULL;
    double *cost = NULL;
    Pair *pairs = NULL;
    NgKluPlan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->numeric = num; p->symbolic = s; p->n = n; p->maxblock = s->maxblock;
    p->threads = threads;
    p->column = calloc(n, sizeof(Column));
    p->run = calloc((size_t)num->lnz+1, sizeof(Run));
    p->stage = calloc((size_t)n+1, sizeof(Stage));
    p->order = malloc((size_t)n * sizeof(int));
    level = calloc(n, sizeof(int)); width = calloc((size_t)n+1, sizeof(int));
    cursor = calloc((size_t)n+1, sizeof(int)); cost = calloc((size_t)n+1, sizeof(double));
    pairs = malloc((size_t)s->maxblock * sizeof(Pair));
    if (!p->column || !p->run || !p->stage || !p->order ||
        !level || !width || !cursor || !cost || !pairs) goto fail;
#ifdef NG_KLU_AVX2
    p->vectorized = __builtin_cpu_supports("avx512f") ? 512 : __builtin_cpu_supports("avx2") ? 256 : 0;
#endif
#ifdef NG_KLU_TEST_MAX_SIMD
    if (p->vectorized > NG_KLU_TEST_MAX_SIMD) p->vectorized = NG_KLU_TEST_MAX_SIMD;
#endif
    /* Sort only entries WITHIN each L column. They update distinct rows, so
     * this changes memory access order but not any row's accumulation order.
     * Do not sort U: its traversal is the original elimination sequence. */
    for (b = 0; b < s->nblocks; ++b) {
        k1 = s->R[b];
        for (c = k1; c < s->R[b+1]; ++c) {
            Column *col = p->column+c;
            Unit *lu = (Unit *)num->LUbx[b];
            col->start = k1; col->size = s->R[b+1]-k1; col->oldcol = s->Q[c];
            col->cost = ap[col->oldcol+1]-ap[col->oldcol];
            if (col->size > 1) {
                GET_POINTER(lu, num->Lip, num->Llen, col->li, col->lx, c, col->llen);
                GET_POINTER(lu, num->Uip, num->Ulen, col->ui, col->ux, c, col->ulen);
                for (k = 0; k < col->llen; ++k) {
                    pairs[k].row = col->li[k]; pairs[k].value = col->lx[k];
                }
                qsort(pairs, col->llen, sizeof(Pair), pair_compare);
                for (k = 0; k < col->llen; ++k) {
                    col->li[k] = pairs[k].row; col->lx[k] = pairs[k].value;
                    if (pairs[k].row < 0 || pairs[k].row >= col->size ||
                        (k && pairs[k-1].row >= pairs[k].row)) goto fail;
                }
                col->first_run = nruns;
                for (k = 0; k < col->llen; k += count) {
                    for (count = 1; k+count < col->llen && col->li[k+count] == col->li[k]+count; ++count) {}
                    p->run[nruns++] = (Run){k, col->li[k], count}; ++col->runs;
                    if (count >= 8) col->long_values += count;
                }
                col->cost += col->llen + col->ulen;
                for (k = 0; k < col->ulen; ++k) {
                    j = k1 + col->ui[k];
                    if (j < k1 || j >= c) goto fail;
                    if (level[j]+1 > level[c]) level[c] = level[j]+1;
                    col->cost += num->Llen[j];
                }
            }
            ++width[level[c]]; cost[level[c]] += col->cost; p->work += col->cost;
            if (level[c] > depth) depth = level[c];
        }
    }
    for (l = 0; l <= depth; ++l) { cursor[l] = offset; offset += width[l]; }
    for (c = 0; c < n; ++c) p->order[cursor[level[c]]++] = c;
    offset = 0;
    for (l = 0; l <= depth; ++l) {
        int parallel = threads > 1 && width[l] >= 4 && cost[l] >= 32768;
        if (!parallel && p->stages && !p->stage[p->stages-1].parallel)
            p->stage[p->stages-1].end += width[l];
        else {
            p->stage[p->stages++] = (Stage){offset, offset+width[l], parallel};
            p->parallel_stages += parallel;
        }
        offset += width[l];
    }
    p->eligible = p->work >= 100000 && s->maxblock > 1;
    if (p->eligible) {
        int workers = p->parallel_stages ? threads : 1;
        size_t values = (size_t)workers * s->maxblock;
        if (values > ((size_t)512*1024*1024)/sizeof(double)) goto fail;
        p->workspace = calloc(values, sizeof(double));
        if (!p->workspace) goto fail;
        fprintf(stderr, "KLU optimized plan: n=%d blocks=%d maxblock=%d levels=%d stages=%d parallel_stages=%d requested_threads=%d work=%.0f runs=%d simd_bits=%d\n",
                n, s->nblocks, s->maxblock, depth+1, p->stages, p->parallel_stages,
                threads, p->work, nruns, p->vectorized);
    }
    free(level); free(width); free(cursor); free(cost); free(pairs);
    return p;
fail:
    free(level); free(width); free(cursor); free(cost); free(pairs);
    ng_klu_plan_free(&p);
    return NULL;
}

static int factor_column(NgKluPlan *p, int c, int *ap, int *ai, double *ax,
                         klu_numeric *num, int scale, double *x)
{
    Column *col = p->column+c;
    int k, row, j, off = num->Offp[c], own = c-col->start;
    double value, pivot = 0;
    double *offx = num->Offx, *diag = num->Udiag;
    for (k = ap[col->oldcol]; k < ap[col->oldcol+1]; ++k) {
        row = num->Pinv[ai[k]] - col->start;
        value = scale > 0 ? ax[k]/num->Rs[ai[k]] : ax[k];
        if (row < 0) offx[off++] = value;
        else if (col->size == 1) pivot = value;
        else x[row] = value;
    }
    if (col->size == 1) { diag[c] = pivot; return isfinite(pivot); }
    for (k = 0; k < col->ulen; ++k) {
        j = col->ui[k]; value = x[j]; x[j] = 0; col->ux[k] = value;
        update_column(p, p->column+col->start+j, x, value);
    }
    pivot = x[own]; x[own] = 0; diag[c] = pivot;
    if (pivot == 0 || !isfinite(pivot)) return 0;
    for (k = 0; k < col->llen; ++k) {
        row = col->li[k]; col->lx[k] = x[row]/pivot; x[row] = 0;
    }
    return 1;
}

int ng_klu_refactor(NgKluPlan **handle, int *ap, int *ai, double *ax,
                    klu_symbolic *s, klu_numeric *num, klu_common *common, int threads)
{
    NgKluPlan *p = *handle;
    int c, st, i, failed = 0, workers = 1, scale;
    if (!s || !num || !common || common->status == KLU_EMPTY_MATRIX || threads < 1 || threads > 256)
        return klu_refactor(ap, ai, ax, s, num, common);
#ifdef _OPENMP
    if (omp_in_parallel()) return klu_refactor(ap, ai, ax, s, num, common);
#else
    threads = 1;
#endif
    if (p && (p->numeric != num || p->symbolic != s || p->threads != threads))
        ng_klu_plan_free(handle);
    if (!*handle) *handle = make_plan(ap, s, num, threads);
    p = *handle;
    if (!p || !p->eligible) return klu_refactor(ap, ai, ax, s, num, common);
    ++p->calls;
    common->status = KLU_OK; common->numerical_rank = EMPTY;
    common->singular_col = EMPTY; common->nrealloc = 0;
    scale = common->scale;
    if (scale > 0 && !num->Rs) {
        num->Rs = klu_malloc(s->n, sizeof(double), common);
        if (!num->Rs) return 0;
    } else if (scale <= 0) num->Rs = klu_free(num->Rs, s->n, sizeof(double), common);
    if (scale >= 0 && !klu_scale(scale, s->n, ap, ai, ax, num->Rs, NULL, common)) return 0;
#ifdef _OPENMP
    if (p->parallel_stages) {
        ++p->parallel_calls;
        workers = 0;
#pragma omp parallel num_threads(threads) shared(workers, failed)
        {
            int stage_index, pos, local_failed;
            unsigned int work_done = 0;
            double *x = p->workspace + (size_t)omp_get_thread_num()*p->maxblock;
            for (stage_index = 0; stage_index < p->stages; ++stage_index) {
                Stage *stage = p->stage+stage_index;
                if (stage->parallel) {
#pragma omp for schedule(dynamic, 4) reduction(|:failed)
                    for (pos = stage->begin; pos < stage->end; ++pos) {
                        failed |= !factor_column(p, p->order[pos], ap, ai, ax, num, scale, x);
                        ++work_done;
                    }
                } else {
#pragma omp master
                    {
                        local_failed = 0;
                        for (pos = stage->begin; pos < stage->end; ++pos) {
                            local_failed |= !factor_column(p, p->order[pos], ap, ai, ax, num, scale, x);
                            ++work_done;
                        }
                        failed |= local_failed;
                    }
#pragma omp barrier
                }
                if (failed) break;
            }
#pragma omp atomic update
            workers += work_done > 0;
        }
    } else
#endif
    {
        for (st = 0; st < p->stages && !failed; ++st)
            for (i = p->stage[st].begin; i < p->stage[st].end; ++i)
                failed |= !factor_column(p, p->order[i], ap, ai, ax, num, scale, p->workspace);
    }
    if (workers > p->max_workers) p->max_workers = workers;
    if (failed) {
        ++p->fallbacks;
        memset(p->workspace, 0, (size_t)(p->parallel_stages ? threads : 1)*p->maxblock*sizeof(double));
        /* The legacy routine recomputes all factors in its original order and
         * reports its original singular rank/column and halt policy. */
        return klu_refactor(ap, ai, ax, s, num, common);
    }
    if (scale > 0) {
        double *tmp = num->Xwork;
        for (c = 0; c < s->n; ++c) tmp[c] = num->Rs[num->Pnum[c]];
        for (c = 0; c < s->n; ++c) num->Rs[c] = tmp[c];
    }
    return 1;
}

#endif /* !NGSPICE_SYSTEM_KLU */
