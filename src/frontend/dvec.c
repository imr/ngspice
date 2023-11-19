

#if defined(__MINGW32__) || defined(_MSC_VER)
#include <windows.h>
#endif

#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"

#if defined SHARED_MODULE

/*Use Windows threads if on W32 without pthreads*/
#ifndef HAVE_LIBPTHREAD

#if defined(__MINGW32__) || defined(_MSC_VER)
//#if defined(_MSC_VER)
#ifdef SRW
#define mutex_lock(a) AcquireSRWLockExclusive(a)
#define mutex_unlock(a) ReleaseSRWLockExclusive(a)
typedef SRWLOCK mutexType;
#else
#define mutex_lock(a) EnterCriticalSection(a)
#define mutex_unlock(a) LeaveCriticalSection(a)
typedef CRITICAL_SECTION mutexType;
#endif
#define thread_self() GetCurrentThread()
#define threadid_self() GetCurrentThreadId()
typedef HANDLE threadId_t;
#define WIN_THREADS
#define THREADS

#endif

#else

#include <pthread.h>
#define mutex_lock(a) pthread_mutex_lock(a)
#define mutex_unlock(a) pthread_mutex_unlock(a)
#define thread_self() pthread_self()
#define threadid_self() 0  //FIXME t.b.d.
typedef pthread_mutex_t mutexType;
typedef pthread_t threadId_t;
#define THREADS
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static bool cont_condition;

#endif

extern mutexType vecreallocMutex;

#endif

struct dvec *dvec_alloc(/* NOT const -- assigned to char */ char *name,
        int type, short flags, int length, void *storage)
{
    struct dvec * const rv = TMALLOC(struct dvec, 1);

    /* If the allocation failed, return NULL as a failure flag.
     * As of 2019-03, TMALLOC will not return on failure, so this check is
     * redundant, but it may be useful if it is decided to allow the
     * allocation functions to return NULL on failure and handle recovery
     * by the calling functions */
    if (!rv) {
        return NULL;
    }

    /* Set all fields to 0 */
    ZERO(rv, struct dvec);

    /* Set information on the vector from parameters. Note that storage for
     * the name string belongs to the dvec when this function returns. */
    rv->v_name = name;
    rv->v_type = type;
    rv->v_flags = flags;
    rv->v_length = length;
    rv->v_alloc_length = length;
    rv->v_numdims = 1; /* Assume 1 D */
    rv->v_dims[0] = length;

    if (length == 0) { /* Redundant due to ZERO() call above */
        rv->v_realdata = NULL;
        rv->v_compdata = NULL;
    }
    else if (flags & VF_REAL) {
        /* Vector consists of real data. Use the supplied storage if given
         * or allocate if not */
        rv->v_realdata = storage
            ? (double *) storage
            : TMALLOC(double, length);
        rv->v_compdata = NULL;
    }
    else if (flags & VF_COMPLEX) {
        /* Vector holds complex data. Perform actions as for real data */
        rv->v_realdata = NULL;
        rv->v_compdata = storage
            ? (ngcomplex_t *) storage
            : TMALLOC(ngcomplex_t, length);
    }

    /* Set remaining fields to none/unknown. Again not required due to
     * the ZERO() call */
    rv->v_plot = NULL;
    rv->v_scale = NULL;

    return rv;
} /* end of function dvec_alloc */


/* Resize dvec to length if storage is NULL or replace
 * its existing allocation with storage if not
 */
void dvec_realloc(struct dvec *v, int length, void *storage)
{
    if (isreal(v)) {
        if (storage) {
            tfree(v->v_realdata);
            v->v_realdata = (double *) storage;
        }
        else {
            v->v_realdata = TREALLOC(double, v->v_realdata, length);
        }
    }
    else {
        if (storage) {
            tfree(v->v_compdata);
            v->v_compdata = (ngcomplex_t *) storage;
        }
        else {
            v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);
        }
    }

    v->v_length = length;
    v->v_alloc_length = length;
} /* end of function dvec_realloc */

/* called from plotAddReal(Complex)Value, to increase
   storage for result vectors.
   In shared ngspice this may be locked, e.g. during plotting in the primary
   thread, while the simulation is running in the background thread. Locking and unlocking 
   is done by API functions ngSpice_LockRealloc(), ngSpice_UnlockRealloc(). */
void dvec_extend(struct dvec *v, int length)
{
#if defined SHARED_MODULE
    mutex_lock(&vecreallocMutex);
#endif
    if (isreal(v)) {
        v->v_realdata = TREALLOC(double, v->v_realdata, length);
    }
    else {
        v->v_compdata = TREALLOC(ngcomplex_t, v->v_compdata, length);
    }
    v->v_alloc_length = length;
#if defined SHARED_MODULE
    mutex_unlock(&vecreallocMutex);
#endif
} /* end of function dvec_extend */



void dvec_trunc(struct dvec *v, int length)
{
    /* Ensure valid */
    if (v->v_alloc_length <= length) {
        v->v_length = length;
    }
} /* end of function dvec_trunc */



void dvec_free(struct dvec *v)
{
    /* Check for freed vector */
    if (v == (struct dvec *) NULL) {
        return;
    }

    /* Free the various allocations */
    if (v->v_name) {
        txfree(v->v_name);
    }
    if (v->v_realdata) {
        txfree(v->v_realdata);
    }
    else if (v->v_compdata) { /* if data real, not complex */
        txfree(v->v_compdata);
    }
    txfree(v);
} /* end of function dvec_free */



