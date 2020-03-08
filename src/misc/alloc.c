/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
/* for thread handling */
#ifdef _WIN32
#include <windows.h>
#endif

#include <stdint.h>

/*
 * Memory alloction functions
 */
#include "ngspice/alloc.h"
#include "ngspice/ngspice.h"
#include "ngspice/cpextern.h"

#if defined HAS_WINGUI || defined SHARED_MODULE
#define EXIT(rc) controlled_exit(rc)
#else
#define EXIT(rc) exit(rc)
#endif

static void overflow_error(size_t num, size_t size);
static inline int product_overflow(size_t a, size_t b, size_t *p_n);


#ifdef SHARED_MODULE
#ifndef HAVE_LIBPTHREAD
#ifdef SRW
#define mutex_lock(a) AcquireSRWLockExclusive(a)
#define mutex_unlock(a) ReleaseSRWLockExclusive(a)
typedef SRWLOCK mutexType;
#else
#define mutex_lock(a) EnterCriticalSection(a)
#define mutex_unlock(a) LeaveCriticalSection(a)
typedef CRITICAL_SECTION mutexType;
#endif
extern mutexType allocMutex;
#else
#include <pthread.h>
#define mutex_lock(a) pthread_mutex_lock(a)
#define mutex_unlock(a) pthread_mutex_unlock(a)
typedef pthread_mutex_t mutexType;
extern mutexType allocMutex;
#endif
#endif


#ifdef HAVE_LIBGC

#include <gc/gc.h>

void *tmalloc(size_t num)
{
    if (GC_malloc(num) == NULL) {
        EXIT(EXIT_FAILURE);
    }
} /* end of function tmalloc */


void *tcalloc(size_t num, size_t count)
{
    if (GC_calloc(num, count) == NULL) {
        EXIT(EXIT_FAILURE);
    }
} /* end of function tmalloc */



void *trealloc(size_t num, size_t count)
{
    if (GC_calloc(num, count) == NULL) {
        EXIT(EXIT_FAILURE);
    }
} /* end of function tmalloc */


/* Free is noop for GC */
void txfree(p)
{
    NG_IGNORE(p);
    return;
} /* end of function txfree */



#else
/*saj For Tcl module locking*/
#ifdef TCL_MODULE
#include <tcl.h>
#endif

/* Malloc num bytes and initialize to zero. Fatal error if the space can't
 * be tmalloc'd. Return NULL for a request for 0 bytes.
 */
void *tmalloc(size_t num)
{
    if (num == 0) {
        return NULL;
    }
    void * const p = tmalloc_raw(num);
    if (p == NULL) {
        (void) fprintf(cp_err, "malloc: Internal Error: can't allocate "
                "%zd bytes. \n", num);
        EXIT(EXIT_FAILURE);
    }
    return memset(p, 0, num);
} /* end of function tmalloc */



/* A "raw" malloc with mutex protection. */
void *tmalloc_raw(size_t num)
{
    if (num == 0) { /* 0-byte request */
        return NULL;
    }

/*saj*/
#ifdef TCL_MODULE
    Tcl_Mutex *alloc;
    alloc = Tcl_GetAllocMutex();
    Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
    mutex_lock(&allocMutex);
#endif
    void * const p = malloc(num);
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
    mutex_unlock(&allocMutex);
#endif
    return p;
} /* end of function tmalloc_raw */



/* calloc with mutex protection */
void *tcalloc(size_t num, size_t size) {
    size_t n;
    if (product_overflow(num, size, &n)) {
        overflow_error(num, size);
        /* Fatal. No return */
    }

    return tmalloc(n);
} /* end of function tcalloc */



/* A "raw" calloc with mutex protection built from tmalloc_raw(). */
void *tcalloc_raw(size_t num, size_t size)
{
    size_t n;
    if (product_overflow(num, size, &n)) {
        return NULL;
    }
    void * const p = tmalloc_raw(n);
    return memset(p, 0, n);
} /* end of function tmalloc_raw */



/* Realloc with mutex protection and exit if failure */
void *trealloc(void *ptr, size_t num)
{
    void *p_new = trealloc_raw(ptr, num);

    if (num != 0 && p_new == NULL) {
        (void) fprintf(stderr, "realloc: Internal Error: "
                "can't allocate %zd bytes.\n",
                num);
        EXIT(EXIT_FAILURE);
    }

    return p_new;
} /* end of function trealloc */



/* A "raw" realloc with mutex protection */
void *trealloc_raw(void *ptr, size_t num)
{
/*saj*/
    if (num == 0) { /* Acts like free() */
        if (ptr != NULL) {
            txfree((void*) ptr);
        }
        return NULL;
    }

    if (ptr == NULL) { /* Acts like malloc() */
        return tmalloc_raw(num);
    }

#ifdef TCL_MODULE
    Tcl_Mutex *alloc;
    alloc = Tcl_GetAllocMutex();
    Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
    mutex_lock(&allocMutex);
#endif
    void * const p_new = realloc(ptr, num);
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
    mutex_unlock(&allocMutex);
#endif

    return p_new;
} /* end of function trealloc_raw */



/* Free with mutex protection */
void txfree(void *ptr)
{
    if (ptr == NULL) {
        return;
    }

/*saj*/
#ifdef TCL_MODULE
    Tcl_Mutex *alloc;
    alloc = Tcl_GetAllocMutex();
    Tcl_MutexLock(alloc);
#endif
#ifdef SHARED_MODULE
    mutex_lock(&allocMutex);
#endif
    free(ptr);
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
    mutex_unlock(&allocMutex);
#endif
} /* end of function txfree */




/* This function returns the product of a and b if it does not overflow.
 *
 * Return codes
 * 0: No overflow
 * 1: overflow
 */
static inline int product_overflow(size_t a, size_t b, size_t *p_n)
{
    /* Some overflow conditions:
     * a == SIZE_MAX and b > 1
     * a > 1 and b == SIZE_MAX
     * a * b < a
     * a * b < b
     */
    if ((a == SIZE_MAX && b > 1) || (a > 1 && b == SIZE_MAX)) {
        return +1;
    }

    const size_t n = a * b;
    if (n < a || n < b) {
        return +1;
    }

    *p_n = n;
    return 0;
} /* end of function product_overflow */



/* Print error related to allocating a product that cannot fit in a
 * size_t and exit. This function does not return. */
static void overflow_error(size_t num, size_t size)
{
    (void) fprintf(cp_err, "Cannot allocate %zu X %zu bytes: "
            "Product exceeds largest size_t = %zu.\n",
            num, size, SIZE_MAX);
    EXIT(EXIT_FAILURE);
} /* end of function overflow_error */


#endif /* #ifndef HAVE_LIBGC */

