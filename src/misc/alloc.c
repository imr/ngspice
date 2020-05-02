/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* for SIZE_MAX */
#include <stdint.h>

/* for thread handling */
#if defined __MINGW32__ || defined _MSC_VER
#include <windows.h>
#endif

/*
 * Memory alloction functions
 */
#include "ngspice/ngspice.h"


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


#ifndef HAVE_LIBGC

/*saj For Tcl module locking*/
#ifdef TCL_MODULE
#include <tcl.h>
#endif

/* Malloc num bytes and initialize to zero. Fatal error if the space can't
 * be tmalloc'd.   Return NULL for a request for 0 bytes.
 */

/* New implementation of tmalloc, it uses calloc and does not call memset()  */
void *
tmalloc(size_t num)
{
  void *s;
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
#endif
    if (!num)
      return NULL;
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
  mutex_lock(&allocMutex);
#endif
    s = calloc(num,1);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
  mutex_unlock(&allocMutex);
#endif
    if (!s){
      fprintf(stderr,"malloc: Internal Error: can't allocate %ld bytes. \n",(long)num);
#if defined HAS_WINGUI || defined SHARED_MODULE
      controlled_exit(EXIT_FAILURE);
#else
      exit(EXIT_FAILURE);
#endif
    }
    return(s);
}


void *
trealloc(const void *ptr, size_t num)
{
  void *s;
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
#endif
  if (!num) {
    if (ptr)
      free((void*) ptr);
    return NULL;
  }

  if (!ptr)
    s = tmalloc(num);
  else {
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
  mutex_lock(&allocMutex);
#endif
    s = realloc((void*) ptr, num);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
  mutex_unlock(&allocMutex);
#endif
  }
  if (!s) {
    fprintf(stderr,"realloc: Internal Error: can't allocate %ld bytes.\n", (long)num);
#if defined HAS_WINGUI || defined SHARED_MODULE
      controlled_exit(EXIT_FAILURE);
#else
      exit(EXIT_FAILURE);
#endif
  }
  return(s);
}


void
txfree(const void *ptr)
{
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
  Tcl_MutexLock(alloc);
#endif
#ifdef SHARED_MODULE
  mutex_lock(&allocMutex);
#endif
	if (ptr)
		free((void*) ptr);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
  mutex_unlock(&allocMutex);
#endif
} /* end of function txfree */

#if 0
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
    (void) fprintf(stderr, "Cannot allocate %zu X %zu bytes: "
            "Product exceeds largest size_t = %zu.\n",
            num, size, SIZE_MAX);
#if defined HAS_WINGUI || defined SHARED_MODULE
    controlled_exit(EXIT_FAILURE);
#else
    exit(EXIT_FAILURE);
#endif
} /* end of function overflow_error */
#endif

#endif /* #ifndef HAVE_LIBGC */
