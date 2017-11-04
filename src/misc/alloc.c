/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* for thread handling */
#if defined __MINGW32__ || defined _MSC_VER
#include <windows.h>
#endif

/*
 * Memory alloction functions
 */
#include "ngspice/ngspice.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hash.h"


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

#if defined(SHARED_MODULE) && (!defined(_MSC_VER) && !defined(__MINGW32__))
void __attribute__((constructor)) mem_init(void);
void __attribute__((destructor)) mem_delete(void);
#else
void mem_init(void);
void mem_delete(void);
#endif

static int memsaved(void *ptr);
static void memdeleted(const void *ptr);

int gc_is_on = 0;

static int mem_in = 0, mem_out = 0;

NGHASHPTR memory_table;

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
    memsaved(s);
    return(s);
}


void *
trealloc(const void *ptr, size_t num)
{
  void *s;

#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
#endif
  if (!num) {
     if (ptr) {
         memdeleted(ptr);
         tfree(ptr);
      }
    return NULL;
  }

  if (!ptr)
    s = tmalloc(num);
  else {

#ifdef TCL_MODULE
    Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
  mutex_lock(&allocMutex);
#endif
    s = realloc((void*) ptr, num);

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
  if (s != ptr) {
      memdeleted(ptr);
      memsaved(s);
  }
  return(s);
}


void
txfree(const void *ptr)
{
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
  Tcl_MutexLock(alloc);
#endif
#ifdef SHARED_MODULE
  mutex_lock(&allocMutex);
#endif
  if (ptr) {
      memdeleted(ptr);
      free((void*)ptr);
  }

#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
  mutex_unlock(&allocMutex);
#endif
}

/* for replacing calloc() in SP_CALLOC from spdefs.h */
void *
tcalloc(size_t num, size_t stype)
{
    void *s;

#ifdef TCL_MODULE
    Tcl_Mutex *alloc;
    alloc = Tcl_GetAllocMutex();
#endif
    if (!num)
       return NULL;
#ifdef TCL_MODULE
    Tcl_MutexLock(alloc);
#elif defined SHARED_MODULE
    mutex_lock(&allocMutex);
#endif
    s = calloc(num, stype);
#ifdef TCL_MODULE
    Tcl_MutexUnlock(alloc);
#elif defined SHARED_MODULE
    mutex_unlock(&allocMutex);
#endif
    if (!s) {
       fprintf(stderr, "calloc: Internal Error: can't allocate %ld bytes. \n", (long)num);
#if defined HAS_WINGUI || defined SHARED_MODULE
       controlled_exit(EXIT_FAILURE);
#else
       exit(EXIT_FAILURE);
#endif
    }
    memsaved(s);
    return(s);
}

#endif

/* initialize hash table to store allocated mem addresses */
void mem_init(void) {
    memory_table = nghash_init_pointer(1024);
    gc_is_on = 1;
    return OK;
}

/* add to counter and hash table if memory is allocated */
static int memsaved(void *ptr) {
    if (gc_is_on) {
        gc_is_on = 0;
        mem_in++;
        nghash_insert(memory_table, ptr, NULL);
        gc_is_on = 1;
    }
    return OK;
}

/* add to counter and remove from hash table if memory is deleted */
static void memdeleted(const void *ptr) {
    if (gc_is_on) {
        gc_is_on = 0;
        mem_out++;
        nghash_delete(memory_table, (void*)ptr);
        gc_is_on = 1;
    }
}

/* helper functions */
void my_free_func(void *data)
{
    if (data)
        free(data);
}

void my_key_free(void * key)
{
    if (key)
        free(key);
}

/* free hash table */
void mem_delete(void) {
    gc_is_on = 0;
    printf("mem allocated %d times, deleted %d times\n", mem_in, mem_out);
    nghash_free(memory_table, NULL, my_key_free);
    return OK;
}

