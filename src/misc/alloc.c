/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* This is a test file including a local garbage collector and removal.
It might be especially useful, if the ngspice shared library (.dll or .so)
is loaded and unloaded several times by a master program.
At ngspice initialization mem_init() is called setting up a hash table
to store memory addresses.
Each time calloc() is called, the resulting address is stored.
Each time free() is called, the address is removed.
For realloc(), the old address is deleted, the new one is stored,
if there is a change in the address.
Upon exiting the program, mem_delete() is called, deleting all memory
at addresses still in the hash table, then the hash table itself is removed.
The Windows dll uses DllMain() to call mem_delete() exit, the LINUX .so
will use void __attribute__((destructor)) mem_delete(void) or atexit()
(both are not tested so far).

The master program has to make copies of all the data that have to be kept
for further use before detaching the shared lib! */

// #define DB_FULL /* uncomment for debugging output, all addresses to files */
#ifdef DB_FULL
#define DB
#else
#define DB /* uncomment for printing some debugging output */
#endif

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

/* add some debugging printout */
#ifdef DB
static int mem_in = 0, mem_out = 0, mem_freed = 0;
#endif
#ifdef DB_FULL
static FILE *alloclog, *freelog, *finallog;
#endif

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
  /* remove the old and add the new memory address */
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
    gc_is_on = 0;
    memory_table = nghash_init_pointer(1024);
    gc_is_on = 1;
#ifdef DB_FULL
    alloclog = fopen("alloc_log.txt", "wt");
    freelog = fopen("free_log.txt", "wt");
    finallog = fopen("final_log.txt", "wt");
#endif
}

/* add to counter and hash table if memory is allocated */
static int memsaved(void *ptr) {
    if (gc_is_on) {
        gc_is_on = 0;

        if (nghash_insert(memory_table, ptr, NULL) == NULL) {
#ifdef DB
            mem_in++;
#ifdef DB_FULL
            fprintf(alloclog, "0x%p\n", ptr);
#endif
#endif
        } else
            fprintf(stderr, "Could not insert item into hashtable at 0x%p\n", ptr);
        gc_is_on = 1;
    }
    return OK;
}

/* add to counter and remove from hash table if memory is deleted */
static void memdeleted(const void *ptr) {
    if (gc_is_on) {
        gc_is_on = 0;
        if (nghash_delete_special(memory_table, (void*)ptr) == NULL) {
#ifdef DB
            mem_out++;
#ifdef DB_FULL
            fprintf(freelog, "0x%p\n", ptr);
#endif
        }
        else
            fprintf(stderr, "Could not delete item from hashtable at 0x%p\n", ptr);
#else
        }
#endif
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
    if (key) {
        free(key);
        key = NULL;
#ifdef DB
        mem_freed++;
#endif
    }
}

/* free hash table, all entries and then the table itself */
void mem_delete(void) {
#ifdef DB
    char buf[128];
    printf("memory allocated %d times, freed %d times\n", mem_in, mem_out);
    printf("Size of hash table to be freed: %d entries.\n", nghash_get_size(memory_table));
#ifdef DB_FULL
    void *data, *key;
    data = nghash_enumeratek(memory_table, &key, TRUE);
    for (void *hkey = key; hkey;) {
        fprintf(finallog, "0x%p\n", hkey);
        data = nghash_enumeratek(memory_table, &hkey, FALSE);
    }
    fclose(alloclog);
    fclose(freelog);
    fclose(finallog);
#endif
#endif
    gc_is_on = 0;
    nghash_free(memory_table, NULL, my_key_free);
#ifdef DB
    /* printf via sh_printf will need some info from variables that have
    been deleted already, therefore we use fputs */
    sprintf(buf, "Number of addresses freed: %d entries.\n", mem_freed);
    fputs(buf, stdout);
#endif
}

