/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
$Id$
**********/

/*
 * Memory alloction functions
 */
#include "ngspice.h"

#ifndef HAVE_LIBGC

/*saj For Tcl module locking*/
#ifdef TCL_MODULE
#include <tcl.h>
#endif

#if defined(HAS_WINDOWS) || defined(HAS_TCLWIN)
#if defined(_MSC_VER) || defined(__MINGW32__)
#undef BOOLEAN
#include <windows.h>
extern HANDLE outheap;
extern void winmessage(char* new_msg);
#endif
#endif


/* Malloc num bytes and initialize to zero. Fatal error if the space can't
 * be tmalloc'd.   Return NULL for a request for 0 bytes.
 */

/* New implementation of tmalloc, it uses calloc and does not call bzero()  */

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
#endif
    s = calloc(num,1);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#endif
    if (!s){
      fprintf(stderr,"malloc: Internal Error: can't allocate %ld bytes. \n",(long)num);
#ifdef HAS_WINDOWS
      winmessage("Fatal error in SPICE");
#endif
      exit(EXIT_BAD);
    }
    return(s);
}

/* Original Berkeley Implementation */
/*
void *
tmalloc(size_t num)
{
     void *s;

    if (!num)
	return NULL;

    s = malloc((unsigned) num);
    if (!s) {
        fprintf(stderr, 
		"malloc: Internal Error: can't allocate %d bytes.\n", num);
        exit(EXIT_BAD);
    }

    bzero(s, num);

    return(s);
}

void *
trealloc(void *str, size_t num)
{
    void *s;

    if (!num) {
	if (str)
		free(str);
	return NULL;
    }

    if (!str)
	s = tmalloc(num);
    else
        s = realloc(str, (unsigned) num);

    if (!s) {
        fprintf(stderr, 
		"realloc: Internal Error: can't allocate %d bytes.\n", num);
        exit(EXIT_BAD);
    }
    return(s);
}

*/

void *
trealloc(void *ptr, size_t num)
{
  void *s;
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
#endif
  if (!num) {
    if (ptr)
      free(ptr);
    return NULL;
  }

  if (!ptr)
    s = tmalloc(num);
  else {
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexLock(alloc);
#endif
    s = realloc(ptr, num);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#endif
  }
  if (!s) {
    fprintf(stderr,"realloc: Internal Error: can't allocate %ld bytes.\n", (long)num);
#ifdef HAS_WINDOWS
    winmessage("Fatal error in SPICE");
#endif
    exit(EXIT_BAD);
  }
  return(s);
}

/* realloc using the output heap. 
   Function is used in outitf.c to prevent heap fragmentation 
   An additional heap outheap is used to store the plot output data.
*/
#if defined(HAS_TCLWIN)
#if defined(_MSC_VER) || defined(__MINGW32__)
void *
hrealloc(void *ptr, size_t num)
{
  void *s;
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
#endif
  if (!num) {
    if (ptr)
      free(ptr);
    return NULL;
  }

  if (!ptr)
    s = HeapAlloc(outheap, HEAP_ZERO_MEMORY, num);
  else {
/*saj*/
#ifdef TCL_MODULE
    Tcl_MutexLock(alloc);
#endif
   s = HeapReAlloc(outheap, HEAP_ZERO_MEMORY, ptr, num);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#endif
  }
  if (!s) {
    fprintf(stderr,"HeapReAlloc: Internal Error: can't allocate %ld bytes.\n", (long)num);
    winmessage("Fatal error in SPICE");
    exit(EXIT_BAD);
  }
  return(s);
}
#endif
#endif


void
txfree(void *ptr)
{
/*saj*/
#ifdef TCL_MODULE
  Tcl_Mutex *alloc;
  alloc = Tcl_GetAllocMutex();
  Tcl_MutexLock(alloc);
#endif
	if (ptr)
		free(ptr);
/*saj*/
#ifdef TCL_MODULE
  Tcl_MutexUnlock(alloc);
#endif
}

#endif
