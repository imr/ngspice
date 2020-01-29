/*************
 * Header file for alloc.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_ALLOC_H
#define ngspice_ALLOC_H

#include <stddef.h>

void *tmalloc(size_t num);
void *tcalloc(size_t num, size_t size);
void *trealloc(void *ptr, size_t num);
void txfree(void *ptr);

#ifdef HAVE_LIBGC
#include <gc/gc.h>

#define tmalloc_raw     GC_malloc
#define tcalloc_raw     GC_calloc
#define trealloc_raw    GC_realloc

#else
void *tmalloc_raw(size_t num);
void *tcalloc_raw(size_t num, size_t size);
void *trealloc_raw(void *ptr, size_t num);
#endif

#endif /* include guard */
