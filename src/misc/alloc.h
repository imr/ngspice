/*************
 * Header file for alloc.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_ALLOC_H
#define ngspice_ALLOC_H

#ifndef HAVE_LIBGC
void * tmalloc(size_t num);
void * trealloc(const void *ptr, size_t num);
void txfree(const void *ptr);
#endif

#endif
