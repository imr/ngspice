/*************
 * Header file for alloc.c
 * 1999 E. Rouat
 ************/

#ifndef ALLOC_H_INCLUDED
#define ALLOC_H_INCLUDED

#ifndef HAVE_LIBGC
void * tmalloc(size_t num);
void * trealloc(void *ptr, size_t num);
void txfree(void *ptr);
#endif

#endif
