#ifndef ngspice_MEMORY_H
#define ngspice_MEMORY_H

#include <stddef.h>

#include "ngspice/alloc.h"

/* "Type" allocations */
#define TMALLOC(t, n)       (t *) tmalloc(sizeof(t) * (size_t)(n))
#define TREALLOC(t, p, n)   (t *) trealloc(p, sizeof(t) * (size_t)(n))


#ifdef HAVE_LIBGC
#define tfree(m)
#else
#define tfree(x) (txfree(x), (x) = 0)
#endif /* HAVE_LIBGC */


#include "ngspice/stringutil.h"

#define FREE(x)          do { if (x) { txfree(x); (x) = NULL; } } while(0)
#define ZERO(PTR, TYPE)  memset(PTR, 0, sizeof(TYPE))


#ifdef CIDER

#define RALLOC(ptr, type, number)                                       \
    do {                                                                \
        if ((number) && (ptr = (type *)calloc((size_t)(number), sizeof(type))) == NULL) \
            return E_NOMEM;                                             \
    } while(0)

#define XALLOC(ptr, type, number)                                       \
    do {                                                                \
        if ((number) && (ptr = (type *)calloc((size_t)(number), sizeof(type))) == NULL) { \
            SPfrontEnd->IFerrorf(E_PANIC, "Out of Memory");             \
            controlled_exit(1);                                         \
        }                                                               \
    } while(0)

#define XCALLOC(ptr, type, number)                                      \
    do {                                                                \
        if ((number) && (ptr = (type *)calloc((size_t)(number), sizeof(type))) == NULL) { \
            fprintf(stderr, "Out of Memory\n");                         \
            controlled_exit(1);                                         \
        }                                                               \
    } while(0)

#endif /* CIDER */

#endif /* include guard */
