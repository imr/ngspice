#ifndef _MEMORY_H
#define _MEMORY_H

#ifndef HAVE_LIBGC
extern void *tmalloc(size_t num);
extern void *trealloc(void *str, size_t num);
extern void txfree(void *ptr);

#define tfree(x) (txfree(x), x = 0)

#else
#include <gc/gc.h>

#define tmalloc(m) GC_malloc(m)
#define trealloc(m,n) GC_realloc((m),(n))
#define tfree(m)
#define txfree(m)
#endif

#include "../misc/stringutil.h" /* va: spice3 internally bzero */

#define alloc(TYPE) ((TYPE *) tmalloc(sizeof(TYPE)))
#define MALLOC(x) tmalloc((unsigned)(x))
#define FREE(x) {if (x) {txfree((char *)(x));(x) = 0;}}
#define REALLOC(x,y) trealloc((char *)(x),(unsigned)(y))
#define ZERO(PTR,TYPE)	(bzero((PTR),sizeof(TYPE)))

#ifdef CIDER


#define RALLOC(ptr,type,number)\
if ((number) && (!(ptr = (type *)calloc((number), (unsigned)(sizeof(type)))))) {\
  return(E_NOMEM);\
}

#define XALLOC(ptr,type,number)   \
if ((number) && (!(ptr = (type *)calloc((number), (unsigned)(sizeof(type)))))) {\
  SPfrontEnd->IFerror( E_PANIC, "Out of Memory", NIL(IFuid) );\
  exit( 1 );\
}

#define XCALLOC(ptr,type,number)   \
if ((number) && (!(ptr = (type *)calloc((number), (unsigned)(sizeof(type)))))) {\
  fprintf( stderr, "Out of Memory\n" );\
  exit( 1 );\
}
#endif /* CIDER */

#endif
