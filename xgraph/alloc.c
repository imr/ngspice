/* $Header$ */
/*
 *
 * alloc.c : Memory checked Malloc. This malloc keeps track of memory usage.
 *
 * Routines:
 *	char * 		Malloc();
 *	char *		Calloc();
 *	char *		Realloc();
 *	void 		Free();
 *	unsigned	MemStat();
 *	unsigned	MemPtr();
 *	void		MemChain();
 *
 * $Log$
 * Revision 1.3  2011-05-08 08:54:21  rlar
 * rename macros public and private  -->  PUBLIC and PRIVATE
 *
 * Revision 1.2  2011/04/27 18:30:17  rlar
 * code cleanup
 *
 * Revision 1.1  2004/01/25 09:00:49  pnenzi
 *
 * Added xgraph plotting program.
 *
 * Revision 1.1.1.1  1999/12/03 23:15:53  heideman
 * xgraph-12.0
 *
 * Revision 1.10  1991/02/01  08:12:55  christos
 * Overhaul... Simplified and added calloc.
 *
 * Revision 1.9  1990/10/02  18:11:24  christos
 * Another Realloc() bug!
 *
 * Revision 1.8  90/10/02  17:32:45  christos
 * Fixed Realloc() bug.
 *
 * Revision 1.7  90/08/24  02:28:15  christos
 * Changed bigstruct_t to align_t
 * for lint.
 *
 * Revision 1.6  90/07/15  17:31:33  christos
 * Fixed MemPtr Bug
 *
 * Revision 1.5  90/07/11  16:19:31  christos
 * Added Realloc()
 *
 * Revision 1.4  90/03/21  12:58:44  christos
 * Fixed void buggy computations.
 *
 * Revision 1.3  90/02/26  02:15:11  christos
 * ANSI conformance.
 *
 * Revision 1.2  89/08/29  14:08:25  christos
 * Fixed.
 *
 * Revision 1.1  89/03/27  14:23:40  christos
 * Initial revision
 *
 */
#ifndef lint
static char rcsid[] = "$Id$";

#endif				/* lint */
#include <stdio.h>
#ifdef __STDC__
#include <stdlib.h>
#include <memory.h>
#else
extern char *malloc();
extern char *calloc();
extern char *realloc();
extern void free();
extern void abort();
extern char *memset();

#endif

#ifndef NIL
#define NIL(a) ((a *) 0)
#endif				/* NIL */

#ifndef MIN
#define MIN(a, b) 		((a) > (b) ? (b) : (a))
#endif				/* MIN */

#ifndef MAX
#define MAX(a, b) 		((a) < (b) ? (b) : (a))
#endif				/* MAX */

#ifndef PRIVATE
#define PRIVATE static
#endif

#ifndef PUBLIC
#define PUBLIC
#endif


#define SIG_GOOD	0x01020304
#define SIG_FREE	0x04030201
#define OVERHEAD	(sizeof(long) + sizeof(unsigned))

PRIVATE unsigned memused = 0;
PRIVATE unsigned memalloc = 0;

#ifdef __STDC__
typedef void *Ptr;

#else
typedef char *Ptr;

#endif

/* _chaina():
 *	Check things for validity and allocate space
 */
PRIVATE Ptr
_chaina(n, routine, action, tptr)
unsigned n;

Ptr(*routine) ();
char   *action;
Ptr     tptr;
{
    char   *ptr;

    if (n == 0) {
	(void) fprintf(stderr, "*** %s zero length block.\n",
		       action);
	if (tptr != (Ptr) 0) {
	    ptr = tptr;
	    *((long *) ptr) = SIG_GOOD;
	    memused += *((unsigned *) &ptr[sizeof(long)]);
	    memalloc++;
	}
	abort();
    }

    ptr = (tptr == (Ptr) 0) ? (char *) routine (n + OVERHEAD) :
	(char *) routine (tptr, n + OVERHEAD);

    if (ptr == NIL(char)) {
	if (tptr != (Ptr) 0)
	    *((long *) tptr) = SIG_GOOD;
	(void) fprintf(stderr,
		       "*** Out of memory in %s (current allocation %d).\n",
		       action, memused, n);

	abort();
    }
    *((long *) ptr) = SIG_GOOD;
    memused += (*((unsigned *) &ptr[sizeof(long)]) = n);
    memalloc++;
    ptr += OVERHEAD;
    return ((Ptr) ptr);
}				/* end _chaina */


/* _chainc():
 *	Check the pointer given
 */
PRIVATE unsigned
_chainc(ptr, action)
char  **ptr;
char   *action;
{
    static char *msg = "*** %s %s pointer.\n";

    if (*ptr == NIL(char)) {
	(void) fprintf(stderr, msg, action, "nil");
	abort();
    }
    *ptr -= OVERHEAD;
    switch (*((long *) *ptr)) {
    case SIG_GOOD:
	return (*((unsigned *) &((*ptr)[sizeof(long)])));
    case SIG_FREE:
	(void) fprintf(stderr, msg, action, "free");
	abort();
    default:
	(void) fprintf(stderr, msg, action, "invalid");
	abort();
    }
    return (0);
}				/* end _chainc */


/* Malloc():
 *	real alloc
 */
PUBLIC  Ptr
Malloc(n)
unsigned n;
{
    static char *routine = "malloc";

    return (_chaina(n, malloc, routine, (Ptr) 0));
}				/* end Malloc */


/* Calloc():
 *	real alloc
 */
PUBLIC  Ptr
Calloc(n, sz)
unsigned n,
        sz;
{
    Ptr     ptr;
    static char *routine = "calloc";

    n *= sz;
    ptr = _chaina(n, malloc, routine, (Ptr) 0);
    memset((char *) ptr, 0, n);
    return (ptr);
}				/* end Calloc */


/* Realloc():
 *	real alloc
 */
PUBLIC  Ptr
Realloc(ptr, n)
Ptr     ptr;
unsigned n;
{
    static char *routine = "realloc";

    memused -= _chainc((char **) &ptr, routine);
    memalloc--;
    *((long *) ptr) = SIG_FREE;
    return (_chaina(n, realloc, routine, ptr));
}				/* end Realloc */


/* Free():
 *	free memory counting the number of bytes freed
 */
PUBLIC void
Free(ptr)
Ptr     ptr;
{
    static char *routine = "free";

    memused -= _chainc((char **) &ptr, routine);
    memalloc--;
    *((long *) ptr) = SIG_FREE;
    free(ptr);
}				/* end Free */


/* MemChain():
 *	Dump the chain
 */
PUBLIC void
MemChain()
{
    if (memused == 0 && memalloc == 0)
	(void) fprintf(stdout, "\tNo memory allocated.\n");
    else {
	(void) fprintf(stdout, "\t%u Bytes allocated in %u chunks.\n", memused,
		       memalloc);
	(void) fprintf(stdout, "\tAverage chunk length %u bytes.\n",
		       memused / memalloc);
    }
}				/* end MemChain */


/* MemStat():
 *	return the amount of memory in use
 */
PUBLIC unsigned
MemStat()
{
    return (memused);
}				/* end MemStat */


/* MemPtr():
 *	return the amount of memory used by the pointer
 */
PUBLIC unsigned
MemPtr(ptr)
Ptr     ptr;
{
    static char *routine = "get size";

    return (_chainc((char **) &ptr, routine));
}				/* end MemPtr */
