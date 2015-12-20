#ifndef _sf_alloc_h
#define _sf_alloc_h


#include <stdio.h>
#include <stdlib.h>


/*------------------------------------------------------------*/
void *sf_alloc (int n,      /* number of elements */
                size_t size /* size of one element */);
/*< output-checking allocation >*/

/*------------------------------------------------------------*/
double *sf_doublealloc (int n /* number of elements */);
/*< double allocation >*/

/*------------------------------------------------------------*/
double **sf_doublealloc2(int n1, /* fast dimension */
                         int n2  /* slow dimension */);
/*< float 2-D allocation, out[0] points to a contiguous array >*/

#endif
