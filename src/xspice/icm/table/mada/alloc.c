/* Convenience allocation programs. */
/*
  Copyright (C) 2004 University of Texas at Austin
  Copyright (C) 2007 Colorado School of Mines

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdlib.h>
#include <sys/types.h>

#include "alloc.h"
#include "ngspice/cm.h"


/*------------------------------------------------------------*/
void *
sf_alloc(int n       /* number of elements */,
         size_t size /* size of one element */)
/*< output-checking allocation >*/
{
    void *ptr;

    size *= (size_t) n;

    if (0 >= size)
        cm_message_printf("%s: illegal allocation(%d bytes)", __FILE__, (int) size);

    ptr = malloc(size);

    if (NULL == ptr)
        cm_message_printf("%s: cannot allocate %d bytes : ", __FILE__, (int) size);

    return ptr;
}

/*------------------------------------------------------------*/
double *
sf_doublealloc(int n /* number of elements */)
/*< float allocation >*/
{
    return (double*) sf_alloc(n, sizeof(double));
}

/*------------------------------------------------------------*/
double **
sf_doublealloc2(int n1 /* fast dimension */,
                int n2 /* slow dimension */)
/*< float 2-D allocation, out[0] points to a contiguous array >*/
{
    int i2;
    double **ptr = (double**) sf_alloc(n2, sizeof(double*));

    ptr[0] = sf_doublealloc(n1 * n2);
    for (i2 = 1; i2 < n2; i2++)
        ptr[i2] = ptr[0] + i2 * n1;

    return ptr;
}
