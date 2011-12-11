/*
 *  MATRIX UTILITY MODULE
 *
 *  This file contains new routines for Spice3f
 *
 *  >>> User accessible functions contained in this file:
 *  spConstMul
 *
 *  >>> Other functions contained in this file:
 */


/*
 *  IMPORTS
 *
 *  >>> Import descriptions:
 *  spConfig.h
 *      Macros that customize the sparse matrix routines.
 *  spMatrix.h
 *      Macros and declarations to be imported by the user.
 *  spDefs.h
 *      Matrix type and macro definitions for the sparse matrix routines.
 */

#define spINSIDE_SPARSE
#include "spconfig.h"
#include "ngspice/spmatrix.h"
#include "spdefs.h"


void
spConstMult(MatrixPtr matrix, double constant)
{
	ElementPtr	e;
	int		i;
	int		size = matrix->Size;

	for (i = 1; i <= size; i++) {
		for (e = matrix->FirstInCol[i]; e; e = e->NextInCol) {
			e->Real *= constant;
			e->Imag *= constant;
		}
	}

}
