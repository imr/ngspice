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
#include "spmatrix.h"
#include "spdefs.h"





/*
 *  Function declarations
 */

#ifdef __STDC__
#if spSEPARATED_COMPLEX_VECTORS
#else
#endif
#else /* __STDC__ */
#endif /* __STDC__ */

void
spConstMult(matrix, constant)
	MatrixPtr	matrix;
	double		constant;
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

#ifdef notdef

int	spccc = 0;
int	spccc_hold = -1;
int	spccc_h1 = 1;
int	spccc_h2 = 1;
int	spccc_h3 = 1;
int	spccc_h4 = 1;
int	spccc_h5 = 1;
int	spccc_h11 = 1;
int	spccc_h12 = 1;
int	spccc_h13 = 1;
int	spccc_h15 = 1;
int	spccc_h99 = 1;

spCheck(matrix, key)
	MatrixPtr	matrix;
	int		key;
{
	ElementPtr	e;
	int		i, n, k;
	int		size = matrix->Size;

	spccc += 1;
	if (spccc == spccc_hold)
		hold_matrix99( );

	for (i = 1; i <= size; i++) {
		k = -1;
		if (!(key & 2)) {
			for (n = 0, e = matrix->FirstInCol[i]; e && n <= size;
				e = e->NextInCol)
			{
				if (k >= e->Row)
					hold_matrix2( );
				if (e->Col != i)
					hold_matrix3( );
				if (e->NextInRow && e->Col >= e->NextInRow->Col)
					hold_matrix5( );

				k = e->Row;
				n += 1;
			}
			if (n > size)
				hold_matrix1( );
		}

		k = -1;
		if (matrix->RowsLinked && !(key & 1)) {
			for (n = 0, e = matrix->FirstInRow[i]; e && n <= size;
				e = e->NextInRow)
			{
				if (k >= e->Col)
					hold_matrix12( );
				if (e->Row != i)
					hold_matrix13( );
				if (e->NextInCol && e->Row >= e->NextInCol->Row)
					hold_matrix15( );

				k = e->Col;
				n += 1;
			}
			if (n > size)
				hold_matrix11( );
		}
	}

}

hold_matrix1( )
{
	if (spccc_h1) {
		printf("BAD MATRIX");
		fflush(stdout);
	}
}

hold_matrix2( )
{
	if (spccc_h2) {
		printf("BAD MATRIX 2");
		fflush(stdout);
	}
}

hold_matrix3( )
{
	if (spccc_h3) {
		printf("BAD MATRIX 3");
		fflush(stdout);
	}
}

hold_matrix4( )
{
	if (spccc_h4) {
		printf("BAD MATRIX 3");
		fflush(stdout);
	}
}

hold_matrix5( )
{
	if (spccc_h5) {
		printf("BAD MATRIX 5");
		fflush(stdout);
	}
}

hold_matrix11( )
{
	if (spccc_h11) {
		printf("BAD MATRIX 11");
		fflush(stdout);
	}
}

hold_matrix12( )
{
	if (spccc_h12) {
		printf("BAD MATRIX 12");
		fflush(stdout);
	}
}

hold_matrix13( )
{
	if (spccc_h13) {
		printf("BAD MATRIX 13");
		fflush(stdout);
	}
}

hold_matrix15( )
{
	if (spccc_h15) {
		printf("BAD MATRIX 15");
		fflush(stdout);
	}
}

hold_matrix99( )
{
	if (spccc_h99) {
		printf("BAD MATRIX 99");
		fflush(stdout);
	}
}

#endif
