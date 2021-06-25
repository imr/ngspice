/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Miscellaneous routines culled from the oned directory so that the twod
 * code will run without having to compile the oned code
 */

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/spmatrix.h"
#include "ngspice/cidersupt.h"
#include "ngspice/cpextern.h"


/* Used in Solution Projection Calculations */
double guessNewConc(double conc, double delta)
{
    BOOLEAN acceptable = FALSE;
    double fib, newConc, lambda, fibn, fibp;
    lambda = 1.0;
    fibn = 1.0;
    fibp = 1.0;
    newConc = 0.0;
    for ( ; !acceptable ; ) {
        fib = fibp;
        fibp = fibn;
        fibn += fib;
        lambda *= fibp / fibn;
        newConc = conc + delta * lambda;
        if( newConc > 0.0 ) {
            acceptable = TRUE;
        }
        else {
            /* newConc is still negative but fibp and fibn are large */
            if ( (fibp > 1e6) || (fibn > 1e6) ) {
                acceptable = TRUE;
                newConc = conc;
            }
        }
    }
    return( newConc );
}

/* Used in Doping Calculation */
/* compute the concentration at x given an array of data.
 * The data is stored in a two-dimensional array x, N(x).
 * x is assumed to be in ascending order. given an x
 * a search is performed to determine the data points closest
 * to it and linear interpolation is performed to generate N(x)
 */

/*
#define LOGSUPREM
*/
double lookup(double **dataTable, double x)
{
    double conc=0.0, x0, x1, y0, y1;
#ifdef LOGSUPREM
    double lnconc, lny0, lny1;
#endif
    int index, numPoints;
    BOOLEAN done = FALSE;

    numPoints = (int)dataTable[ 0 ][ 0 ];
    for( index = 2; index <= numPoints && (!done); index++ ) {
	x1 = dataTable[ 0 ][ index ];
	/* check if x1 > x */
	if( x1 >= x ) {
	    /* found an x1 larger than x, so linear interpolate */
	    x0 = dataTable[ 0 ][ index - 1 ];
	    y0 = dataTable[ 1 ][ index - 1 ];
	    y1 = dataTable[ 1 ][ index ];
#ifdef LOGSUPREM
	    /* Ignore concentrations below 1.0 */
	    if ( ABS(y0) < 1.0 )
		lny0 = 0.0;
	    else
	        lny0 = SGN(y0) * log( ABS(y0) );
	    if ( ABS(y1) < 1.0 )
		lny1 = 0.0;
	    else
	        lny1 = SGN(y0) * log( ABS(y0) );
	    lnconc = lny0 + (lny1 - lny0) * (x - x0) / (x1 - x0);
	    conc = SGN(lnconc) * exp( ABS(lnconc) );
#else
	    conc = y0 + (y1 - y0) * (x - x0) / (x1 - x0);
#endif /* LOGSUPREM */
	    done = TRUE;
	} else {
	    if( index == numPoints ) {
		/* set to concentration of last node - due to roundoff errors */
		conc = dataTable[ 1 ][ numPoints ];
	    }
	}
    }
    return ( conc );
}

/* Used in admittance calculations */
/* this function returns TRUE is SOR iteration converges otherwise FALSE */

BOOLEAN hasSORConverged(double *oldSolution, double *newSolution,
                        int numEqns)
{
    BOOLEAN converged = TRUE;
    int index;
    double xOld, xNew, tol;
    double absTol = 1e-12;
    double relTol = 1e-3;
    for( index = 1 ; index <= numEqns ; index++ ) {
	xOld = oldSolution[ index ];
	xNew = newSolution[ index ];
	tol = absTol + relTol * MAX( ABS( xOld ), ABS( xNew ));
	if( ABS( xOld - xNew ) > tol ) {
	    converged = FALSE;
	    printf("hasSORconverged failed\n");
	    break;
	}
    }
    return( converged ); 
}

/* Used to Check Sparse Matrix Errors */
BOOLEAN
foundError(int error)
{
    BOOLEAN matrixError;

    switch( error ) {
	/*  Removed for Spice3e1 Compatibility 
	case spSMALL_PIVOT:
	    printf( "Warning: LU Decomposition Problem - SMALL PIVOT\n" );
	    matrixError = FALSE;
	    break;
	*/
	case spPANIC:
	    printf( "Error: LU Decomposition Failed - PANIC\n" );
	    matrixError = TRUE;
	    break;
	case spSINGULAR:
	    printf( "Error: LU Decomposition Failed - SINGULAR\n" );
	    matrixError = TRUE;
	    break;
	/*  Removed for Spice3e1 Compatibility 
	case spZERO_DIAG:
	    printf( "Error: LU Decomposition Failed - ZERO PIVOT\n" );
	    matrixError = TRUE;
	    break;
	*/
	case spNO_MEMORY:
	    printf( "Error: LU Decomposition Failed - NO MEMORY\n" );
	    matrixError = TRUE;
	    break;
	default:
	    matrixError = FALSE;
	    break;
    }
    return( matrixError );
}

/* Return TRUE if the filetype variable matches the string 's' */
BOOLEAN compareFiletypeVar(char *s)
{
	char buf[BSIZE_SP];

	if (cp_getvar("filetype", CP_STRING, buf, sizeof(buf))) {
		if (!strcmp(buf, s)) {
			return TRUE;
		} else {
			return FALSE;
		}
	} else {
		return FALSE;
	} 
}
