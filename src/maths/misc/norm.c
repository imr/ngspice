/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "norm.h"

/* functions to compute max and one norms of a given vector of doubles */

double 
maxNorm(double *vector, int size)
{
    double norm = 0.0;
    double candidate;
    int index;

    for( index = 1; index <= size; index++ ) {
	candidate = fabs(vector[ index ]);
	if( candidate > norm ) {
	    norm = candidate;
	}
    }

    return( norm );
}


double 
oneNorm(double *vector, int size)
{
    double norm = 0.0;
    double value;
    int index;

    for( index = 1; index <= size; index++ ) {
	value = vector[ index ];
	if( value < 0.0 )
	    norm -= value;
	else
	    norm += value;
    }
    return( norm );
}

double 
l2Norm(double *vector, int size)
{
    double norm = 0.0;
    double value; 
    int index;

    for( index = 1; index <= size; index++ ) {
	value = vector[ index ];
	norm += value * value;
    }
    norm = sqrt( norm );
    return( norm );
}

/* 
 * dot():
 *   computes dot product of two vectors
 */
double 
dot(double *vector1, double *vector2, int size)
{
    register double dot = 0.0;
    register int index;

    for( index = 1; index <= size; index++ ) {
	dot += vector1[ index ] * vector2[ index ];
    }
    return( dot );
}
