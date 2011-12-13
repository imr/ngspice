/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numenum.h"
#include "ngspice/cidersupt.h"

/* compute the coefficient for the integration and predictor methods */
/* based on the Lagrange polynomial method in Liniger et. al. */

void
computeIntegCoeff(int method, int order, double *intCoeff, double *delta)
{
    int i, j, k;
    double sum, temp, preMult;
    double num, denom, prod;

    switch( method ) {
    case BDF:
	/* determine coeff[0] first */

	sum = 0.0;
	temp = 0.0;
	for( j = 0 ; j < order; j++ ) {
	    temp += delta[ j ];
	    sum += 1.0 / temp;
	}
	intCoeff[ 0 ] = sum;

	/* now compute the higher order coefficients */
	for( j = 1; j <= order ; j++ ) {
	    /* compute the pre multiplier */
	    temp = 0.0;
	    for( i = 0; i < j; i++ ) {
		temp += delta[ i ];
	    }
	    preMult = 1.0 / temp;
	    prod = 1.0;
	    /* now compute the product */
	    for( i = 1; i <= order; i++ ) {
		/* product loop */
		if( i != j ) {
		    num = 0.0;
		    for( k = 0; k < i; k++ ) {
			/* first the numerator */
			num += delta[ k ];
		    }
		    if( i > j ) {
			/* denom is positive */
			denom = 0.0;
			for( k = j; k < i; k++ ) {
			    denom += delta[ k ];
			}
		    }
		    else {
			/* i < j */
			denom = 0.0;
			for( k = i; k < j; k++ ) {
			    denom += delta[ k ];
			}
			denom = -denom;
		    }
		    prod *= num / denom ;
		}
	    }
	    intCoeff[ j ] = -preMult * prod;
	}
	break;
    case TRAPEZOIDAL:
    default:
	switch( order ) {
	case 1:
	    temp = 1.0 / delta[ 0 ];
	    intCoeff[ 0 ] = temp;
	    intCoeff[ 1 ] = -temp;
	    break;
	case 2:
	    temp = 2.0 / delta[ 0 ];
	    intCoeff[ 0 ] = temp;
	    intCoeff[ 1 ] = -temp;
	    intCoeff[ 2 ] = -1.0;
	    break;
	}
	break;
    }
}
		    


void
computePredCoeff(int method, int order, double *predCoeff, double *delta)
{
    int i, j, k;
    double num, denom, prod;

    if( method == TRAPEZOIDAL && order > 2 ) {
	printf("\n computePredCoeff: order > 2 for trapezoidal");
	exit( -1 );
    }
    for( j = 1; j <= order+1 ; j++ ) {
	prod = 1.0;
	/* now compute the product */
	for( i = 1; i <= order+1; i++ ) {
	    /* product loop */
	    if( i != j ) {
		num = 0.0;
		for( k = 0; k < i; k++ ) {
		    /* first the numerator */
		    num += delta[ k ];
		}
		if( i > j ) {
		    /* denom is positive */
		    denom = 0.0;
		    for( k = j; k < i; k++ ) {
			denom += delta[ k ];
		    }
		}
		else {
		    /* i < j */
		    denom = 0.0;
		    for( k = i; k < j; k++ ) {
			denom += delta[ k ];
		    }
		    denom = -denom;
		}
		prod *= num / denom ;
	    }
	}
	predCoeff[ j - 1 ] = prod;
    }
}



/* main program to check the coefficients
main()
{
    double intCoeff[ 7 ], predCoeff[ 7 ];
    double delta[ 7 ];
    int order = 1;
    int i;

    for( i = 0; i <= 6; i++ ) {
	delta[ i ] = 1.0;
    }

    computeIntegCoeff(TRAPEZOIDAL, order, intCoeff, delta );
    computePredCoeff(TRAPEZOIDAL, order, predCoeff, delta );

    for(i = 0; i <= order; i++ ) {
	printf("\n IntCoeff[ %d ] = %e  PredCoeff[ %d ] = %e ", i, intCoeff[ i ], i, predCoeff[ i ] );
    }
}
*/
