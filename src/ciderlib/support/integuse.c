/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numenum.h"
#include "ngspice/gendev.h"
#include "ngspice/cidersupt.h"

/* function to compute the integrated variables discretization */

#define ccap qcap+1

double
integrate(double **devStates, TranInfo *info, int qcap )
{
    double value;
    double *coeff = info->intCoeff;

    switch ( info->method ) {
    case BDF:
	switch( info->order ) {
	case 1:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap];
	    break;
	case 2:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap];
	    break;
	case 3:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap];
	    break;
	case 4:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap];
	    break;
	case 5:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap] +
		    coeff[5] * devStates[5][qcap];
	    break;
	case 6:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap] +
		    coeff[5] * devStates[5][qcap] +
		    coeff[6] * devStates[6][qcap];
	    break;
	default:
	    printf( "\n integration order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    case TRAPEZOIDAL:
    default:
	switch( info->order ) {
	case 1:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap];
	    devStates[0][ccap] = value;
	    break;
	case 2:
	    value = coeff[0] * devStates[0][qcap] +
		    coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[1][ccap];
	    devStates[0][ccap] = value;
            break;
        default:
	    printf( "\n integration order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    }

    return( value );
}

/* function to predict the value of the variables */

double
predict(double **devStates, TranInfo *info, int qcap )
{
    double value;
    double *coeff = info->predCoeff;

    switch ( info->method ) {
    case BDF:	
	switch( info->order ) {
	case 1:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap];
	    break;
	case 2:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap];
	    break;
	case 3:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap] +
		    coeff[3] * devStates[4][qcap];
	    break;
	case 4:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap] +
		    coeff[3] * devStates[4][qcap] +
		    coeff[4] * devStates[5][qcap];
	    break;
	case 5:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap] +
		    coeff[3] * devStates[4][qcap] +
		    coeff[4] * devStates[5][qcap] +
		    coeff[5] * devStates[6][qcap];
	    break;
	case 6:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap] +
		    coeff[3] * devStates[4][qcap] +
		    coeff[4] * devStates[5][qcap] +
		    coeff[5] * devStates[6][qcap] +
		    coeff[6] * devStates[7][qcap];
	    break;
	default:
	    printf( "\n prediction order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    case TRAPEZOIDAL:	
    default:
	switch( info->order ) {
	case 1:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap];
	    break;
	case 2:
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[3][qcap];
	    /*
	    value = coeff[0] * devStates[1][qcap] +
		    coeff[1] * devStates[2][qcap] +
		    coeff[2] * devStates[1][ccap];
	    */
	    break;
	default:
	    printf( "\n prediction order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    }
	
    return( value );
}

double
computeLTECoeff( TranInfo *info )
{
    double *delta = info->delta;
    double temp, denom, lteCoeff;

    switch ( info->method ) {
    case BDF:
	switch( info->order ) {
	    case 1:
		denom = delta[ 0 ] + delta[ 1 ];
		break;
	    case 2:
		denom = delta[ 0 ] + delta[ 1 ] + delta[ 2 ];
		break;
	    case 3:
		denom = delta[ 0 ] + delta[ 1 ] + delta[ 2 ] +
			delta[ 3 ];
		break;
	    case 4:
		denom = delta[ 0 ] + delta[ 1 ] + delta[ 2 ] +
			delta[ 3 ] + delta[ 4 ];
		break;
	    case 5:
		denom = delta[ 0 ] + delta[ 1 ] + delta[ 2 ] +
			delta[ 3 ] + delta[ 4 ] + delta[ 5 ];
		break;
	    case 6:
		denom = delta[ 0 ] + delta[ 1 ] + delta[ 2 ] +
			delta[ 3 ] + delta[ 4 ] + delta[ 5 ] + delta[ 6 ];
		break;
	    default:
		printf( "\n integration order %d !! STOP \n", info->order );
		exit( 0 );
		break;
	}
	break;
    case TRAPEZOIDAL:
    default:
	switch( info->order ) {
	    case 1:
		denom = delta[ 0 ] + delta[ 1 ];
		break;
	    case 2:
		/*
		denom = delta[ 0 ] + delta[ 1 ];
		*/
		temp = delta[ 0 ] + delta [ 1 ];
		denom = 2.0 * temp * (temp + delta[ 2 ]) / delta[ 0 ];
		break;
	    default:
		printf( "\n integration order %d !! STOP \n", info->order );
		exit( 0 );
		break;
	}
	break;
    }
    lteCoeff = delta[ 0 ] / denom;
    return( lteCoeff );
}

/* function to integrate a linear DAE */
double
integrateLin(double **devStates, TranInfo *info, int qcap )
{
    double value;
    double *coeff = info->intCoeff;

    switch ( info->method ) {
    case BDF:
	switch( info->order ) {
	case 1:
	    value = coeff[1] * devStates[1][qcap];
	    break;
	case 2:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap];
	    break;
	case 3:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap];
	    break;
	case 4:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap];
	    break;
	case 5:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap] +
		    coeff[5] * devStates[5][qcap];
	    break;
	case 6:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[2][qcap] +
		    coeff[3] * devStates[3][qcap] +
		    coeff[4] * devStates[4][qcap] +
		    coeff[5] * devStates[5][qcap] +
		    coeff[6] * devStates[6][qcap];
	    break;
	default:
	    printf( "\n integration order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    case TRAPEZOIDAL:
    default:
	switch( info->order ) {
	case 1:
	    value = coeff[1] * devStates[1][qcap];
	    break;
	case 2:
	    value = coeff[1] * devStates[1][qcap] +
		    coeff[2] * devStates[1][ccap];
            break;
        default:
	    printf( "\n integration order %d !! STOP \n", info->order );
	    exit( 0 );
	}
	break;
    }

    return( value );
}

