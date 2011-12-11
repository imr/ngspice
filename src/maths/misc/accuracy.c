/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "accuracy.h"


/* XXX Globals are hiding here.
 * All globals have been moved to main.c were all true globals 
 * should reside. May be in the future that the machine dependent
 * global symbols will be gathered into a structure.
 *
 * Paolo Nenzi 2002 
 */

/*
 * void
 * evalAccLimits(void)
 *
 * This function computes the floating point accuracy on your machine.
 * This part of the code is critical and will affect the result of
 * CIDER simulator. CIDER uses directly some constants calculated here
 * while SPICE does not (yet), but remember that spice runs on the same 
 * machine that runs CIDER and uses the same floating point implementation.
 *
 * A note to x86 Linux users:
 *
 * Intel processors (from i386 up to the latest Pentiums) do FP calculations 
 * in two ways:
 *               53-bit mantissa mode (64 bits overall)
 *               64-bit mantissa mode (80 bits overall)
 *
 * The 53-bit mantissa mode conforms to the IEEE 754 standard (double 
 * precision), the other (64-bit mantissa mode) does not conform to 
 * IEEE 754, butlet programmers to use the higher precision "long double" 
 * data type.
 * 
 * Now the problem: the x86 FPU can be in on mode only, therefore is
 * not possible to have IEEE conformant double and "long double" data
 * type at the same time. You have to choose which one you prefer.  
 *
 * Linux developers decided that was better to give "long double" to
 * programmers that to provide a standard "double" implementation and,
 * by default, Linux set the FPU in 64 bit mantissa mode. FreeBSD, on 
 * the other side, adopt the opposite solution : the FPU is in 53 bit 
 * mantissa mode.
 *
 * Since no one but the programmers really knows what a program requires,
 * the final decision on the FPU mode of operation is left to her/him.
 * It is possible to alter the FPU mode of operation using instruction
 * produced by the operating system.
 *
 * Thanks to AMAKAWA Shuhei for the information on x86 FPU Linux and
 * freeBSD on which the above text was derived.
 * Paolo Nenzi 2002.
 */ 

void
evalAccLimits(void)
{
    double acc = 1.0;
    double xl = 0.0;
    double xu = 1.0;
    double xh, x1, x2, expLim;
    double muLim, temp1, temp2, temp3, temp4;
    
    double xhold, dif;  /* Introduced to avoid numerical trap if
                      using non IEEE754 FPU */

/* First we compute accuracy */ 
 
    for( ; (acc + 1.0) > 1.0 ; ) {
	acc *= 0.5;	
    }
    acc *= 2.0;
    Accuracy = acc;

/*  
 * This loop has been modified to include a variable to track 
 * xh change. If it does not change, we exit the loop. This is
 * an ugly countermeasure to avoid infinite cycling when in
 * x86 64-bit mantissa mode.
 *
 * Paolo Nenzi 2002
 */

    xh = 0.5 * (xl + xu);
    for( ; (xu-xl > (2.0 * acc * (xu + xl))); ) {
	x1 = 1.0 / ( 1.0 + (0.5 * xh) );
	x2 = xh / ( exp(xh) - 1.0 );
	if( (x1 - x2) <= (acc * (x1 + x2))) {
	    xl = xh;
	    xhold = xh;
	} else {
	    xu = xh;
	    xhold = xh;
	}
	xh = 0.5 * (xl + xu);
	
	dif = fabs(xhold - xh);
	if (dif <= DBL_EPSILON) break;
    }
    BMin = xh;
    BMax = -log( acc );


/* 
 * This loop calculate the maximum exponent x for
 * which the function exp(-x) returns a value greater
 * than 0. The result is used to prevent underflow 
 * on large negative arguments to exponential.
 * AFAIK: used only in Bernoulli function.
 */

    expLim = 80.0;
    for( ; exp( -expLim ) > 0.0; ) {
	expLim += 1.0;
    }
    expLim -= 1.0;
    ExpLim = expLim;

/*
 * What this loop does ???
 */

    muLim = 1.0;
    temp4 = 0.0;
    for( ; 1.0 - temp4 > acc; ){
        muLim *= 0.5;
	temp1 = muLim;
	temp2 = pow( temp1 , 0.333 ) ;
	temp3 = 1.0 / (1.0 + temp1 * temp2 );
	temp4 = pow( temp3 , 0.37/1.333 );
    }
    muLim *= 2.0;
    MuLim = muLim;

    muLim = 1.0;
    temp3 = 0.0;
    for( ; 1.0 - temp3 > acc; ){
        muLim *= 0.5;
	temp1 = muLim;
	temp2 = 1.0 / (1.0 + temp1 * temp1 );
	temp3 = sqrt( temp2 );
    }
    muLim *= 2.0;
    MutLim = muLim;

}

/*
 * Other misterious code.
 * Berkeley's people love to leave spare code for info archeologists.
 * 
main ()
{
    double x;
    double bx, dbx, bMx, dbMx;

    evalAccLimits();
    for( x = 0.0; x <= 100.0 ; x += 1.0 ) {
	bernoulliNew( x, &bx, &dbx, &bMx, &dbMx, 1);
	printf( "\nbernoulliNew: x = %e bx = %e dbx = %e bMx = %e dbMx = %e ", x, bx, dbx, bMx, dbMx );
	bernoulli( x, &bx, &dbx, &bMx, &dbMx);
	printf( "\nbernoulli: x = %e bx = %e dbx = %e bMx = %e dbMx = %e ", x, bx, dbx, bMx, dbMx );
    }
    for( x = 0.0; x >= -100.0 ; x -= 1.0 ) {
	bernoulliNew( x, &bx, &dbx, &bMx, &dbMx, 1);
	printf( "\nbernoulliNew: x = %e bx = %e dbx = %e bMx = %e dbMx = %e ", x, bx, dbx, bMx, dbMx );
	bernoulli( x, &bx, &dbx, &bMx, &dbMx);
	printf( "\nbernoulli: x = %e bx = %e dbx = %e bMx = %e dbMx = %e ", x, bx, dbx, bMx, dbMx );
    }
}

*/
