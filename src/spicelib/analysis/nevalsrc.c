/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

/*
 * NevalSrc (noise, lnNoise, ckt, type, node1, node2, param)
 *   This routine evaluates the noise due to different physical
 *   phenomena.  This includes the "shot" noise associated with dc
 *   currents in semiconductors and the "thermal" noise associated with
 *   resistance.  Although semiconductors also display "flicker" (1/f)
 *   noise, the lack of a unified model requires us to handle it on a
 *   "case by case" basis.  What we CAN provide, though, is the noise
 *   gain associated with the 1/f source.
 */


#include "ngspice.h"
#include "cktdefs.h"
#include "const.h"
#include "noisedef.h"


void
NevalSrc (double *noise, double *lnNoise, CKTcircuit *ckt, int type, int node1, int node2, double param)
{
    double realVal;
    double imagVal;
    double gain;

    realVal = *((ckt->CKTrhs) + node1) - *((ckt->CKTrhs) + node2);
    imagVal = *((ckt->CKTirhs) + node1) - *((ckt->CKTirhs) + node2);
    gain = (realVal*realVal) + (imagVal*imagVal);
    switch (type) {

    case SHOTNOISE:
	*noise = gain * 2 * CHARGE * fabs(param);          /* param is the dc current in a semiconductor */
  	*lnNoise = log( MAX(*noise,N_MINLOG) ); 
	break;

    case THERMNOISE:
	*noise = gain * 4 * CONSTboltz * ckt->CKTtemp * param;         /* param is the conductance of a resistor */
        *lnNoise = log( MAX(*noise,N_MINLOG) );
        break;

    case N_GAIN:
	*noise = gain;
	break;

    }
}


/*
PN 2003:
The following function includes instance dtemp in 
thermal noise calculation. 
It will replace NevalSrc as soon as all devices 
will implement dtemp feature.
*/

void
NevalSrc2 (double *noise, double *lnNoise, CKTcircuit *ckt, int type, 
           int node1, int node2, double param, double param2)
{
    double realVal;
    double imagVal;
    double gain;

    realVal = *((ckt->CKTrhs) + node1) - *((ckt->CKTrhs) + node2);
    imagVal = *((ckt->CKTirhs) + node1) - *((ckt->CKTirhs) + node2);
    gain = (realVal*realVal) + (imagVal*imagVal);
    switch (type) {

    case SHOTNOISE:
	*noise = gain * 2 * CHARGE * fabs(param);          /* param is the dc current in a semiconductor */
  	*lnNoise = log( MAX(*noise,N_MINLOG) ); 
	break;

    case THERMNOISE:
	*noise = gain * 4 * CONSTboltz * (ckt->CKTtemp + param2)  /* param2 is the instance temperature difference */ 
	         * param;                                         /* param is the conductance of a resistor */
        *lnNoise = log( MAX(*noise,N_MINLOG) );
        break;

    case N_GAIN:
	*noise = gain;
	break;

    }
}
