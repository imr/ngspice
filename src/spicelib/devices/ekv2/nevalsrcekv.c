/*
 * Author: 2000 Wladek Grabinski; EKV v2.6 Model Upgrade
 * Author: 1997 Eckhard Brass;    EKV v2.5 Model Implementation
 *     (C) 1990 Regents of the University of California. Spice3 Format
 */

/*
 * NevalSrcEKV (noise, lnNoise, ckt, type, node1, node2, param)
 *   This routine evaluates the noise due to different physical
 *   phenomena.  This includes the "shot" noise associated with dc
 *   currents in semiconductors and the "thermal" noise associated with
 *   resistance.  Although semiconductors also display "flicker" (1/f)
 *   noise, the lack of a unified model requires us to handle it on a
 *   "case by case" basis.  What we CAN provide, though, is the noise
 *   gain associated with the 1/f source.
 */


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"
#include "ekvdefs.h"

void
NevalSrcEKV (
double ,
EKVmodel *,
EKVinstance *,
double *,
double *,
CKTcircuit *,
int ,
int ,
int ,
double);

void
NevalSrcEKV (
double temp,
EKVmodel *model,
EKVinstance *here,
double *noise,
double *lnNoise,
CKTcircuit *ckt,
int type,
int node1,
int node2,
double param)
{
	double realVal;
	double imagVal;
	double gain;
	double alpha, sqrtalpha_1, gamma;

	realVal = *((ckt->CKTrhs) + node1) - *((ckt->CKTrhs) + node2);
	imagVal = *((ckt->CKTirhs) + node1) - *((ckt->CKTirhs) + node2);
	gain = (realVal*realVal) + (imagVal*imagVal);
	switch (type) {

	case SHOTNOISE:
		*noise = gain * 2 * CHARGE * fabs(param);
		/* param is the dc current in a semiconductor */
		*lnNoise = log( MAX(*noise,N_MINLOG) );
		break;

	case THERMNOISE:
		if (model->EKVnlevel==1) {
			*noise = gain * 4 * CONSTboltz * temp * param;
			/* param is the conductance of a resistor */
		}
		else {
			alpha=here->EKVir/here->EKVif;
			sqrtalpha_1 = sqrt(alpha) + 1;
			gamma = 0.5 * (1.0 + alpha);
			gamma += 2.0/3.0 * here->EKVif * (sqrtalpha_1 + alpha);
			gamma /= sqrtalpha_1 * (1.0 + here->EKVif);
			*noise = gain * 4 * CONSTboltz * temp * gamma * fabs(here->EKVgms);
		}
		*lnNoise = log( MAX(*noise,N_MINLOG) );
		break;

	case N_GAIN:
		*noise = gain;
		break;

	}
}
