/* $Id$  */
/* 
$Log$
Revision 1.1  2000-04-27 20:03:59  pnenzi
Initial revision

 * Revision 3.1  96/12/08  19:58:46  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1pzld.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim3v1def.h"
#include "suffix.h"

int
BSIM3V1pzLoad(inModel,ckt,s)
GENmodel *inModel;
register CKTcircuit *ckt;
register SPcomplex *s;
{
register BSIM3V1model *model = (BSIM3V1model*)inModel;
register BSIM3V1instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

    for (; model != NULL; model = model->BSIM3V1nextModel) 
    {    for (here = model->BSIM3V1instances; here!= NULL;
              here = here->BSIM3V1nextInstance) 
	 {  
            if (here->BSIM3V1owner != ARCHme) continue;
            if (here->BSIM3V1mode >= 0) 
	    {   Gm = here->BSIM3V1gm;
		Gmbs = here->BSIM3V1gmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->BSIM3V1cggb;
                cgsb = here->BSIM3V1cgsb;
                cgdb = here->BSIM3V1cgdb;

                cbgb = here->BSIM3V1cbgb;
                cbsb = here->BSIM3V1cbsb;
                cbdb = here->BSIM3V1cbdb;

                cdgb = here->BSIM3V1cdgb;
                cdsb = here->BSIM3V1cdsb;
                cddb = here->BSIM3V1cddb;
            }
	    else
	    {   Gm = -here->BSIM3V1gm;
		Gmbs = -here->BSIM3V1gmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->BSIM3V1cggb;
                cgsb = here->BSIM3V1cgdb;
                cgdb = here->BSIM3V1cgsb;

                cbgb = here->BSIM3V1cbgb;
                cbsb = here->BSIM3V1cbdb;
                cbdb = here->BSIM3V1cbsb;

                cdgb = -(here->BSIM3V1cdgb + cggb + cbgb);
                cdsb = -(here->BSIM3V1cddb + cgsb + cbsb);
                cddb = -(here->BSIM3V1cdsb + cgdb + cbdb);
            }
            gdpr=here->BSIM3V1drainConductance;
            gspr=here->BSIM3V1sourceConductance;
            gds= here->BSIM3V1gds;
            gbd= here->BSIM3V1gbd;
            gbs= here->BSIM3V1gbs;
            capbd= here->BSIM3V1capbd;
            capbs= here->BSIM3V1capbs;
	    GSoverlapCap = here->BSIM3V1cgso;
	    GDoverlapCap = here->BSIM3V1cgdo;
	    GBoverlapCap = here->pParam->BSIM3V1cgbo;

            xcdgb = (cdgb - GDoverlapCap);
            xcddb = (cddb + capbd + GDoverlapCap);
            xcdsb = cdsb;
            xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap);
            xcsdb = -(cgdb + cbdb + cddb);
            xcssb = (capbs + GSoverlapCap - (cgsb+cbsb+cdsb));
            xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap);
            xcgdb = (cgdb - GDoverlapCap);
            xcgsb = (cgsb - GSoverlapCap);
            xcbgb = (cbgb - GBoverlapCap);
            xcbdb = (cbdb - capbd);
            xcbsb = (cbsb - capbs);


            *(here->BSIM3V1GgPtr ) += xcggb * s->real;
            *(here->BSIM3V1GgPtr +1) += xcggb * s->imag;
            *(here->BSIM3V1BbPtr ) += (-xcbgb-xcbdb-xcbsb) * s->real;
            *(here->BSIM3V1BbPtr +1) += (-xcbgb-xcbdb-xcbsb) * s->imag;
            *(here->BSIM3V1DPdpPtr ) += xcddb * s->real;
            *(here->BSIM3V1DPdpPtr +1) += xcddb * s->imag;
            *(here->BSIM3V1SPspPtr ) += xcssb * s->real;
            *(here->BSIM3V1SPspPtr +1) += xcssb * s->imag;
            *(here->BSIM3V1GbPtr ) += (-xcggb-xcgdb-xcgsb) * s->real;
            *(here->BSIM3V1GbPtr +1) += (-xcggb-xcgdb-xcgsb) * s->imag;
            *(here->BSIM3V1GdpPtr ) += xcgdb * s->real;
            *(here->BSIM3V1GdpPtr +1) += xcgdb * s->imag;
            *(here->BSIM3V1GspPtr ) += xcgsb * s->real;
            *(here->BSIM3V1GspPtr +1) += xcgsb * s->imag;
            *(here->BSIM3V1BgPtr ) += xcbgb * s->real;
            *(here->BSIM3V1BgPtr +1) += xcbgb * s->imag;
            *(here->BSIM3V1BdpPtr ) += xcbdb * s->real;
            *(here->BSIM3V1BdpPtr +1) += xcbdb * s->imag;
            *(here->BSIM3V1BspPtr ) += xcbsb * s->real;
            *(here->BSIM3V1BspPtr +1) += xcbsb * s->imag;
            *(here->BSIM3V1DPgPtr ) += xcdgb * s->real;
            *(here->BSIM3V1DPgPtr +1) += xcdgb * s->imag;
            *(here->BSIM3V1DPbPtr ) += (-xcdgb-xcddb-xcdsb) * s->real;
            *(here->BSIM3V1DPbPtr +1) += (-xcdgb-xcddb-xcdsb) * s->imag;
            *(here->BSIM3V1DPspPtr ) += xcdsb * s->real;
            *(here->BSIM3V1DPspPtr +1) += xcdsb * s->imag;
            *(here->BSIM3V1SPgPtr ) += xcsgb * s->real;
            *(here->BSIM3V1SPgPtr +1) += xcsgb * s->imag;
            *(here->BSIM3V1SPbPtr ) += (-xcsgb-xcsdb-xcssb) * s->real;
            *(here->BSIM3V1SPbPtr +1) += (-xcsgb-xcsdb-xcssb) * s->imag;
            *(here->BSIM3V1SPdpPtr ) += xcsdb * s->real;
            *(here->BSIM3V1SPdpPtr +1) += xcsdb * s->imag;
            *(here->BSIM3V1DdPtr) += gdpr;
            *(here->BSIM3V1SsPtr) += gspr;
            *(here->BSIM3V1BbPtr) += gbd+gbs;
            *(here->BSIM3V1DPdpPtr) += gdpr+gds+gbd+RevSum;
            *(here->BSIM3V1SPspPtr) += gspr+gds+gbs+FwdSum;
            *(here->BSIM3V1DdpPtr) -= gdpr;
            *(here->BSIM3V1SspPtr) -= gspr;
            *(here->BSIM3V1BdpPtr) -= gbd;
            *(here->BSIM3V1BspPtr) -= gbs;
            *(here->BSIM3V1DPdPtr) -= gdpr;
            *(here->BSIM3V1DPgPtr) += Gm;
            *(here->BSIM3V1DPbPtr) -= gbd - Gmbs;
            *(here->BSIM3V1DPspPtr) -= gds + FwdSum;
            *(here->BSIM3V1SPgPtr) -= Gm;
            *(here->BSIM3V1SPsPtr) -= gspr;
            *(here->BSIM3V1SPbPtr) -= gbs + Gmbs;
            *(here->BSIM3V1SPdpPtr) -= gds + RevSum;

        }
    }
    return(OK);
}


