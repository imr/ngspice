/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1pzld.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim3v1def.h"
#include "ngspice/suffix.h"

int
BSIM3v1pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = BSIM3v1nextModel(model)) 
    {    for (here = BSIM3v1instances(model); here!= NULL;
              here = BSIM3v1nextInstance(here)) 
	 {
            if (here->BSIM3v1mode >= 0) 
	    {   Gm = here->BSIM3v1gm;
		Gmbs = here->BSIM3v1gmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->BSIM3v1cggb;
                cgsb = here->BSIM3v1cgsb;
                cgdb = here->BSIM3v1cgdb;

                cbgb = here->BSIM3v1cbgb;
                cbsb = here->BSIM3v1cbsb;
                cbdb = here->BSIM3v1cbdb;

                cdgb = here->BSIM3v1cdgb;
                cdsb = here->BSIM3v1cdsb;
                cddb = here->BSIM3v1cddb;
            }
	    else
	    {   Gm = -here->BSIM3v1gm;
		Gmbs = -here->BSIM3v1gmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->BSIM3v1cggb;
                cgsb = here->BSIM3v1cgdb;
                cgdb = here->BSIM3v1cgsb;

                cbgb = here->BSIM3v1cbgb;
                cbsb = here->BSIM3v1cbdb;
                cbdb = here->BSIM3v1cbsb;

                cdgb = -(here->BSIM3v1cdgb + cggb + cbgb);
                cdsb = -(here->BSIM3v1cddb + cgsb + cbsb);
                cddb = -(here->BSIM3v1cdsb + cgdb + cbdb);
            }
            gdpr=here->BSIM3v1drainConductance;
            gspr=here->BSIM3v1sourceConductance;
            gds= here->BSIM3v1gds;
            gbd= here->BSIM3v1gbd;
            gbs= here->BSIM3v1gbs;
            capbd= here->BSIM3v1capbd;
            capbs= here->BSIM3v1capbs;
	    GSoverlapCap = here->BSIM3v1cgso;
	    GDoverlapCap = here->BSIM3v1cgdo;
	    GBoverlapCap = here->pParam->BSIM3v1cgbo;

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

            m = here->BSIM3v1m;

            *(here->BSIM3v1GgPtr ) += m * (xcggb * s->real);
            *(here->BSIM3v1GgPtr +1) += m * (xcggb * s->imag);
            *(here->BSIM3v1BbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->BSIM3v1BbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->BSIM3v1DPdpPtr ) += m * (xcddb * s->real);
            *(here->BSIM3v1DPdpPtr +1) += xcddb * s->imag;
            *(here->BSIM3v1SPspPtr ) += m * (xcssb * s->real);
            *(here->BSIM3v1SPspPtr +1) += m * (xcssb * s->imag);
            *(here->BSIM3v1GbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->BSIM3v1GbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->BSIM3v1GdpPtr ) += m * (xcgdb * s->real);
            *(here->BSIM3v1GdpPtr +1) += m * (xcgdb * s->imag);
            *(here->BSIM3v1GspPtr ) += m * (xcgsb * s->real);
            *(here->BSIM3v1GspPtr +1) += m * (xcgsb * s->imag);
            *(here->BSIM3v1BgPtr ) += m * (xcbgb * s->real);
            *(here->BSIM3v1BgPtr +1) += m * (xcbgb * s->imag);
            *(here->BSIM3v1BdpPtr ) += m * (xcbdb * s->real);
            *(here->BSIM3v1BdpPtr +1) += m * (xcbdb * s->imag);
            *(here->BSIM3v1BspPtr ) += m * (xcbsb * s->real);
            *(here->BSIM3v1BspPtr +1) += m * (xcbsb * s->imag);
            *(here->BSIM3v1DPgPtr ) += m * (xcdgb * s->real);
            *(here->BSIM3v1DPgPtr +1) += m * (xcdgb * s->imag);
            *(here->BSIM3v1DPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->BSIM3v1DPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->BSIM3v1DPspPtr ) += m * (xcdsb * s->real);
            *(here->BSIM3v1DPspPtr +1) += m * (xcdsb * s->imag);
            *(here->BSIM3v1SPgPtr ) += m * (xcsgb * s->real);
            *(here->BSIM3v1SPgPtr +1) += m * (xcsgb * s->imag);
            *(here->BSIM3v1SPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->BSIM3v1SPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->BSIM3v1SPdpPtr ) += m * (xcsdb * s->real);
            *(here->BSIM3v1SPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->BSIM3v1DdPtr) += m * gdpr;
            *(here->BSIM3v1SsPtr) += m * gspr;
            *(here->BSIM3v1BbPtr) += m * (gbd + gbs);
            *(here->BSIM3v1DPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->BSIM3v1SPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->BSIM3v1DdpPtr) -= m * gdpr;
            *(here->BSIM3v1SspPtr) -= m * gspr;
            *(here->BSIM3v1BdpPtr) -= m * gbd;
            *(here->BSIM3v1BspPtr) -= m * gbs;
            *(here->BSIM3v1DPdPtr) -= m * gdpr;
            *(here->BSIM3v1DPgPtr) += m * Gm;
            *(here->BSIM3v1DPbPtr) -= m * (gbd - Gmbs);
            *(here->BSIM3v1DPspPtr) -= m * (gds + FwdSum);
            *(here->BSIM3v1SPgPtr) -= m * Gm;
            *(here->BSIM3v1SPsPtr) -= m * gspr;
            *(here->BSIM3v1SPbPtr) -= m * (gbs + Gmbs);
            *(here->BSIM3v1SPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


