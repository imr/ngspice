/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3pzld.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim3v0def.h"
#include "ngspice/suffix.h"

int
BSIM3v0pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = BSIM3v0nextModel(model)) 
    {    for (here = BSIM3v0instances(model); here!= NULL;
              here = BSIM3v0nextInstance(here)) 
	 {
            if (here->BSIM3v0mode >= 0) 
	    {   Gm = here->BSIM3v0gm;
		Gmbs = here->BSIM3v0gmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->BSIM3v0cggb;
                cgsb = here->BSIM3v0cgsb;
                cgdb = here->BSIM3v0cgdb;

                cbgb = here->BSIM3v0cbgb;
                cbsb = here->BSIM3v0cbsb;
                cbdb = here->BSIM3v0cbdb;

                cdgb = here->BSIM3v0cdgb;
                cdsb = here->BSIM3v0cdsb;
                cddb = here->BSIM3v0cddb;
            }
	    else
	    {   Gm = -here->BSIM3v0gm;
		Gmbs = -here->BSIM3v0gmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->BSIM3v0cggb;
                cgsb = here->BSIM3v0cgdb;
                cgdb = here->BSIM3v0cgsb;

                cbgb = here->BSIM3v0cbgb;
                cbsb = here->BSIM3v0cbdb;
                cbdb = here->BSIM3v0cbsb;

                cdgb = -(here->BSIM3v0cdgb + cggb + cbgb);
                cdsb = -(here->BSIM3v0cddb + cgsb + cbsb);
                cddb = -(here->BSIM3v0cdsb + cgdb + cbdb);
            }
            gdpr=here->BSIM3v0drainConductance;
            gspr=here->BSIM3v0sourceConductance;
            gds= here->BSIM3v0gds;
            gbd= here->BSIM3v0gbd;
            gbs= here->BSIM3v0gbs;
            capbd= here->BSIM3v0capbd;
            capbs= here->BSIM3v0capbs;
	    GSoverlapCap = here->BSIM3v0cgso;
	    GDoverlapCap = here->BSIM3v0cgdo;
	    GBoverlapCap = here->pParam->BSIM3v0cgbo;

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

	    m = here->BSIM3v0m;
	    
            *(here->BSIM3v0GgPtr ) += m * (xcggb * s->real);
            *(here->BSIM3v0GgPtr +1) += m * (xcggb * s->imag);
            *(here->BSIM3v0BbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->BSIM3v0BbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->BSIM3v0DPdpPtr ) += m * (xcddb * s->real);
            *(here->BSIM3v0DPdpPtr +1) += m * (xcddb * s->imag);
            *(here->BSIM3v0SPspPtr ) += m * (xcssb * s->real);
            *(here->BSIM3v0SPspPtr +1) += m * (xcssb * s->imag);
            *(here->BSIM3v0GbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->BSIM3v0GbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->BSIM3v0GdpPtr ) += m * (xcgdb * s->real);
            *(here->BSIM3v0GdpPtr +1) += m * (xcgdb * s->imag);
            *(here->BSIM3v0GspPtr ) += m * (xcgsb * s->real);
            *(here->BSIM3v0GspPtr +1) += m * (xcgsb * s->imag);
            *(here->BSIM3v0BgPtr ) += m * (xcbgb * s->real);
            *(here->BSIM3v0BgPtr +1) += m * (xcbgb * s->imag);
            *(here->BSIM3v0BdpPtr ) += m * (xcbdb * s->real);
            *(here->BSIM3v0BdpPtr +1) += m * (xcbdb * s->imag);
            *(here->BSIM3v0BspPtr ) += m * (xcbsb * s->real);
            *(here->BSIM3v0BspPtr +1) += m * (xcbsb * s->imag);
            *(here->BSIM3v0DPgPtr ) += m * (xcdgb * s->real);
            *(here->BSIM3v0DPgPtr +1) += m * (xcdgb * s->imag);
            *(here->BSIM3v0DPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->BSIM3v0DPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->BSIM3v0DPspPtr ) += m * (xcdsb * s->real);
            *(here->BSIM3v0DPspPtr +1) += m * (xcdsb * s->imag);
            *(here->BSIM3v0SPgPtr ) += m * (xcsgb * s->real);
            *(here->BSIM3v0SPgPtr +1) += m * (xcsgb * s->imag);
            *(here->BSIM3v0SPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->BSIM3v0SPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->BSIM3v0SPdpPtr ) += m * (xcsdb * s->real);
            *(here->BSIM3v0SPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->BSIM3v0DdPtr) += m * gdpr;
            *(here->BSIM3v0SsPtr) += m * gspr;
            *(here->BSIM3v0BbPtr) += m * (gbd + gbs);
            *(here->BSIM3v0DPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->BSIM3v0SPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->BSIM3v0DdpPtr) -= m * gdpr;
            *(here->BSIM3v0SspPtr) -= m * gspr;
            *(here->BSIM3v0BdpPtr) -= m * gbd;
            *(here->BSIM3v0BspPtr) -= m * gbs;
            *(here->BSIM3v0DPdPtr) -= m * gdpr;
            *(here->BSIM3v0DPgPtr) += m * Gm;
            *(here->BSIM3v0DPbPtr) -= m * (gbd - Gmbs);
            *(here->BSIM3v0DPspPtr) -= m * (gds + FwdSum);
            *(here->BSIM3v0SPgPtr) -= m * Gm;
            *(here->BSIM3v0SPsPtr) -= m * gspr;
            *(here->BSIM3v0SPbPtr) -= m * (gbs + Gmbs);
            *(here->BSIM3v0SPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


