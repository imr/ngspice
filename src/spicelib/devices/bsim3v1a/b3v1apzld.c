/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1apzld.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim3v1adef.h"
#include "suffix.h"

int
BSIM3v1ApzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    for (; model != NULL; model = model->BSIM3v1AnextModel) 
    {    for (here = model->BSIM3v1Ainstances; here!= NULL;
              here = here->BSIM3v1AnextInstance) 
	 {
            
	     if (here->BSIM3v1Aowner != ARCHme)
	           continue;

	    
	    if (here->BSIM3v1Amode >= 0) 
	    {   Gm = here->BSIM3v1Agm;
		Gmbs = here->BSIM3v1Agmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->BSIM3v1Acggb;
                cgsb = here->BSIM3v1Acgsb;
                cgdb = here->BSIM3v1Acgdb;

                cbgb = here->BSIM3v1Acbgb;
                cbsb = here->BSIM3v1Acbsb;
                cbdb = here->BSIM3v1Acbdb;

                cdgb = here->BSIM3v1Acdgb;
                cdsb = here->BSIM3v1Acdsb;
                cddb = here->BSIM3v1Acddb;
            }
	    else
	    {   Gm = -here->BSIM3v1Agm;
		Gmbs = -here->BSIM3v1Agmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->BSIM3v1Acggb;
                cgsb = here->BSIM3v1Acgdb;
                cgdb = here->BSIM3v1Acgsb;

                cbgb = here->BSIM3v1Acbgb;
                cbsb = here->BSIM3v1Acbdb;
                cbdb = here->BSIM3v1Acbsb;

                cdgb = -(here->BSIM3v1Acdgb + cggb + cbgb);
                cdsb = -(here->BSIM3v1Acddb + cgsb + cbsb);
                cddb = -(here->BSIM3v1Acdsb + cgdb + cbdb);
            }
            gdpr=here->BSIM3v1AdrainConductance;
            gspr=here->BSIM3v1AsourceConductance;
            gds= here->BSIM3v1Agds;
            gbd= here->BSIM3v1Agbd;
            gbs= here->BSIM3v1Agbs;
            capbd= here->BSIM3v1Acapbd;
            capbs= here->BSIM3v1Acapbs;
	    GSoverlapCap = here->BSIM3v1Acgso;
	    GDoverlapCap = here->BSIM3v1Acgdo;
	    GBoverlapCap = here->pParam->BSIM3v1Acgbo;

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

            m = here->BSIM3v1Am;

            *(here->BSIM3v1AGgPtr ) += m * (xcggb * s->real);
            *(here->BSIM3v1AGgPtr +1) += m * (xcggb * s->imag);
            *(here->BSIM3v1ABbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->BSIM3v1ABbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->BSIM3v1ADPdpPtr ) += m * (xcddb * s->real);
            *(here->BSIM3v1ADPdpPtr +1) += m * (xcddb * s->imag);
            *(here->BSIM3v1ASPspPtr ) += m * (xcssb * s->real);
            *(here->BSIM3v1ASPspPtr +1) += m * (xcssb * s->imag);
            *(here->BSIM3v1AGbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->BSIM3v1AGbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->BSIM3v1AGdpPtr ) += m * (xcgdb * s->real);
            *(here->BSIM3v1AGdpPtr +1) += m * (xcgdb * s->imag);
            *(here->BSIM3v1AGspPtr ) += m * (xcgsb * s->real);
            *(here->BSIM3v1AGspPtr +1) += m * (xcgsb * s->imag);
            *(here->BSIM3v1ABgPtr ) += m * (xcbgb * s->real);
            *(here->BSIM3v1ABgPtr +1) += m * (xcbgb * s->imag);
            *(here->BSIM3v1ABdpPtr ) += m * (xcbdb * s->real);
            *(here->BSIM3v1ABdpPtr +1) += m * (xcbdb * s->imag);
            *(here->BSIM3v1ABspPtr ) += m * (xcbsb * s->real);
            *(here->BSIM3v1ABspPtr +1) += m * (xcbsb * s->imag);
            *(here->BSIM3v1ADPgPtr ) += m * (xcdgb * s->real);
            *(here->BSIM3v1ADPgPtr +1) += m * (xcdgb * s->imag);
            *(here->BSIM3v1ADPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->BSIM3v1ADPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->BSIM3v1ADPspPtr ) += m * (xcdsb * s->real);
            *(here->BSIM3v1ADPspPtr +1) += m * (xcdsb * s->imag);
            *(here->BSIM3v1ASPgPtr ) += m * (xcsgb * s->real);
            *(here->BSIM3v1ASPgPtr +1) += m * (xcsgb * s->imag);
            *(here->BSIM3v1ASPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->BSIM3v1ASPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->BSIM3v1ASPdpPtr ) += m * (xcsdb * s->real);
            *(here->BSIM3v1ASPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->BSIM3v1ADdPtr) += m * gdpr;
            *(here->BSIM3v1ASsPtr) += m * gspr;
            *(here->BSIM3v1ABbPtr) += m * (gbd + gbs);
            *(here->BSIM3v1ADPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->BSIM3v1ASPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->BSIM3v1ADdpPtr) -= m * gdpr;
            *(here->BSIM3v1ASspPtr) -= m * gspr;
            *(here->BSIM3v1ABdpPtr) -= m * gbd;
            *(here->BSIM3v1ABspPtr) -= m * gbs;
            *(here->BSIM3v1ADPdPtr) -= m * gdpr;
            *(here->BSIM3v1ADPgPtr) += m * Gm;
            *(here->BSIM3v1ADPbPtr) -= m * (gbd - Gmbs);
            *(here->BSIM3v1ADPspPtr) -= m * (gds + FwdSum);
            *(here->BSIM3v1ASPgPtr) -= m * Gm;
            *(here->BSIM3v1ASPsPtr) -= m * gspr;
            *(here->BSIM3v1ASPbPtr) -= m * (gbs + Gmbs);
            *(here->BSIM3v1ASPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


