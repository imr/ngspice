/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1pzld.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim3v1sdef.h"
#include "suffix.h"

int
BSIM3v1SpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

    for (; model != NULL; model = model->BSIM3v1SnextModel) 
    {    for (here = model->BSIM3v1Sinstances; here!= NULL;
              here = here->BSIM3v1SnextInstance) 
	 {  
            if (here->BSIM3v1Sowner != ARCHme) 
	            continue;
		    
            if (here->BSIM3v1Smode >= 0) 
	    {   Gm = here->BSIM3v1Sgm;
		Gmbs = here->BSIM3v1Sgmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->BSIM3v1Scggb;
                cgsb = here->BSIM3v1Scgsb;
                cgdb = here->BSIM3v1Scgdb;

                cbgb = here->BSIM3v1Scbgb;
                cbsb = here->BSIM3v1Scbsb;
                cbdb = here->BSIM3v1Scbdb;

                cdgb = here->BSIM3v1Scdgb;
                cdsb = here->BSIM3v1Scdsb;
                cddb = here->BSIM3v1Scddb;
            }
	    else
	    {   Gm = -here->BSIM3v1Sgm;
		Gmbs = -here->BSIM3v1Sgmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->BSIM3v1Scggb;
                cgsb = here->BSIM3v1Scgdb;
                cgdb = here->BSIM3v1Scgsb;

                cbgb = here->BSIM3v1Scbgb;
                cbsb = here->BSIM3v1Scbdb;
                cbdb = here->BSIM3v1Scbsb;

                cdgb = -(here->BSIM3v1Scdgb + cggb + cbgb);
                cdsb = -(here->BSIM3v1Scddb + cgsb + cbsb);
                cddb = -(here->BSIM3v1Scdsb + cgdb + cbdb);
            }
            gdpr=here->BSIM3v1SdrainConductance;
            gspr=here->BSIM3v1SsourceConductance;
            gds= here->BSIM3v1Sgds;
            gbd= here->BSIM3v1Sgbd;
            gbs= here->BSIM3v1Sgbs;
            capbd= here->BSIM3v1Scapbd;
            capbs= here->BSIM3v1Scapbs;
	    GSoverlapCap = here->BSIM3v1Scgso;
	    GDoverlapCap = here->BSIM3v1Scgdo;
	    GBoverlapCap = here->pParam->BSIM3v1Scgbo;

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


            *(here->BSIM3v1SGgPtr ) += xcggb * s->real;
            *(here->BSIM3v1SGgPtr +1) += xcggb * s->imag;
            *(here->BSIM3v1SBbPtr ) += (-xcbgb-xcbdb-xcbsb) * s->real;
            *(here->BSIM3v1SBbPtr +1) += (-xcbgb-xcbdb-xcbsb) * s->imag;
            *(here->BSIM3v1SDPdpPtr ) += xcddb * s->real;
            *(here->BSIM3v1SDPdpPtr +1) += xcddb * s->imag;
            *(here->BSIM3v1SSPspPtr ) += xcssb * s->real;
            *(here->BSIM3v1SSPspPtr +1) += xcssb * s->imag;
            *(here->BSIM3v1SGbPtr ) += (-xcggb-xcgdb-xcgsb) * s->real;
            *(here->BSIM3v1SGbPtr +1) += (-xcggb-xcgdb-xcgsb) * s->imag;
            *(here->BSIM3v1SGdpPtr ) += xcgdb * s->real;
            *(here->BSIM3v1SGdpPtr +1) += xcgdb * s->imag;
            *(here->BSIM3v1SGspPtr ) += xcgsb * s->real;
            *(here->BSIM3v1SGspPtr +1) += xcgsb * s->imag;
            *(here->BSIM3v1SBgPtr ) += xcbgb * s->real;
            *(here->BSIM3v1SBgPtr +1) += xcbgb * s->imag;
            *(here->BSIM3v1SBdpPtr ) += xcbdb * s->real;
            *(here->BSIM3v1SBdpPtr +1) += xcbdb * s->imag;
            *(here->BSIM3v1SBspPtr ) += xcbsb * s->real;
            *(here->BSIM3v1SBspPtr +1) += xcbsb * s->imag;
            *(here->BSIM3v1SDPgPtr ) += xcdgb * s->real;
            *(here->BSIM3v1SDPgPtr +1) += xcdgb * s->imag;
            *(here->BSIM3v1SDPbPtr ) += (-xcdgb-xcddb-xcdsb) * s->real;
            *(here->BSIM3v1SDPbPtr +1) += (-xcdgb-xcddb-xcdsb) * s->imag;
            *(here->BSIM3v1SDPspPtr ) += xcdsb * s->real;
            *(here->BSIM3v1SDPspPtr +1) += xcdsb * s->imag;
            *(here->BSIM3v1SSPgPtr ) += xcsgb * s->real;
            *(here->BSIM3v1SSPgPtr +1) += xcsgb * s->imag;
            *(here->BSIM3v1SSPbPtr ) += (-xcsgb-xcsdb-xcssb) * s->real;
            *(here->BSIM3v1SSPbPtr +1) += (-xcsgb-xcsdb-xcssb) * s->imag;
            *(here->BSIM3v1SSPdpPtr ) += xcsdb * s->real;
            *(here->BSIM3v1SSPdpPtr +1) += xcsdb * s->imag;
            *(here->BSIM3v1SDdPtr) += gdpr;
            *(here->BSIM3v1SSsPtr) += gspr;
            *(here->BSIM3v1SBbPtr) += gbd+gbs;
            *(here->BSIM3v1SDPdpPtr) += gdpr+gds+gbd+RevSum;
            *(here->BSIM3v1SSPspPtr) += gspr+gds+gbs+FwdSum;
            *(here->BSIM3v1SDdpPtr) -= gdpr;
            *(here->BSIM3v1SSspPtr) -= gspr;
            *(here->BSIM3v1SBdpPtr) -= gbd;
            *(here->BSIM3v1SBspPtr) -= gbs;
            *(here->BSIM3v1SDPdPtr) -= gdpr;
            *(here->BSIM3v1SDPgPtr) += Gm;
            *(here->BSIM3v1SDPbPtr) -= gbd - Gmbs;
            *(here->BSIM3v1SDPspPtr) -= gds + FwdSum;
            *(here->BSIM3v1SSPgPtr) -= Gm;
            *(here->BSIM3v1SSPsPtr) -= gspr;
            *(here->BSIM3v1SSPbPtr) -= gbs + Gmbs;
            *(here->BSIM3v1SSPdpPtr) -= gds + RevSum;

        }
    }
    return(OK);
}


