/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim2def.h"
#include "sperror.h"
#include "suffix.h"


int
B2acLoad(inModel,ckt)
    GENmodel *inModel;
    register CKTcircuit *ckt;
{
    register B2model *model = (B2model*)inModel;
    register B2instance *here;
    int xnrm;
    int xrev;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double gmbs;
    double gbd;
    double gbs;
    double capbd;
    double capbs;
    double xcggb;
    double xcgdb;
    double xcgsb;
    double xcbgb;
    double xcbdb;
    double xcbsb;
    double xcddb;
    double xcssb;
    double xcdgb;
    double xcsgb;
    double xcdsb;
    double xcsdb;
    double cggb;
    double cgdb;
    double cgsb;
    double cbgb;
    double cbdb;
    double cbsb;
    double cddb;
    double cdgb;
    double cdsb;
    double omega; /* angular fequency of the signal */

    omega = ckt->CKTomega;
    for( ; model != NULL; model = model->B2nextModel) {
        for(here = model->B2instances; here!= NULL;
                here = here->B2nextInstance) {
	    if (here->B2owner != ARCHme) continue;
        
            if (here->B2mode >= 0) {
                xnrm=1;
                xrev=0;
            } else {
                xnrm=0;
                xrev=1;
            }
            gdpr=here->B2drainConductance;
            gspr=here->B2sourceConductance;
            gm= *(ckt->CKTstate0 + here->B2gm);
            gds= *(ckt->CKTstate0 + here->B2gds);
            gmbs= *(ckt->CKTstate0 + here->B2gmbs);
            gbd= *(ckt->CKTstate0 + here->B2gbd);
            gbs= *(ckt->CKTstate0 + here->B2gbs);
            capbd= *(ckt->CKTstate0 + here->B2capbd);
            capbs= *(ckt->CKTstate0 + here->B2capbs);
            /*
             *    charge oriented model parameters
             */

            cggb = *(ckt->CKTstate0 + here->B2cggb);
            cgsb = *(ckt->CKTstate0 + here->B2cgsb);
            cgdb = *(ckt->CKTstate0 + here->B2cgdb);

            cbgb = *(ckt->CKTstate0 + here->B2cbgb);
            cbsb = *(ckt->CKTstate0 + here->B2cbsb);
            cbdb = *(ckt->CKTstate0 + here->B2cbdb);

            cdgb = *(ckt->CKTstate0 + here->B2cdgb);
            cdsb = *(ckt->CKTstate0 + here->B2cdsb);
            cddb = *(ckt->CKTstate0 + here->B2cddb);

            xcdgb = (cdgb - here->pParam->B2GDoverlapCap) * omega;
            xcddb = (cddb + capbd + here->pParam->B2GDoverlapCap) * omega;
            xcdsb = cdsb * omega;
            xcsgb = -(cggb + cbgb + cdgb + here->pParam->B2GSoverlapCap)
		  * omega;
            xcsdb = -(cgdb + cbdb + cddb) * omega;
            xcssb = (capbs + here->pParam->B2GSoverlapCap
		  - (cgsb+cbsb+cdsb)) * omega;
            xcggb = (cggb + here->pParam->B2GDoverlapCap 
		  + here->pParam->B2GSoverlapCap
		  + here->pParam->B2GBoverlapCap) * omega;
            xcgdb = (cgdb - here->pParam->B2GDoverlapCap ) * omega;
            xcgsb = (cgsb - here->pParam->B2GSoverlapCap) * omega;
            xcbgb = (cbgb - here->pParam->B2GBoverlapCap) * omega;
            xcbdb = (cbdb - capbd ) * omega;
            xcbsb = (cbsb - capbs ) * omega;


            *(here->B2GgPtr +1) += xcggb;
            *(here->B2BbPtr +1) += -xcbgb-xcbdb-xcbsb;
            *(here->B2DPdpPtr +1) += xcddb;
            *(here->B2SPspPtr +1) += xcssb;
            *(here->B2GbPtr +1) += -xcggb-xcgdb-xcgsb;
            *(here->B2GdpPtr +1) += xcgdb;
            *(here->B2GspPtr +1) += xcgsb;
            *(here->B2BgPtr +1) += xcbgb;
            *(here->B2BdpPtr +1) += xcbdb;
            *(here->B2BspPtr +1) += xcbsb;
            *(here->B2DPgPtr +1) += xcdgb;
            *(here->B2DPbPtr +1) += -xcdgb-xcddb-xcdsb;
            *(here->B2DPspPtr +1) += xcdsb;
            *(here->B2SPgPtr +1) += xcsgb;
            *(here->B2SPbPtr +1) += -xcsgb-xcsdb-xcssb;
            *(here->B2SPdpPtr +1) += xcsdb;
            *(here->B2DdPtr) += gdpr;
            *(here->B2SsPtr) += gspr;
            *(here->B2BbPtr) += gbd+gbs;
            *(here->B2DPdpPtr) += gdpr+gds+gbd+xrev*(gm+gmbs);
            *(here->B2SPspPtr) += gspr+gds+gbs+xnrm*(gm+gmbs);
            *(here->B2DdpPtr) -= gdpr;
            *(here->B2SspPtr) -= gspr;
            *(here->B2BdpPtr) -= gbd;
            *(here->B2BspPtr) -= gbs;
            *(here->B2DPdPtr) -= gdpr;
            *(here->B2DPgPtr) += (xnrm-xrev)*gm;
            *(here->B2DPbPtr) += -gbd+(xnrm-xrev)*gmbs;
            *(here->B2DPspPtr) += -gds-xnrm*(gm+gmbs);
            *(here->B2SPgPtr) += -(xnrm-xrev)*gm;
            *(here->B2SPsPtr) -= gspr;
            *(here->B2SPbPtr) += -gbs-(xnrm-xrev)*gmbs;
            *(here->B2SPdpPtr) += -gds-xrev*(gm+gmbs);

        }
    }
return(OK);
}



