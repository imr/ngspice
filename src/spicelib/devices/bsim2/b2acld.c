/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B2acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model*)inModel;
    B2instance *here;
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

    double m;     /* parallel multiplier */

    omega = ckt->CKTomega;
    for( ; model != NULL; model = B2nextModel(model)) {
        for(here = B2instances(model); here!= NULL;
                here = B2nextInstance(here)) {
        
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

            m = here->B2m;

            *(here->B2GgPtr +1) +=   m * (xcggb);
            *(here->B2BbPtr +1) +=   m * (-xcbgb-xcbdb-xcbsb);
            *(here->B2DPdpPtr +1) += m * (xcddb);
            *(here->B2SPspPtr +1) += m * (xcssb);
            *(here->B2GbPtr +1) +=   m * (-xcggb-xcgdb-xcgsb);
            *(here->B2GdpPtr +1) +=  m * (xcgdb);
            *(here->B2GspPtr +1) +=  m * (xcgsb);
            *(here->B2BgPtr +1) +=   m * (xcbgb);
            *(here->B2BdpPtr +1) +=  m * (xcbdb);
            *(here->B2BspPtr +1) +=  m * (xcbsb);
            *(here->B2DPgPtr +1) +=  m * (xcdgb);
            *(here->B2DPbPtr +1) +=  m * (-xcdgb-xcddb-xcdsb);
            *(here->B2DPspPtr +1) += m * (xcdsb);
            *(here->B2SPgPtr +1) +=  m * (xcsgb);
            *(here->B2SPbPtr +1) +=  m * (-xcsgb-xcsdb-xcssb);
            *(here->B2SPdpPtr +1) += m * (xcsdb);
            *(here->B2DdPtr) +=      m * (gdpr);
            *(here->B2SsPtr) +=      m * (gspr);
            *(here->B2BbPtr) +=      m * (gbd+gbs);
            *(here->B2DPdpPtr) +=    m * (gdpr+gds+gbd+xrev*(gm+gmbs));
            *(here->B2SPspPtr) +=    m * (gspr+gds+gbs+xnrm*(gm+gmbs));
            *(here->B2DdpPtr) -=     m * (gdpr);
            *(here->B2SspPtr) -=     m * (gspr);
            *(here->B2BdpPtr) -=     m * (gbd);
            *(here->B2BspPtr) -=     m * (gbs);
            *(here->B2DPdPtr) -=     m * (gdpr);
            *(here->B2DPgPtr) +=     m * ((xnrm-xrev)*gm);
            *(here->B2DPbPtr) +=     m * (-gbd+(xnrm-xrev)*gmbs);
            *(here->B2DPspPtr) +=    m * (-gds-xnrm*(gm+gmbs));
            *(here->B2SPgPtr) +=     m * (-(xnrm-xrev)*gm);
            *(here->B2SPsPtr) -=     m * (gspr);
            *(here->B2SPbPtr) +=     m * (-gbs-(xnrm-xrev)*gmbs);
            *(here->B2SPdpPtr) +=    m * (-gds-xrev*(gm+gmbs));

        }
    }
return(OK);
}



