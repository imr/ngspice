/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1sacld.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1SacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Smodel *model = (BSIM3v1Smodel*)inModel;
BSIM3v1Sinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;

double dxpart, sxpart, cqgb, cqdb, cqsb, cqbb, xcqgb, xcqdb, xcqsb, xcqbb;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM3v1SnextModel) 
    {    


      for (here = model->BSIM3v1Sinstances; here!= NULL;
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

                  cqgb = here->BSIM3v1Scqgb;
                  cqdb = here->BSIM3v1Scqdb;
                  cqsb = here->BSIM3v1Scqsb;
                  cqbb = here->BSIM3v1Scqbb;
                  sxpart = 0.6;
                  dxpart = 0.4;

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

                  cqgb = here->BSIM3v1Scqgb;
                  cqdb = here->BSIM3v1Scqsb;
                  cqsb = here->BSIM3v1Scqdb;
                  cqbb = here->BSIM3v1Scqbb;
                  sxpart = 0.4;
                  dxpart = 0.6;
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

              xcdgb = (cdgb - GDoverlapCap) * omega;
              xcddb = (cddb + capbd + GDoverlapCap) * omega;
              xcdsb = cdsb * omega;
              xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap) * omega;
              xcsdb = -(cgdb + cbdb + cddb) * omega;
              xcssb = (capbs + GSoverlapCap - (cgsb + cbsb + cdsb)) * omega;
              xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap)
		    * omega;
              xcgdb = (cgdb - GDoverlapCap ) * omega;
              xcgsb = (cgsb - GSoverlapCap) * omega;
              xcbgb = (cbgb - GBoverlapCap) * omega;
              xcbdb = (cbdb - capbd ) * omega;
              xcbsb = (cbsb - capbs ) * omega;
              xcqgb = cqgb * omega;
              xcqdb = cqdb * omega;
              xcqsb = cqsb * omega;
              xcqbb = cqbb * omega;

              *(here->BSIM3v1SGgPtr +1) += xcggb;
              *(here->BSIM3v1SBbPtr +1) -= xcbgb + xcbdb + xcbsb;
              *(here->BSIM3v1SDPdpPtr +1) += xcddb;
              *(here->BSIM3v1SSPspPtr +1) += xcssb;
              *(here->BSIM3v1SGbPtr +1) -= xcggb + xcgdb + xcgsb;
              *(here->BSIM3v1SGdpPtr +1) += xcgdb;
              *(here->BSIM3v1SGspPtr +1) += xcgsb;
              *(here->BSIM3v1SBgPtr +1) += xcbgb;
              *(here->BSIM3v1SBdpPtr +1) += xcbdb;
              *(here->BSIM3v1SBspPtr +1) += xcbsb;
              *(here->BSIM3v1SDPgPtr +1) += xcdgb;
              *(here->BSIM3v1SDPbPtr +1) -= xcdgb + xcddb + xcdsb;
              *(here->BSIM3v1SDPspPtr +1) += xcdsb;
              *(here->BSIM3v1SSPgPtr +1) += xcsgb;
              *(here->BSIM3v1SSPbPtr +1) -= xcsgb + xcsdb + xcssb;
              *(here->BSIM3v1SSPdpPtr +1) += xcsdb;
 
              *(here->BSIM3v1SQqPtr +1) += omega;

              *(here->BSIM3v1SQgPtr +1) -= xcqgb;
              *(here->BSIM3v1SQdpPtr +1) -= xcqdb;
              *(here->BSIM3v1SQspPtr +1) -= xcqsb;
              *(here->BSIM3v1SQbPtr +1) -= xcqbb;


              *(here->BSIM3v1SDdPtr) += gdpr;
              *(here->BSIM3v1SSsPtr) += gspr;
              *(here->BSIM3v1SBbPtr) += gbd + gbs;
              *(here->BSIM3v1SDPdpPtr) += gdpr + gds + gbd + RevSum + dxpart*here->BSIM3v1Sgtd;
              *(here->BSIM3v1SSPspPtr) += gspr + gds + gbs + FwdSum + sxpart*here->BSIM3v1Sgts;
              *(here->BSIM3v1SDdpPtr) -= gdpr;
              *(here->BSIM3v1SSspPtr) -= gspr;
              *(here->BSIM3v1SBdpPtr) -= gbd;
              *(here->BSIM3v1SBspPtr) -= gbs;
              *(here->BSIM3v1SDPdPtr) -= gdpr;
              *(here->BSIM3v1SDPgPtr) += Gm + dxpart * here->BSIM3v1Sgtg;
              *(here->BSIM3v1SDPbPtr) -= gbd - Gmbs - dxpart * here->BSIM3v1Sgtb;
              *(here->BSIM3v1SDPspPtr) -= gds + FwdSum - dxpart * here->BSIM3v1Sgts;
              *(here->BSIM3v1SSPgPtr) -= Gm - sxpart * here->BSIM3v1Sgtg;
              *(here->BSIM3v1SSPsPtr) -= gspr;
              *(here->BSIM3v1SSPbPtr) -= gbs + Gmbs - sxpart * here->BSIM3v1Sgtg;
              *(here->BSIM3v1SSPdpPtr) -= gds + RevSum - sxpart * here->BSIM3v1Sgtd;
              *(here->BSIM3v1SGgPtr) -= here->BSIM3v1Sgtg;
              *(here->BSIM3v1SGbPtr) -=  here->BSIM3v1Sgtb;
              *(here->BSIM3v1SGdpPtr) -= here->BSIM3v1Sgtd;
              *(here->BSIM3v1SGspPtr) -= here->BSIM3v1Sgts;

              *(here->BSIM3v1SQqPtr) += here->BSIM3v1Sgtau;
 
              *(here->BSIM3v1SDPqPtr) += dxpart * here->BSIM3v1Sgtau;
              *(here->BSIM3v1SSPqPtr) += sxpart * here->BSIM3v1Sgtau;
              *(here->BSIM3v1SGqPtr) -=  here->BSIM3v1Sgtau;
 
              *(here->BSIM3v1SQgPtr) +=  here->BSIM3v1Sgtg;
              *(here->BSIM3v1SQdpPtr) += here->BSIM3v1Sgtd;
              *(here->BSIM3v1SQspPtr) += here->BSIM3v1Sgts;
              *(here->BSIM3v1SQbPtr) += here->BSIM3v1Sgtb;

        }
    }
    return(OK);
}

