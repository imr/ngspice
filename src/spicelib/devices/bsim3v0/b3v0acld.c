/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0acld.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v0model *model = (BSIM3v0model*)inModel;
BSIM3v0instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;

double dxpart, sxpart, cqgb, cqdb, cqsb, cqbb, xcqgb, xcqdb, xcqsb, xcqbb;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = BSIM3v0nextModel(model)) 
    {    


      for (here = BSIM3v0instances(model); here!= NULL;
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

                  cqgb = here->BSIM3v0cqgb;
                  cqdb = here->BSIM3v0cqdb;
                  cqsb = here->BSIM3v0cqsb;
                  cqbb = here->BSIM3v0cqbb;
                  sxpart = 0.6;
                  dxpart = 0.4;

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

                  cqgb = here->BSIM3v0cqgb;
                  cqdb = here->BSIM3v0cqsb;
                  cqsb = here->BSIM3v0cqdb;
                  cqbb = here->BSIM3v0cqbb;
                  sxpart = 0.4;
                  dxpart = 0.6;
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
	      
	      m = here->BSIM3v0m;

              *(here->BSIM3v0GgPtr +1) += m * xcggb;
              *(here->BSIM3v0BbPtr +1) -= m * (xcbgb + xcbdb + xcbsb);
              *(here->BSIM3v0DPdpPtr +1) += m * xcddb;
              *(here->BSIM3v0SPspPtr +1) += m * xcssb;
              *(here->BSIM3v0GbPtr +1) -= m * (xcggb + xcgdb + xcgsb);
              *(here->BSIM3v0GdpPtr +1) += m * xcgdb;
              *(here->BSIM3v0GspPtr +1) += m * xcgsb;
              *(here->BSIM3v0BgPtr +1) += m * xcbgb;
              *(here->BSIM3v0BdpPtr +1) += m * xcbdb;
              *(here->BSIM3v0BspPtr +1) += m * xcbsb;
              *(here->BSIM3v0DPgPtr +1) += m * xcdgb;
              *(here->BSIM3v0DPbPtr +1) -= m * (xcdgb + xcddb + xcdsb);
              *(here->BSIM3v0DPspPtr +1) += m * xcdsb;
              *(here->BSIM3v0SPgPtr +1) += m * xcsgb;
              *(here->BSIM3v0SPbPtr +1) -= m * (xcsgb + xcsdb + xcssb);
              *(here->BSIM3v0SPdpPtr +1) += m * xcsdb;
 
              *(here->BSIM3v0QqPtr +1) += m * omega;

              *(here->BSIM3v0QgPtr +1) -= m * xcqgb;
              *(here->BSIM3v0QdpPtr +1) -= m * xcqdb;
              *(here->BSIM3v0QspPtr +1) -= m * xcqsb;
              *(here->BSIM3v0QbPtr +1) -= m * xcqbb;


              *(here->BSIM3v0DdPtr) += m * gdpr;
              *(here->BSIM3v0SsPtr) += m * gspr;
              *(here->BSIM3v0BbPtr) += m * (gbd + gbs);
              *(here->BSIM3v0DPdpPtr) += m * (gdpr + gds + gbd + RevSum + dxpart*here->BSIM3v0gtd);
              *(here->BSIM3v0SPspPtr) += m * (gspr + gds + gbs + FwdSum + sxpart*here->BSIM3v0gts);
              *(here->BSIM3v0DdpPtr) -= m * gdpr;
              *(here->BSIM3v0SspPtr) -= m * gspr;
              *(here->BSIM3v0BdpPtr) -= m * gbd;
              *(here->BSIM3v0BspPtr) -= m * gbs;
              *(here->BSIM3v0DPdPtr) -= m * gdpr;
              *(here->BSIM3v0DPgPtr) += m * (Gm + dxpart * here->BSIM3v0gtg);
              *(here->BSIM3v0DPbPtr) -= m * (gbd - Gmbs - dxpart * here->BSIM3v0gtb);
              *(here->BSIM3v0DPspPtr) -= m * (gds + FwdSum - dxpart * here->BSIM3v0gts);
              *(here->BSIM3v0SPgPtr) -= m * (Gm - sxpart * here->BSIM3v0gtg);
              *(here->BSIM3v0SPsPtr) -= m * gspr;
              *(here->BSIM3v0SPbPtr) -= m * (gbs + Gmbs - sxpart * here->BSIM3v0gtg);
              *(here->BSIM3v0SPdpPtr) -= m * (gds + RevSum - sxpart * here->BSIM3v0gtd);
              *(here->BSIM3v0GgPtr) -= m * here->BSIM3v0gtg;
              *(here->BSIM3v0GbPtr) -=  m * here->BSIM3v0gtb;
              *(here->BSIM3v0GdpPtr) -= m * here->BSIM3v0gtd;
              *(here->BSIM3v0GspPtr) -= m * here->BSIM3v0gts;

              *(here->BSIM3v0QqPtr) += m * here->BSIM3v0gtau;
 
              *(here->BSIM3v0DPqPtr) += m * (dxpart * here->BSIM3v0gtau);
              *(here->BSIM3v0SPqPtr) += m * (sxpart * here->BSIM3v0gtau);
              *(here->BSIM3v0GqPtr) -=  m * here->BSIM3v0gtau;
 
              *(here->BSIM3v0QgPtr) +=  m * here->BSIM3v0gtg;
              *(here->BSIM3v0QdpPtr) += m * here->BSIM3v0gtd;
              *(here->BSIM3v0QspPtr) += m * here->BSIM3v0gts;
              *(here->BSIM3v0QbPtr) += m * here->BSIM3v0gtb;

        }
    }
    return(OK);
}

