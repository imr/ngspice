/* $Id$  */
/* 
$Log$
Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
Imported sources

 * Revision 3.1  96/12/08  19:51:00  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1acld.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V1acLoad(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM3V1model *model = (BSIM3V1model*)inModel;
register BSIM3V1instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;

double dxpart, sxpart, cqgb, cqdb, cqsb, cqbb, xcqgb, xcqdb, xcqsb, xcqbb;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM3V1nextModel) 
    {    


      for (here = model->BSIM3V1instances; here!= NULL;
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

                  cqgb = here->BSIM3V1cqgb;
                  cqdb = here->BSIM3V1cqdb;
                  cqsb = here->BSIM3V1cqsb;
                  cqbb = here->BSIM3V1cqbb;
                  sxpart = 0.6;
                  dxpart = 0.4;

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

                  cqgb = here->BSIM3V1cqgb;
                  cqdb = here->BSIM3V1cqsb;
                  cqsb = here->BSIM3V1cqdb;
                  cqbb = here->BSIM3V1cqbb;
                  sxpart = 0.4;
                  dxpart = 0.6;
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

              *(here->BSIM3V1GgPtr +1) += xcggb;
              *(here->BSIM3V1BbPtr +1) -= xcbgb + xcbdb + xcbsb;
              *(here->BSIM3V1DPdpPtr +1) += xcddb;
              *(here->BSIM3V1SPspPtr +1) += xcssb;
              *(here->BSIM3V1GbPtr +1) -= xcggb + xcgdb + xcgsb;
              *(here->BSIM3V1GdpPtr +1) += xcgdb;
              *(here->BSIM3V1GspPtr +1) += xcgsb;
              *(here->BSIM3V1BgPtr +1) += xcbgb;
              *(here->BSIM3V1BdpPtr +1) += xcbdb;
              *(here->BSIM3V1BspPtr +1) += xcbsb;
              *(here->BSIM3V1DPgPtr +1) += xcdgb;
              *(here->BSIM3V1DPbPtr +1) -= xcdgb + xcddb + xcdsb;
              *(here->BSIM3V1DPspPtr +1) += xcdsb;
              *(here->BSIM3V1SPgPtr +1) += xcsgb;
              *(here->BSIM3V1SPbPtr +1) -= xcsgb + xcsdb + xcssb;
              *(here->BSIM3V1SPdpPtr +1) += xcsdb;
 
              *(here->BSIM3V1QqPtr +1) += omega;

              *(here->BSIM3V1QgPtr +1) -= xcqgb;
              *(here->BSIM3V1QdpPtr +1) -= xcqdb;
              *(here->BSIM3V1QspPtr +1) -= xcqsb;
              *(here->BSIM3V1QbPtr +1) -= xcqbb;


              *(here->BSIM3V1DdPtr) += gdpr;
              *(here->BSIM3V1SsPtr) += gspr;
              *(here->BSIM3V1BbPtr) += gbd + gbs;
              *(here->BSIM3V1DPdpPtr) += gdpr + gds + gbd + RevSum + dxpart*here->BSIM3V1gtd;
              *(here->BSIM3V1SPspPtr) += gspr + gds + gbs + FwdSum + sxpart*here->BSIM3V1gts;
              *(here->BSIM3V1DdpPtr) -= gdpr;
              *(here->BSIM3V1SspPtr) -= gspr;
              *(here->BSIM3V1BdpPtr) -= gbd;
              *(here->BSIM3V1BspPtr) -= gbs;
              *(here->BSIM3V1DPdPtr) -= gdpr;
              *(here->BSIM3V1DPgPtr) += Gm + dxpart * here->BSIM3V1gtg;
              *(here->BSIM3V1DPbPtr) -= gbd - Gmbs - dxpart * here->BSIM3V1gtb;
              *(here->BSIM3V1DPspPtr) -= gds + FwdSum - dxpart * here->BSIM3V1gts;
              *(here->BSIM3V1SPgPtr) -= Gm - sxpart * here->BSIM3V1gtg;
              *(here->BSIM3V1SPsPtr) -= gspr;
              *(here->BSIM3V1SPbPtr) -= gbs + Gmbs - sxpart * here->BSIM3V1gtg;
              *(here->BSIM3V1SPdpPtr) -= gds + RevSum - sxpart * here->BSIM3V1gtd;
              *(here->BSIM3V1GgPtr) -= here->BSIM3V1gtg;
              *(here->BSIM3V1GbPtr) -=  here->BSIM3V1gtb;
              *(here->BSIM3V1GdpPtr) -= here->BSIM3V1gtd;
              *(here->BSIM3V1GspPtr) -= here->BSIM3V1gts;

              *(here->BSIM3V1QqPtr) += here->BSIM3V1gtau;
 
              *(here->BSIM3V1DPqPtr) += dxpart * here->BSIM3V1gtau;
              *(here->BSIM3V1SPqPtr) += sxpart * here->BSIM3V1gtau;
              *(here->BSIM3V1GqPtr) -=  here->BSIM3V1gtau;
 
              *(here->BSIM3V1QgPtr) +=  here->BSIM3V1gtg;
              *(here->BSIM3V1QdpPtr) += here->BSIM3V1gtd;
              *(here->BSIM3V1QspPtr) += here->BSIM3V1gts;
              *(here->BSIM3V1QbPtr) += here->BSIM3V1gtb;

        }
    }
    return(OK);
}

