/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1aacld.c
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1AacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v1Amodel *model = (BSIM3v1Amodel*)inModel;
BSIM3v1Ainstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;

double dxpart, sxpart, cqgb, cqdb, cqsb, cqbb, xcqgb, xcqdb, xcqsb, xcqbb;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM3v1AnextModel) 
    {    


      for (here = model->BSIM3v1Ainstances; here!= NULL;
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

                  cqgb = here->BSIM3v1Acqgb;
                  cqdb = here->BSIM3v1Acqdb;
                  cqsb = here->BSIM3v1Acqsb;
                  cqbb = here->BSIM3v1Acqbb;
                  sxpart = 0.6;
                  dxpart = 0.4;

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

                  cqgb = here->BSIM3v1Acqgb;
                  cqdb = here->BSIM3v1Acqsb;
                  cqsb = here->BSIM3v1Acqdb;
                  cqbb = here->BSIM3v1Acqbb;
                  sxpart = 0.4;
                  dxpart = 0.6;
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

              m = here->BSIM3v1Am;

              *(here->BSIM3v1AGgPtr +1) += m * xcggb;
              *(here->BSIM3v1ABbPtr +1) -= m * (xcbgb + xcbdb + xcbsb);
              *(here->BSIM3v1ADPdpPtr +1) += m * xcddb;
              *(here->BSIM3v1ASPspPtr +1) += m * xcssb;
              *(here->BSIM3v1AGbPtr +1) -= m * (xcggb + xcgdb + xcgsb);
              *(here->BSIM3v1AGdpPtr +1) += m * xcgdb;
              *(here->BSIM3v1AGspPtr +1) += m * xcgsb;
              *(here->BSIM3v1ABgPtr +1) += m * xcbgb;
              *(here->BSIM3v1ABdpPtr +1) += m * xcbdb;
              *(here->BSIM3v1ABspPtr +1) += m * xcbsb;
              *(here->BSIM3v1ADPgPtr +1) += m * xcdgb;
              *(here->BSIM3v1ADPbPtr +1) -= m * (xcdgb + xcddb + xcdsb);
              *(here->BSIM3v1ADPspPtr +1) += m * xcdsb;
              *(here->BSIM3v1ASPgPtr +1) += m * xcsgb;
              *(here->BSIM3v1ASPbPtr +1) -= m * (xcsgb + xcsdb + xcssb);
              *(here->BSIM3v1ASPdpPtr +1) += m * xcsdb;
 
              *(here->BSIM3v1AQqPtr +1) += m * omega;

              *(here->BSIM3v1AQgPtr +1) -= m * xcqgb;
              *(here->BSIM3v1AQdpPtr +1) -= m * xcqdb;
              *(here->BSIM3v1AQspPtr +1) -= m * xcqsb;
              *(here->BSIM3v1AQbPtr +1) -= m * xcqbb;

              *(here->BSIM3v1ADdPtr) += m * gdpr;
              *(here->BSIM3v1ASsPtr) += m * gspr;
              *(here->BSIM3v1ABbPtr) += m * (gbd + gbs);
              *(here->BSIM3v1ADPdpPtr) += m*(gdpr+gds+gbd+RevSum+dxpart*here->BSIM3v1Agtd);
              *(here->BSIM3v1ASPspPtr) += m*(gspr+gds+gbs+FwdSum+sxpart*here->BSIM3v1Agts);
              *(here->BSIM3v1ADdpPtr) -= m * gdpr;
              *(here->BSIM3v1ASspPtr) -= m * gspr;
              *(here->BSIM3v1ABdpPtr) -= m * gbd;
              *(here->BSIM3v1ABspPtr) -= m * gbs;
              *(here->BSIM3v1ADPdPtr) -= m * gdpr;
              *(here->BSIM3v1ADPgPtr) += m * (Gm + dxpart * here->BSIM3v1Agtg);
              *(here->BSIM3v1ADPbPtr) -= m * (gbd-Gmbs - dxpart * here->BSIM3v1Agtb);
              *(here->BSIM3v1ADPspPtr) -= m * (gds+FwdSum-dxpart * here->BSIM3v1Agts);
              *(here->BSIM3v1ASPgPtr) -= m * (Gm - sxpart * here->BSIM3v1Agtg);
              *(here->BSIM3v1ASPsPtr) -= m * gspr;
              *(here->BSIM3v1ASPbPtr) -= m * (gbs+Gmbs - sxpart * here->BSIM3v1Agtg);
              *(here->BSIM3v1ASPdpPtr) -= m * (gds+RevSum-sxpart * here->BSIM3v1Agtd);
              *(here->BSIM3v1AGgPtr) -= m * here->BSIM3v1Agtg;
              *(here->BSIM3v1AGbPtr) -=  m * here->BSIM3v1Agtb;
              *(here->BSIM3v1AGdpPtr) -= m * here->BSIM3v1Agtd;
              *(here->BSIM3v1AGspPtr) -= m * here->BSIM3v1Agts;

              *(here->BSIM3v1AQqPtr) += m * here->BSIM3v1Agtau;
 
              *(here->BSIM3v1ADPqPtr) += m * dxpart * here->BSIM3v1Agtau;
              *(here->BSIM3v1ASPqPtr) += m * sxpart * here->BSIM3v1Agtau;
              *(here->BSIM3v1AGqPtr) -=  m * here->BSIM3v1Agtau;
 
              *(here->BSIM3v1AQgPtr) +=  m * here->BSIM3v1Agtg;
              *(here->BSIM3v1AQdpPtr) += m * here->BSIM3v1Agtd;
              *(here->BSIM3v1AQspPtr) += m * here->BSIM3v1Agts;
              *(here->BSIM3v1AQbPtr) += m * here->BSIM3v1Agtb;

        }
    }
    return(OK);
}

