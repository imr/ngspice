/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1acld.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/

/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v1acLoad (GENmodel * inModel, CKTcircuit * ckt)
{
  BSIM3v1model *model = (BSIM3v1model *) inModel;
  BSIM3v1instance *here;
  double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
  double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
  double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
  double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;

  double dxpart, sxpart, cqgb, cqdb, cqsb, cqbb, xcqgb, xcqdb, xcqsb, xcqbb;

  double m;

  omega = ckt->CKTomega;
  for (; model != NULL; model = BSIM3v1nextModel(model))
    {


      for (here = BSIM3v1instances(model); here != NULL;
	   here = BSIM3v1nextInstance(here))
	{
	  if (here->BSIM3v1mode >= 0)
	    {
	      Gm = here->BSIM3v1gm;
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

	      cqgb = here->BSIM3v1cqgb;
	      cqdb = here->BSIM3v1cqdb;
	      cqsb = here->BSIM3v1cqsb;
	      cqbb = here->BSIM3v1cqbb;
	      sxpart = 0.6;
	      dxpart = 0.4;

	    }
	  else
	    {
	      Gm = -here->BSIM3v1gm;
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

	      cqgb = here->BSIM3v1cqgb;
	      cqdb = here->BSIM3v1cqsb;
	      cqsb = here->BSIM3v1cqdb;
	      cqbb = here->BSIM3v1cqbb;
	      sxpart = 0.4;
	      dxpart = 0.6;
	    }

	  gdpr = here->BSIM3v1drainConductance;
	  gspr = here->BSIM3v1sourceConductance;
	  gds = here->BSIM3v1gds;
	  gbd = here->BSIM3v1gbd;
	  gbs = here->BSIM3v1gbs;
	  capbd = here->BSIM3v1capbd;
	  capbs = here->BSIM3v1capbs;

	  GSoverlapCap = here->BSIM3v1cgso;
	  GDoverlapCap = here->BSIM3v1cgdo;
	  GBoverlapCap = here->pParam->BSIM3v1cgbo;

	  xcdgb = (cdgb - GDoverlapCap) * omega;
	  xcddb = (cddb + capbd + GDoverlapCap) * omega;
	  xcdsb = cdsb * omega;
	  xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap) * omega;
	  xcsdb = -(cgdb + cbdb + cddb) * omega;
	  xcssb = (capbs + GSoverlapCap - (cgsb + cbsb + cdsb)) * omega;
	  xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap) * omega;
	  xcgdb = (cgdb - GDoverlapCap) * omega;
	  xcgsb = (cgsb - GSoverlapCap) * omega;
	  xcbgb = (cbgb - GBoverlapCap) * omega;
	  xcbdb = (cbdb - capbd) * omega;
	  xcbsb = (cbsb - capbs) * omega;
	  xcqgb = cqgb * omega;
	  xcqdb = cqdb * omega;
	  xcqsb = cqsb * omega;
	  xcqbb = cqbb * omega;

	  m = here->BSIM3v1m;

	  *(here->BSIM3v1GgPtr + 1) += m * xcggb;
	  *(here->BSIM3v1BbPtr + 1) -= m * (xcbgb + xcbdb + xcbsb);
	  *(here->BSIM3v1DPdpPtr + 1) += m * xcddb;
	  *(here->BSIM3v1SPspPtr + 1) += m * xcssb;
	  *(here->BSIM3v1GbPtr + 1) -= m * (xcggb + xcgdb + xcgsb);
	  *(here->BSIM3v1GdpPtr + 1) += m * xcgdb;
	  *(here->BSIM3v1GspPtr + 1) += m * xcgsb;
	  *(here->BSIM3v1BgPtr + 1) += m * xcbgb;
	  *(here->BSIM3v1BdpPtr + 1) += m * xcbdb;
	  *(here->BSIM3v1BspPtr + 1) += m * xcbsb;
	  *(here->BSIM3v1DPgPtr + 1) += m * xcdgb;
	  *(here->BSIM3v1DPbPtr + 1) -= m * (xcdgb + xcddb + xcdsb);
	  *(here->BSIM3v1DPspPtr + 1) += m * xcdsb;
	  *(here->BSIM3v1SPgPtr + 1) += m * xcsgb;
	  *(here->BSIM3v1SPbPtr + 1) -= m * (xcsgb + xcsdb + xcssb);
	  *(here->BSIM3v1SPdpPtr + 1) += m * xcsdb;

	  *(here->BSIM3v1QqPtr + 1) += m * omega;

	  *(here->BSIM3v1QgPtr + 1) -= m * xcqgb;
	  *(here->BSIM3v1QdpPtr + 1) -= m * xcqdb;
	  *(here->BSIM3v1QspPtr + 1) -= m * xcqsb;
	  *(here->BSIM3v1QbPtr + 1) -= m * xcqbb;


	  *(here->BSIM3v1DdPtr) += m * gdpr;
	  *(here->BSIM3v1SsPtr) += m * gspr;
	  *(here->BSIM3v1BbPtr) += m * (gbd + gbs);
	  *(here->BSIM3v1DPdpPtr) += m * (gdpr + gds + gbd + RevSum + dxpart * here->BSIM3v1gtd);
	  *(here->BSIM3v1SPspPtr) += m * (gspr + gds + gbs + FwdSum + sxpart * here->BSIM3v1gts);
	  *(here->BSIM3v1DdpPtr) -= m * gdpr;
	  *(here->BSIM3v1SspPtr) -= m * gspr;
	  *(here->BSIM3v1BdpPtr) -= m * gbd;
	  *(here->BSIM3v1BspPtr) -= m * gbs;
	  *(here->BSIM3v1DPdPtr) -= m * gdpr;
	  *(here->BSIM3v1DPgPtr) += m * (Gm + dxpart * here->BSIM3v1gtg);
	  *(here->BSIM3v1DPbPtr) -= m * (gbd - Gmbs - dxpart * here->BSIM3v1gtb);
	  *(here->BSIM3v1DPspPtr) -= m * (gds + FwdSum - dxpart * here->BSIM3v1gts);
	  *(here->BSIM3v1SPgPtr) -= m * (Gm - sxpart * here->BSIM3v1gtg);
	  *(here->BSIM3v1SPsPtr) -= m * gspr;
	  *(here->BSIM3v1SPbPtr) -= m * (gbs + Gmbs - sxpart * here->BSIM3v1gtg);
	  *(here->BSIM3v1SPdpPtr) -= m * (gds + RevSum - sxpart * here->BSIM3v1gtd);
	  *(here->BSIM3v1GgPtr) -= m * here->BSIM3v1gtg;
	  *(here->BSIM3v1GbPtr) -= m * here->BSIM3v1gtb;
	  *(here->BSIM3v1GdpPtr) -= m * here->BSIM3v1gtd;
	  *(here->BSIM3v1GspPtr) -= m * here->BSIM3v1gts;

	  *(here->BSIM3v1QqPtr) += m * here->BSIM3v1gtau;

	  *(here->BSIM3v1DPqPtr) += m * dxpart * here->BSIM3v1gtau;
	  *(here->BSIM3v1SPqPtr) += m * sxpart * here->BSIM3v1gtau;
	  *(here->BSIM3v1GqPtr) -= m * here->BSIM3v1gtau;

	  *(here->BSIM3v1QgPtr) += m * here->BSIM3v1gtg;
	  *(here->BSIM3v1QdpPtr) += m * here->BSIM3v1gtd;
	  *(here->BSIM3v1QspPtr) += m * here->BSIM3v1gts;
	  *(here->BSIM3v1QbPtr) += m * here->BSIM3v1gtb;

	}
    }
  return (OK);
}
