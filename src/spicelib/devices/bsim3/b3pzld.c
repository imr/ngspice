/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3pzld.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim3def.h"
#include "ngspice/suffix.h"

int
BSIM3pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb=0.0, xcqdb=0.0, xcqsb=0.0, xcqbb=0.0;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T1, CoxWL, qcheq, Cdg, Cdd, Cds, Csg, Csd, Css;
double ScalingFactor = 1.0e-9;
double m;

    for (; model != NULL; model = BSIM3nextModel(model)) 
    {    for (here = BSIM3instances(model); here!= NULL;
              here = BSIM3nextInstance(here)) 
	 {
	      if (here->BSIM3mode >= 0) 
              {   Gm = here->BSIM3gm;
                  Gmbs = here->BSIM3gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -here->BSIM3gbds;
                  gbbsp = here->BSIM3gbds + here->BSIM3gbgs + here->BSIM3gbbs;

                  gbdpg = here->BSIM3gbgs;
                  gbdpdp = here->BSIM3gbds;
                  gbdpb = here->BSIM3gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspg = 0.0;
                  gbspdp = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (here->BSIM3nqsMod == 0 && here->BSIM3acnqsMod == 0)
                  {   cggb = here->BSIM3cggb;
                      cgsb = here->BSIM3cgsb;
                      cgdb = here->BSIM3cgdb;

                      cbgb = here->BSIM3cbgb;
                      cbsb = here->BSIM3cbsb;
                      cbdb = here->BSIM3cbdb;

                      cdgb = here->BSIM3cdgb;
                      cdsb = here->BSIM3cdsb;
                      cddb = here->BSIM3cddb;

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.6;
                      dxpart = 0.4;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   cggb = cgdb = cgsb = 0.0;
                      cbgb = cbdb = cbsb = 0.0;
                      cdgb = cddb = cdsb = 0.0;

		      xgtg = here->BSIM3gtg;
                      xgtd = here->BSIM3gtd;
                      xgts = here->BSIM3gts;
                      xgtb = here->BSIM3gtb;

                      xcqgb = here->BSIM3cqgb;
                      xcqdb = here->BSIM3cqdb;
                      xcqsb = here->BSIM3cqsb;
                      xcqbb = here->BSIM3cqbb;

		      CoxWL = model->BSIM3cox * here->pParam->BSIM3weffCV
                            * here->pParam->BSIM3leffCV;
		      qcheq = -(here->BSIM3qgate + here->BSIM3qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM3xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM3xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM3qdrn / qcheq;
		          Cdd = here->BSIM3cddb;
		          Csd = -(here->BSIM3cgdb + here->BSIM3cddb
			      + here->BSIM3cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM3cdgb;
		          Csg = -(here->BSIM3cggb + here->BSIM3cdgb
			      + here->BSIM3cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM3cdsb;
		          Css = -(here->BSIM3cgsb + here->BSIM3cdsb
			      + here->BSIM3cbsb);
		          ddxpart_dVs = (Cds - dxpart * (Cds + Css)) / qcheq;

		          ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg 
				      + ddxpart_dVs);
		      }
		      sxpart = 1.0 - dxpart;
		      dsxpart_dVd = -ddxpart_dVd;
		      dsxpart_dVg = -ddxpart_dVg;
		      dsxpart_dVs = -ddxpart_dVs;
		      dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                  }
              }
              else
              {   Gm = -here->BSIM3gm;
                  Gmbs = -here->BSIM3gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -here->BSIM3gbds;
                  gbbdp = here->BSIM3gbds + here->BSIM3gbgs + here->BSIM3gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM3gbgs;
                  gbspsp = here->BSIM3gbds;
                  gbspb = here->BSIM3gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

		  if (here->BSIM3nqsMod == 0 && here->BSIM3acnqsMod == 0)
                  {   cggb = here->BSIM3cggb;
                      cgsb = here->BSIM3cgdb;
                      cgdb = here->BSIM3cgsb;

                      cbgb = here->BSIM3cbgb;
                      cbsb = here->BSIM3cbdb;
                      cbdb = here->BSIM3cbsb;

                      cdgb = -(here->BSIM3cdgb + cggb + cbgb);
                      cdsb = -(here->BSIM3cddb + cgsb + cbsb);
                      cddb = -(here->BSIM3cdsb + cgdb + cbdb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.4;
                      dxpart = 0.6;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   cggb = cgdb = cgsb = 0.0;
                      cbgb = cbdb = cbsb = 0.0;
                      cdgb = cddb = cdsb = 0.0;

		      xgtg = here->BSIM3gtg;
                      xgtd = here->BSIM3gts;
                      xgts = here->BSIM3gtd;
                      xgtb = here->BSIM3gtb;

                      xcqgb = here->BSIM3cqgb;
                      xcqdb = here->BSIM3cqsb;
                      xcqsb = here->BSIM3cqdb;
                      xcqbb = here->BSIM3cqbb;

		      CoxWL = model->BSIM3cox * here->pParam->BSIM3weffCV
                            * here->pParam->BSIM3leffCV;
		      qcheq = -(here->BSIM3qgate + here->BSIM3qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM3xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM3xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM3qdrn / qcheq;
		          Css = here->BSIM3cddb;
		          Cds = -(here->BSIM3cgdb + here->BSIM3cddb
			      + here->BSIM3cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM3cdgb;
		          Cdg = -(here->BSIM3cggb + here->BSIM3cdgb
			      + here->BSIM3cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM3cdsb;
		          Cdd = -(here->BSIM3cgsb + here->BSIM3cdsb
			      + here->BSIM3cbsb);
		          dsxpart_dVd = (Csd - sxpart * (Csd + Cdd)) / qcheq;

		          dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg 
				      + dsxpart_dVs);
		      }
		      dxpart = 1.0 - sxpart;
		      ddxpart_dVd = -dsxpart_dVd;
		      ddxpart_dVg = -dsxpart_dVg;
		      ddxpart_dVs = -dsxpart_dVs;
		      ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                  }
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM3qdef) * here->BSIM3gtau;
              gdpr = here->BSIM3drainConductance;
              gspr = here->BSIM3sourceConductance;
              gds = here->BSIM3gds;
              gbd = here->BSIM3gbd;
              gbs = here->BSIM3gbs;
              capbd = here->BSIM3capbd;
              capbs = here->BSIM3capbs;

	      GSoverlapCap = here->BSIM3cgso;
	      GDoverlapCap = here->BSIM3cgdo;
	      GBoverlapCap = here->pParam->BSIM3cgbo;

              xcdgb = (cdgb - GDoverlapCap);
              xcddb = (cddb + capbd + GDoverlapCap);
              xcdsb = cdsb;
              xcdbb = -(xcdgb + xcddb + xcdsb);
              xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap);
              xcsdb = -(cgdb + cbdb + cddb);
              xcssb = (capbs + GSoverlapCap - (cgsb + cbsb + cdsb));
              xcsbb = -(xcsgb + xcsdb + xcssb); 
              xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap);
              xcgdb = (cgdb - GDoverlapCap);
              xcgsb = (cgsb - GSoverlapCap);
              xcgbb = -(xcggb + xcgdb + xcgsb);
              xcbgb = (cbgb - GBoverlapCap);
              xcbdb = (cbdb - capbd);
              xcbsb = (cbsb - capbs);
              xcbbb = -(xcbgb + xcbdb + xcbsb);

	      m = here->BSIM3m;

              *(here->BSIM3GgPtr ) += m * (xcggb * s->real);
              *(here->BSIM3GgPtr +1) += m * (xcggb * s->imag);
              *(here->BSIM3BbPtr ) += m * (xcbbb * s->real);
              *(here->BSIM3BbPtr +1) += m * (xcbbb * s->imag);
              *(here->BSIM3DPdpPtr ) += m * (xcddb * s->real);
              *(here->BSIM3DPdpPtr +1) += m * (xcddb * s->imag);
              *(here->BSIM3SPspPtr ) += m * (xcssb * s->real);
              *(here->BSIM3SPspPtr +1) += m * (xcssb * s->imag);

              *(here->BSIM3GbPtr ) += m * (xcgbb * s->real);
              *(here->BSIM3GbPtr +1) += m * (xcgbb * s->imag);
              *(here->BSIM3GdpPtr ) += m * (xcgdb * s->real);
              *(here->BSIM3GdpPtr +1) += m * (xcgdb * s->imag);
              *(here->BSIM3GspPtr ) += m * (xcgsb * s->real);
              *(here->BSIM3GspPtr +1) += m * (xcgsb * s->imag);

              *(here->BSIM3BgPtr ) += m * (xcbgb * s->real);
              *(here->BSIM3BgPtr +1) += m * (xcbgb * s->imag);
              *(here->BSIM3BdpPtr ) += m * (xcbdb * s->real);
              *(here->BSIM3BdpPtr +1) += m * (xcbdb * s->imag);
              *(here->BSIM3BspPtr ) += m * (xcbsb * s->real);
              *(here->BSIM3BspPtr +1) += m * (xcbsb * s->imag);

              *(here->BSIM3DPgPtr ) += m * (xcdgb * s->real);
              *(here->BSIM3DPgPtr +1) += m * (xcdgb * s->imag);
              *(here->BSIM3DPbPtr ) += m * (xcdbb * s->real);
              *(here->BSIM3DPbPtr +1) += m * (xcdbb * s->imag);
              *(here->BSIM3DPspPtr ) += m * (xcdsb * s->real);
              *(here->BSIM3DPspPtr +1) += m * (xcdsb * s->imag);

              *(here->BSIM3SPgPtr ) += m * (xcsgb * s->real);
              *(here->BSIM3SPgPtr +1) += m * (xcsgb * s->imag);
              *(here->BSIM3SPbPtr ) += m * (xcsbb * s->real);
              *(here->BSIM3SPbPtr +1) += m * (xcsbb * s->imag);
              *(here->BSIM3SPdpPtr ) += m * (xcsdb * s->real);
              *(here->BSIM3SPdpPtr +1) += m * (xcsdb * s->imag);

              *(here->BSIM3DdPtr) += m * gdpr;
              *(here->BSIM3DdpPtr) -= m * gdpr;
              *(here->BSIM3DPdPtr) -= m * gdpr;

              *(here->BSIM3SsPtr) += m * gspr;
              *(here->BSIM3SspPtr) -= m * gspr;
              *(here->BSIM3SPsPtr) -= m * gspr;

              *(here->BSIM3BgPtr) -= m * here->BSIM3gbgs;
              *(here->BSIM3BbPtr) += m * (gbd + gbs - here->BSIM3gbbs);
              *(here->BSIM3BdpPtr) -= m * (gbd - gbbdp);
              *(here->BSIM3BspPtr) -= m * (gbs - gbbsp);

              *(here->BSIM3DPgPtr) += m * (Gm + dxpart * xgtg 
				   + T1 * ddxpart_dVg + gbdpg);
              *(here->BSIM3DPdpPtr) += m * (gdpr + gds + gbd + RevSum
                                    + dxpart * xgtd + T1 * ddxpart_dVd + gbdpdp);
              *(here->BSIM3DPspPtr) -= m * (gds + FwdSum - dxpart * xgts
				    - T1 * ddxpart_dVs - gbdpsp);
              *(here->BSIM3DPbPtr) -= m * (gbd - Gmbs - dxpart * xgtb
				   - T1 * ddxpart_dVb - gbdpb);

              *(here->BSIM3SPgPtr) -= m * (Gm - sxpart * xgtg
				   - T1 * dsxpart_dVg - gbspg);
              *(here->BSIM3SPspPtr) += m * (gspr + gds + gbs + FwdSum
                                   + sxpart * xgts + T1 * dsxpart_dVs + gbspsp);
              *(here->BSIM3SPbPtr) -= m * (gbs + Gmbs - sxpart * xgtb
				   - T1 * dsxpart_dVb - gbspb);
              *(here->BSIM3SPdpPtr) -= m * (gds + RevSum - sxpart * xgtd
				    - T1 * dsxpart_dVd - gbspdp);

              *(here->BSIM3GgPtr) -= m * xgtg;
              *(here->BSIM3GbPtr) -= m * xgtb;
              *(here->BSIM3GdpPtr) -= m * xgtd;
              *(here->BSIM3GspPtr) -= m * xgts;

              if (here->BSIM3nqsMod || here->BSIM3acnqsMod)
              {   *(here->BSIM3QqPtr ) += m * (s->real * ScalingFactor);
                  *(here->BSIM3QqPtr +1) += m * (s->imag * ScalingFactor);
                  *(here->BSIM3QgPtr ) -= m * (xcqgb * s->real);
                  *(here->BSIM3QgPtr +1) -= m * (xcqgb * s->imag);
                  *(here->BSIM3QdpPtr ) -= m * (xcqdb * s->real);
                  *(here->BSIM3QdpPtr +1) -= m * (xcqdb * s->imag);
                  *(here->BSIM3QbPtr ) -= m * (xcqbb * s->real);
                  *(here->BSIM3QbPtr +1) -= m * (xcqbb * s->imag);
                  *(here->BSIM3QspPtr ) -= m * (xcqsb * s->real);
                  *(here->BSIM3QspPtr +1) -= m * (xcqsb * s->imag);

                  *(here->BSIM3GqPtr) -= m * (here->BSIM3gtau);
                  *(here->BSIM3DPqPtr) += m * (dxpart * here->BSIM3gtau);
                  *(here->BSIM3SPqPtr) += m * (sxpart * here->BSIM3gtau);

                  *(here->BSIM3QqPtr) += m * (here->BSIM3gtau);
                  *(here->BSIM3QgPtr) += m * xgtg;
                  *(here->BSIM3QdpPtr) += m * xgtd;
                  *(here->BSIM3QbPtr) += m * xgtb;
                  *(here->BSIM3QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}

