/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3acld.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V2acLoad(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM3V2model *model = (BSIM3V2model*)inModel;
register BSIM3V2instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb, xcqdb, xcqsb, xcqbb;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T1, CoxWL, qcheq, Cdg, Cdd, Cds, Cdb, Csg, Csd, Css, Csb;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM3V2nextModel) 
    {    for (here = model->BSIM3V2instances; here!= NULL;
              here = here->BSIM3V2nextInstance) 

	 {   if (here->BSIM3V2owner != ARCHme) continue; 
              if (here->BSIM3V2mode >= 0) 
	      {   Gm = here->BSIM3V2gm;
                  Gmbs = here->BSIM3V2gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -here->BSIM3V2gbds;
                  gbbsp = here->BSIM3V2gbds + here->BSIM3V2gbgs + here->BSIM3V2gbbs;

                  gbdpg = here->BSIM3V2gbgs;
                  gbdpb = here->BSIM3V2gbbs;
                  gbdpdp = here->BSIM3V2gbds;
                  gbdpsp = -(gbdpg + gbdpb + gbdpdp);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

		  if (here->BSIM3V2nqsMod == 0)
                  {   cggb = here->BSIM3V2cggb;
                      cgsb = here->BSIM3V2cgsb;
                      cgdb = here->BSIM3V2cgdb;

                      cbgb = here->BSIM3V2cbgb;
                      cbsb = here->BSIM3V2cbsb;
                      cbdb = here->BSIM3V2cbdb;

                      cdgb = here->BSIM3V2cdgb;
                      cdsb = here->BSIM3V2cdsb;
                      cddb = here->BSIM3V2cddb;

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

		      xgtg = here->BSIM3V2gtg;
                      xgtd = here->BSIM3V2gtd;
                      xgts = here->BSIM3V2gts;
                      xgtb = here->BSIM3V2gtb; 
 
                      xcqgb = here->BSIM3V2cqgb * omega;
                      xcqdb = here->BSIM3V2cqdb * omega;
                      xcqsb = here->BSIM3V2cqsb * omega;
                      xcqbb = here->BSIM3V2cqbb * omega;

		      CoxWL = model->BSIM3V2cox * here->pParam->BSIM3V2weffCV
                            * here->pParam->BSIM3V2leffCV;
		      qcheq = -(here->BSIM3V2qgate + here->BSIM3V2qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM3V2xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM3V2xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM3V2qdrn / qcheq;
		          Cdd = here->BSIM3V2cddb;
		          Csd = -(here->BSIM3V2cgdb + here->BSIM3V2cddb
			      + here->BSIM3V2cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM3V2cdgb;
		          Csg = -(here->BSIM3V2cggb + here->BSIM3V2cdgb
			      + here->BSIM3V2cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM3V2cdsb;
		          Css = -(here->BSIM3V2cgsb + here->BSIM3V2cdsb
			      + here->BSIM3V2cbsb);
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
              {   Gm = -here->BSIM3V2gm;
                  Gmbs = -here->BSIM3V2gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -here->BSIM3V2gbds;
                  gbbdp = here->BSIM3V2gbds + here->BSIM3V2gbgs + here->BSIM3V2gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM3V2gbgs;
                  gbspsp = here->BSIM3V2gbds;
                  gbspb = here->BSIM3V2gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

		  if (here->BSIM3V2nqsMod == 0)
                  {   cggb = here->BSIM3V2cggb;
                      cgsb = here->BSIM3V2cgdb;
                      cgdb = here->BSIM3V2cgsb;

                      cbgb = here->BSIM3V2cbgb;
                      cbsb = here->BSIM3V2cbdb;
                      cbdb = here->BSIM3V2cbsb;

                      cdgb = -(here->BSIM3V2cdgb + cggb + cbgb);
                      cdsb = -(here->BSIM3V2cddb + cgsb + cbsb);
                      cddb = -(here->BSIM3V2cdsb + cgdb + cbdb);

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

		      xgtg = here->BSIM3V2gtg;
                      xgtd = here->BSIM3V2gts;
                      xgts = here->BSIM3V2gtd;
                      xgtb = here->BSIM3V2gtb;

                      xcqgb = here->BSIM3V2cqgb * omega;
                      xcqdb = here->BSIM3V2cqsb * omega;
                      xcqsb = here->BSIM3V2cqdb * omega;
                      xcqbb = here->BSIM3V2cqbb * omega;

		      CoxWL = model->BSIM3V2cox * here->pParam->BSIM3V2weffCV
                            * here->pParam->BSIM3V2leffCV;
		      qcheq = -(here->BSIM3V2qgate + here->BSIM3V2qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM3V2xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM3V2xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM3V2qdrn / qcheq;
		          Css = here->BSIM3V2cddb;
		          Cds = -(here->BSIM3V2cgdb + here->BSIM3V2cddb
			      + here->BSIM3V2cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM3V2cdgb;
		          Cdg = -(here->BSIM3V2cggb + here->BSIM3V2cdgb
			      + here->BSIM3V2cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM3V2cdsb;
		          Cdd = -(here->BSIM3V2cgsb + here->BSIM3V2cdsb
			      + here->BSIM3V2cbsb);
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

	      T1 = *(ckt->CKTstate0 + here->BSIM3V2qdef) * here->BSIM3V2gtau;
              gdpr = here->BSIM3V2drainConductance;
              gspr = here->BSIM3V2sourceConductance;
              gds = here->BSIM3V2gds;
              gbd = here->BSIM3V2gbd;
              gbs = here->BSIM3V2gbs;
              capbd = here->BSIM3V2capbd;
              capbs = here->BSIM3V2capbs;

	      GSoverlapCap = here->BSIM3V2cgso;
	      GDoverlapCap = here->BSIM3V2cgdo;
	      GBoverlapCap = here->pParam->BSIM3V2cgbo;

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

              *(here->BSIM3V2GgPtr +1) += xcggb;
              *(here->BSIM3V2BbPtr +1) -= xcbgb + xcbdb + xcbsb;
              *(here->BSIM3V2DPdpPtr +1) += xcddb;
              *(here->BSIM3V2SPspPtr +1) += xcssb;
              *(here->BSIM3V2GbPtr +1) -= xcggb + xcgdb + xcgsb;
              *(here->BSIM3V2GdpPtr +1) += xcgdb;
              *(here->BSIM3V2GspPtr +1) += xcgsb;
              *(here->BSIM3V2BgPtr +1) += xcbgb;
              *(here->BSIM3V2BdpPtr +1) += xcbdb;
              *(here->BSIM3V2BspPtr +1) += xcbsb;
              *(here->BSIM3V2DPgPtr +1) += xcdgb;
              *(here->BSIM3V2DPbPtr +1) -= xcdgb + xcddb + xcdsb;
              *(here->BSIM3V2DPspPtr +1) += xcdsb;
              *(here->BSIM3V2SPgPtr +1) += xcsgb;
              *(here->BSIM3V2SPbPtr +1) -= xcsgb + xcsdb + xcssb;
              *(here->BSIM3V2SPdpPtr +1) += xcsdb;

              *(here->BSIM3V2DdPtr) += gdpr;
              *(here->BSIM3V2SsPtr) += gspr;
              *(here->BSIM3V2BbPtr) += gbd + gbs - here->BSIM3V2gbbs;
              *(here->BSIM3V2DPdpPtr) += gdpr + gds + gbd + RevSum 
                                    + dxpart * xgtd + T1 * ddxpart_dVd + gbdpdp;
              *(here->BSIM3V2SPspPtr) += gspr + gds + gbs + FwdSum 
                                    + sxpart * xgts + T1 * dsxpart_dVs + gbspsp;

              *(here->BSIM3V2DdpPtr) -= gdpr;
              *(here->BSIM3V2SspPtr) -= gspr;

              *(here->BSIM3V2BgPtr) -= here->BSIM3V2gbgs;
              *(here->BSIM3V2BdpPtr) -= gbd - gbbdp;
              *(here->BSIM3V2BspPtr) -= gbs - gbbsp;

              *(here->BSIM3V2DPdPtr) -= gdpr;
              *(here->BSIM3V2DPgPtr) += Gm + dxpart * xgtg + T1 * ddxpart_dVg
				   + gbdpg;
              *(here->BSIM3V2DPbPtr) -= gbd - Gmbs - dxpart * xgtb
				   - T1 * ddxpart_dVb - gbdpb;
              *(here->BSIM3V2DPspPtr) -= gds + FwdSum - dxpart * xgts 
				    - T1 * ddxpart_dVs - gbdpsp;

              *(here->BSIM3V2SPgPtr) -= Gm - sxpart * xgtg - T1 * dsxpart_dVg
				   - gbspg;
              *(here->BSIM3V2SPsPtr) -= gspr;
              *(here->BSIM3V2SPbPtr) -= gbs + Gmbs - sxpart * xgtb
				   - T1 * dsxpart_dVb - gbspb;
              *(here->BSIM3V2SPdpPtr) -= gds + RevSum - sxpart * xgtd 
				    - T1 * dsxpart_dVd - gbspdp;

              *(here->BSIM3V2GgPtr) -= xgtg;
              *(here->BSIM3V2GbPtr) -= xgtb;
              *(here->BSIM3V2GdpPtr) -= xgtd;
              *(here->BSIM3V2GspPtr) -= xgts;

              if (here->BSIM3V2nqsMod)
              {   *(here->BSIM3V2QqPtr +1) += omega;
                  *(here->BSIM3V2QgPtr +1) -= xcqgb;
                  *(here->BSIM3V2QdpPtr +1) -= xcqdb;
                  *(here->BSIM3V2QspPtr +1) -= xcqsb;
                  *(here->BSIM3V2QbPtr +1) -= xcqbb;

                  *(here->BSIM3V2QqPtr) += here->BSIM3V2gtau;

                  *(here->BSIM3V2DPqPtr) += dxpart * here->BSIM3V2gtau;
                  *(here->BSIM3V2SPqPtr) += sxpart * here->BSIM3V2gtau;
                  *(here->BSIM3V2GqPtr) -=  here->BSIM3V2gtau;

                  *(here->BSIM3V2QgPtr) +=  xgtg;
                  *(here->BSIM3V2QdpPtr) += xgtd;
                  *(here->BSIM3V2QspPtr) += xgts;
                  *(here->BSIM3V2QbPtr) += xgtb;
              }
        }
    }
    return(OK);
}

