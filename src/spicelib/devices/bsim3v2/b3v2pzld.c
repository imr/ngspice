/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3v2pzld.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim3v2def.h"
#include "suffix.h"

int
BSIM3V2pzLoad(inModel,ckt,s)
GENmodel *inModel;
CKTcircuit *ckt;
SPcomplex *s;
{
BSIM3V2model *model = (BSIM3V2model*)inModel;
BSIM3V2instance *here;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb, xcqdb, xcqsb, xcqbb;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T1, CoxWL, qcheq, Cdg, Cdd, Cds, Cdb, Csg, Csd, Css, Csb;

    for (; model != NULL; model = model->BSIM3V2nextModel) 
    {    for (here = model->BSIM3V2instances; here!= NULL;
              here = here->BSIM3V2nextInstance) 
	 {    
              if (here->BSIM3V2owner != ARCHme) continue;
              if (here->BSIM3V2mode >= 0) 
              {   Gm = here->BSIM3V2gm;
                  Gmbs = here->BSIM3V2gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -here->BSIM3V2gbds;
                  gbbsp = here->BSIM3V2gbds + here->BSIM3V2gbgs + here->BSIM3V2gbbs;

                  gbdpg = here->BSIM3V2gbgs;
                  gbdpdp = here->BSIM3V2gbds;
                  gbdpb = here->BSIM3V2gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspg = 0.0;
                  gbspdp = 0.0;
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

                      xcqgb = here->BSIM3V2cqgb;
                      xcqdb = here->BSIM3V2cqdb;
                      xcqsb = here->BSIM3V2cqsb;
                      xcqbb = here->BSIM3V2cqbb;

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

                      xcqgb = here->BSIM3V2cqgb;
                      xcqdb = here->BSIM3V2cqsb;
                      xcqsb = here->BSIM3V2cqdb;
                      xcqbb = here->BSIM3V2cqbb;

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

              *(here->BSIM3V2GgPtr ) += xcggb * s->real;
              *(here->BSIM3V2GgPtr +1) += xcggb * s->imag;
              *(here->BSIM3V2BbPtr ) += xcbbb * s->real;
              *(here->BSIM3V2BbPtr +1) += xcbbb * s->imag;
              *(here->BSIM3V2DPdpPtr ) += xcddb * s->real;
              *(here->BSIM3V2DPdpPtr +1) += xcddb * s->imag;
              *(here->BSIM3V2SPspPtr ) += xcssb * s->real;
              *(here->BSIM3V2SPspPtr +1) += xcssb * s->imag;

              *(here->BSIM3V2GbPtr ) += xcgbb * s->real;
              *(here->BSIM3V2GbPtr +1) += xcgbb * s->imag;
              *(here->BSIM3V2GdpPtr ) += xcgdb * s->real;
              *(here->BSIM3V2GdpPtr +1) += xcgdb * s->imag;
              *(here->BSIM3V2GspPtr ) += xcgsb * s->real;
              *(here->BSIM3V2GspPtr +1) += xcgsb * s->imag;

              *(here->BSIM3V2BgPtr ) += xcbgb * s->real;
              *(here->BSIM3V2BgPtr +1) += xcbgb * s->imag;
              *(here->BSIM3V2BdpPtr ) += xcbdb * s->real;
              *(here->BSIM3V2BdpPtr +1) += xcbdb * s->imag;
              *(here->BSIM3V2BspPtr ) += xcbsb * s->real;
              *(here->BSIM3V2BspPtr +1) += xcbsb * s->imag;

              *(here->BSIM3V2DPgPtr ) += xcdgb * s->real;
              *(here->BSIM3V2DPgPtr +1) += xcdgb * s->imag;
              *(here->BSIM3V2DPbPtr ) += xcdbb * s->real;
              *(here->BSIM3V2DPbPtr +1) += xcdbb * s->imag;
              *(here->BSIM3V2DPspPtr ) += xcdsb * s->real;
              *(here->BSIM3V2DPspPtr +1) += xcdsb * s->imag;

              *(here->BSIM3V2SPgPtr ) += xcsgb * s->real;
              *(here->BSIM3V2SPgPtr +1) += xcsgb * s->imag;
              *(here->BSIM3V2SPbPtr ) += xcsbb * s->real;
              *(here->BSIM3V2SPbPtr +1) += xcsbb * s->imag;
              *(here->BSIM3V2SPdpPtr ) += xcsdb * s->real;
              *(here->BSIM3V2SPdpPtr +1) += xcsdb * s->imag;

              *(here->BSIM3V2DdPtr) += gdpr;
              *(here->BSIM3V2DdpPtr) -= gdpr;
              *(here->BSIM3V2DPdPtr) -= gdpr;

              *(here->BSIM3V2SsPtr) += gspr;
              *(here->BSIM3V2SspPtr) -= gspr;
              *(here->BSIM3V2SPsPtr) -= gspr;

              *(here->BSIM3V2BgPtr) -= here->BSIM3V2gbgs;
              *(here->BSIM3V2BbPtr) += gbd + gbs - here->BSIM3V2gbbs;
              *(here->BSIM3V2BdpPtr) -= gbd - gbbdp;
              *(here->BSIM3V2BspPtr) -= gbs - gbbsp;

              *(here->BSIM3V2DPgPtr) += Gm + dxpart * xgtg 
				   + T1 * ddxpart_dVg + gbdpg;
              *(here->BSIM3V2DPdpPtr) += gdpr + gds + gbd + RevSum
                                    + dxpart * xgtd + T1 * ddxpart_dVd + gbdpdp;
              *(here->BSIM3V2DPspPtr) -= gds + FwdSum - dxpart * xgts
				    - T1 * ddxpart_dVs - gbdpsp;
              *(here->BSIM3V2DPbPtr) -= gbd - Gmbs - dxpart * xgtb
				   - T1 * ddxpart_dVb - gbdpb;

              *(here->BSIM3V2SPgPtr) -= Gm - sxpart * xgtg
				   - T1 * dsxpart_dVg - gbspg;
              *(here->BSIM3V2SPspPtr) += gspr + gds + gbs + FwdSum
                                   + sxpart * xgts + T1 * dsxpart_dVs + gbspsp;
              *(here->BSIM3V2SPbPtr) -= gbs + Gmbs - sxpart * xgtb
				   - T1 * dsxpart_dVb - gbspb;
              *(here->BSIM3V2SPdpPtr) -= gds + RevSum - sxpart * xgtd
				    - T1 * dsxpart_dVd - gbspdp;

              *(here->BSIM3V2GgPtr) -= xgtg;
              *(here->BSIM3V2GbPtr) -= xgtb;
              *(here->BSIM3V2GdpPtr) -= xgtd;
              *(here->BSIM3V2GspPtr) -= xgts;

              if (here->BSIM3V2nqsMod)
              {   *(here->BSIM3V2QqPtr ) += s->real;
                  *(here->BSIM3V2QqPtr +1) += s->imag;
                  *(here->BSIM3V2QgPtr ) -= xcqgb * s->real;
                  *(here->BSIM3V2QgPtr +1) -= xcqgb * s->imag;
                  *(here->BSIM3V2QdpPtr ) -= xcqdb * s->real;
                  *(here->BSIM3V2QdpPtr +1) -= xcqdb * s->imag;
                  *(here->BSIM3V2QbPtr ) -= xcqbb * s->real;
                  *(here->BSIM3V2QbPtr +1) -= xcqbb * s->imag;
                  *(here->BSIM3V2QspPtr ) -= xcqsb * s->real;
                  *(here->BSIM3V2QspPtr +1) -= xcqsb * s->imag;

                  *(here->BSIM3V2GqPtr) -= here->BSIM3V2gtau;
                  *(here->BSIM3V2DPqPtr) += dxpart * here->BSIM3V2gtau;
                  *(here->BSIM3V2SPqPtr) += sxpart * here->BSIM3V2gtau;

                  *(here->BSIM3V2QqPtr) += here->BSIM3V2gtau;
                  *(here->BSIM3V2QgPtr) += xgtg;
                  *(here->BSIM3V2QdpPtr) += xgtd;
                  *(here->BSIM3V2QbPtr) += xgtb;
                  *(here->BSIM3V2QspPtr) += xgts;
              }
         }
    }
    return(OK);
}

