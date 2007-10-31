/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim4v4def.h"
#include "suffix.h"

int
BSIM4V4pzLoad(inModel,ckt,s)
GENmodel *inModel;
CKTcircuit *ckt;
SPcomplex *s;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gds, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double xcdbdb=0.0, xcsbsb=0.0, xcgmgmb=0.0, xcgmdb=0.0, xcgmsb=0.0, xcdgmb=0.0, xcsgmb=0.0;
double xcgmbb=0.0, xcbgmb=0.0;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb=0.0, xcqdb=0.0, xcqsb=0.0, xcqbb=0.0;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T0=0.0, T1, CoxWL, qcheq, Cdg, Cdd, Cds, Csg, Csd, Css;
double ScalingFactor = 1.0e-9;
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = model->BSIM4V4nextModel) 
    {    for (here = model->BSIM4V4instances; here!= NULL;
              here = here->BSIM4V4nextInstance) 
	       {    if (here->BSIM4V4owner != ARCHme) continue;
	            pParam = here->pParam;
              capbd = here->BSIM4V4capbd;
              capbs = here->BSIM4V4capbs;
              cgso = here->BSIM4V4cgso;
              cgdo = here->BSIM4V4cgdo;
              cgbo = pParam->BSIM4V4cgbo;

              if (here->BSIM4V4mode >= 0) 
              {   Gm = here->BSIM4V4gm;
                  Gmbs = here->BSIM4V4gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4V4gbds);
                  gbbsp = here->BSIM4V4gbds + here->BSIM4V4gbgs + here->BSIM4V4gbbs;
                  gbdpg = here->BSIM4V4gbgs;
                  gbdpdp = here->BSIM4V4gbds;
                  gbdpb = here->BSIM4V4gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4V4igcMod)
                  {   gIstotg = here->BSIM4V4gIgsg + here->BSIM4V4gIgcsg;
                      gIstotd = here->BSIM4V4gIgcsd;
                      gIstots = here->BSIM4V4gIgss + here->BSIM4V4gIgcss;
                      gIstotb = here->BSIM4V4gIgcsb;

                      gIdtotg = here->BSIM4V4gIgdg + here->BSIM4V4gIgcdg;
                      gIdtotd = here->BSIM4V4gIgdd + here->BSIM4V4gIgcdd;
                      gIdtots = here->BSIM4V4gIgcds;
                      gIdtotb = here->BSIM4V4gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4V4igbMod)
                  {   gIbtotg = here->BSIM4V4gIgbg;
                      gIbtotd = here->BSIM4V4gIgbd;
                      gIbtots = here->BSIM4V4gIgbs;
                      gIbtotb = here->BSIM4V4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4V4igcMod != 0) || (model->BSIM4V4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4V4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4V4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4V4vgs);
                  else if (here->BSIM4V4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4V4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4V4vgs);
                  if (here->BSIM4V4rgateMod > 1)
                  {   gcrgd = here->BSIM4V4gcrgd * T0;
                      gcrgg = here->BSIM4V4gcrgg * T0;
                      gcrgs = here->BSIM4V4gcrgs * T0;
                      gcrgb = here->BSIM4V4gcrgb * T0;
                      gcrgg -= here->BSIM4V4gcrg;
                      gcrg = here->BSIM4V4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4V4acnqsMod == 0)
                  {   if (here->BSIM4V4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4V4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4V4cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4V4cggb;
                          xcgdb = here->BSIM4V4cgdb;
                          xcgsb = here->BSIM4V4cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4V4cdgb;
                          xcsgb = -(here->BSIM4V4cggb + here->BSIM4V4cbgb
                                + here->BSIM4V4cdgb);
                          xcbgb = here->BSIM4V4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4V4cggb + cgdo + cgso
                                + pParam->BSIM4V4cgbo;
                          xcgdb = here->BSIM4V4cgdb - cgdo;
                          xcgsb = here->BSIM4V4cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4V4cdgb - cgdo;
                          xcsgb = -(here->BSIM4V4cggb + here->BSIM4V4cbgb
                                + here->BSIM4V4cdgb + cgso);
                          xcbgb = here->BSIM4V4cbgb - pParam->BSIM4V4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4V4cddb + here->BSIM4V4capbd + cgdo;
                      xcdsb = here->BSIM4V4cdsb;

                      xcsdb = -(here->BSIM4V4cgdb + here->BSIM4V4cbdb
                            + here->BSIM4V4cddb);
                      xcssb = here->BSIM4V4capbs + cgso - (here->BSIM4V4cgsb
                            + here->BSIM4V4cbsb + here->BSIM4V4cdsb);

                      if (!here->BSIM4V4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4V4cbdb - here->BSIM4V4capbd;
                          xcbsb = here->BSIM4V4cbsb - here->BSIM4V4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4V4cddb + here->BSIM4V4cdgb
                                 + here->BSIM4V4cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4V4capbs;
                          xcbdb = here->BSIM4V4cbdb;
                          xcbsb = here->BSIM4V4cbsb;

                          xcdbdb = -here->BSIM4V4capbd;
                          xcsbsb = -here->BSIM4V4capbs;
                      }
                      xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.6;
                      dxpart = 0.4;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   xcggb = xcgdb = xcgsb = xcgbb = 0.0;
                      xcbgb = xcbdb = xcbsb = xcbbb = 0.0;
                      xcdgb = xcddb = xcdsb = xcdbb = 0.0;
                      xcsgb = xcsdb = xcssb = xcsbb = 0.0;

		      xgtg = here->BSIM4V4gtg;
                      xgtd = here->BSIM4V4gtd;
                      xgts = here->BSIM4V4gts;
                      xgtb = here->BSIM4V4gtb;

                      xcqgb = here->BSIM4V4cqgb;
                      xcqdb = here->BSIM4V4cqdb;
                      xcqsb = here->BSIM4V4cqsb;
                      xcqbb = here->BSIM4V4cqbb;

		      CoxWL = model->BSIM4V4coxe * here->pParam->BSIM4V4weffCV
                            * here->BSIM4V4nf * here->pParam->BSIM4V4leffCV;
		      qcheq = -(here->BSIM4V4qgate + here->BSIM4V4qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4V4xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4V4xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4V4qdrn / qcheq;
		          Cdd = here->BSIM4V4cddb;
		          Csd = -(here->BSIM4V4cgdb + here->BSIM4V4cddb
			      + here->BSIM4V4cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4V4cdgb;
		          Csg = -(here->BSIM4V4cggb + here->BSIM4V4cdgb
			      + here->BSIM4V4cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4V4cdsb;
		          Css = -(here->BSIM4V4cgsb + here->BSIM4V4cdsb
			      + here->BSIM4V4cbsb);
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
              {   Gm = -here->BSIM4V4gm;
                  Gmbs = -here->BSIM4V4gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4V4gbds);
                  gbbdp = here->BSIM4V4gbds + here->BSIM4V4gbgs + here->BSIM4V4gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4V4gbgs;
                  gbspsp = here->BSIM4V4gbds;
                  gbspb = here->BSIM4V4gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4V4igcMod)
                  {   gIstotg = here->BSIM4V4gIgsg + here->BSIM4V4gIgcdg;
                      gIstotd = here->BSIM4V4gIgcds;
                      gIstots = here->BSIM4V4gIgss + here->BSIM4V4gIgcdd;
                      gIstotb = here->BSIM4V4gIgcdb;

                      gIdtotg = here->BSIM4V4gIgdg + here->BSIM4V4gIgcsg;
                      gIdtotd = here->BSIM4V4gIgdd + here->BSIM4V4gIgcss;
                      gIdtots = here->BSIM4V4gIgcsd;
                      gIdtotb = here->BSIM4V4gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4V4igbMod)
                  {   gIbtotg = here->BSIM4V4gIgbg;
                      gIbtotd = here->BSIM4V4gIgbs;
                      gIbtots = here->BSIM4V4gIgbd;
                      gIbtotb = here->BSIM4V4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4V4igcMod != 0) || (model->BSIM4V4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4V4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4V4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4V4vgs);
                  else if (here->BSIM4V4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4V4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4V4vgs);
                  if (here->BSIM4V4rgateMod > 1)
                  {   gcrgd = here->BSIM4V4gcrgs * T0;
                      gcrgg = here->BSIM4V4gcrgg * T0;
                      gcrgs = here->BSIM4V4gcrgd * T0;
                      gcrgb = here->BSIM4V4gcrgb * T0;
                      gcrgg -= here->BSIM4V4gcrg;
                      gcrg = here->BSIM4V4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4V4acnqsMod == 0)
                  {   if (here->BSIM4V4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4V4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4V4cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4V4cggb;
                          xcgdb = here->BSIM4V4cgsb;
                          xcgsb = here->BSIM4V4cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4V4cggb + here->BSIM4V4cbgb
                                + here->BSIM4V4cdgb);
                          xcsgb = here->BSIM4V4cdgb;
                          xcbgb = here->BSIM4V4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4V4cggb + cgdo + cgso
                                + pParam->BSIM4V4cgbo;
                          xcgdb = here->BSIM4V4cgsb - cgdo;
                          xcgsb = here->BSIM4V4cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4V4cggb + here->BSIM4V4cbgb
                                + here->BSIM4V4cdgb + cgdo);
                          xcsgb = here->BSIM4V4cdgb - cgso;
                          xcbgb = here->BSIM4V4cbgb - pParam->BSIM4V4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4V4capbd + cgdo - (here->BSIM4V4cgsb
                            + here->BSIM4V4cbsb + here->BSIM4V4cdsb);
                      xcdsb = -(here->BSIM4V4cgdb + here->BSIM4V4cbdb
                            + here->BSIM4V4cddb);

                      xcsdb = here->BSIM4V4cdsb;
                      xcssb = here->BSIM4V4cddb + here->BSIM4V4capbs + cgso;

                      if (!here->BSIM4V4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4V4cbsb - here->BSIM4V4capbd;
                          xcbsb = here->BSIM4V4cbdb - here->BSIM4V4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4V4capbd;
                          xcsbb = -(here->BSIM4V4cddb + here->BSIM4V4cdgb
                                + here->BSIM4V4cdsb);
                          xcbdb = here->BSIM4V4cbsb;
                          xcbsb = here->BSIM4V4cbdb;
                          xcdbdb = -here->BSIM4V4capbd;
                          xcsbsb = -here->BSIM4V4capbs;
                      }
                      xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.4;
                      dxpart = 0.6;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   xcggb = xcgdb = xcgsb = xcgbb = 0.0;
                      xcbgb = xcbdb = xcbsb = xcbbb = 0.0;
                      xcdgb = xcddb = xcdsb = xcdbb = 0.0;
                      xcsgb = xcsdb = xcssb = xcsbb = 0.0;

		      xgtg = here->BSIM4V4gtg;
                      xgtd = here->BSIM4V4gts;
                      xgts = here->BSIM4V4gtd;
                      xgtb = here->BSIM4V4gtb;

                      xcqgb = here->BSIM4V4cqgb;
                      xcqdb = here->BSIM4V4cqsb;
                      xcqsb = here->BSIM4V4cqdb;
                      xcqbb = here->BSIM4V4cqbb;

		      CoxWL = model->BSIM4V4coxe * here->pParam->BSIM4V4weffCV
                            * here->BSIM4V4nf * here->pParam->BSIM4V4leffCV;
		      qcheq = -(here->BSIM4V4qgate + here->BSIM4V4qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4V4xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4V4xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4V4qdrn / qcheq;
		          Css = here->BSIM4V4cddb;
		          Cds = -(here->BSIM4V4cgdb + here->BSIM4V4cddb
			      + here->BSIM4V4cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4V4cdgb;
		          Cdg = -(here->BSIM4V4cggb + here->BSIM4V4cdgb
			      + here->BSIM4V4cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4V4cdsb;
		          Cdd = -(here->BSIM4V4cgsb + here->BSIM4V4cdsb
			      + here->BSIM4V4cbsb);
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

              if (model->BSIM4V4rdsMod == 1)
              {   gstot = here->BSIM4V4gstot;
                  gstotd = here->BSIM4V4gstotd;
                  gstotg = here->BSIM4V4gstotg;
                  gstots = here->BSIM4V4gstots - gstot;
                  gstotb = here->BSIM4V4gstotb;

                  gdtot = here->BSIM4V4gdtot;
                  gdtotd = here->BSIM4V4gdtotd - gdtot;
                  gdtotg = here->BSIM4V4gdtotg;
                  gdtots = here->BSIM4V4gdtots;
                  gdtotb = here->BSIM4V4gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4V4qdef) * here->BSIM4V4gtau;
              gds = here->BSIM4V4gds;

              /*
               * Loading PZ matrix
               */

   	          m = here->BSIM4V4m;

              if (!model->BSIM4V4rdsMod)
              {   gdpr = here->BSIM4V4drainConductance;
                  gspr = here->BSIM4V4sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4V4rbodyMod)
              {   gjbd = here->BSIM4V4gbd;
                  gjbs = here->BSIM4V4gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4V4grgeltd;

              if (here->BSIM4V4rgateMod == 1)
              {   *(here->BSIM4V4GEgePtr) += m * geltd;
                  *(here->BSIM4V4GPgePtr) -= m * geltd;
                  *(here->BSIM4V4GEgpPtr) -= m * geltd;

                  *(here->BSIM4V4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4V4GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4V4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4V4GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4V4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4V4GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4V4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4V4GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4V4rgateMod == 2)
              {   *(here->BSIM4V4GEgePtr) += m * gcrg;
                  *(here->BSIM4V4GEgpPtr) += m * gcrgg;
                  *(here->BSIM4V4GEdpPtr) += m * gcrgd;
                  *(here->BSIM4V4GEspPtr) += m * gcrgs;
                  *(here->BSIM4V4GEbpPtr) += m * gcrgb;

                  *(here->BSIM4V4GPgePtr) -= m * gcrg;
                  *(here->BSIM4V4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4V4GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4V4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4V4GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4V4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4V4GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4V4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4V4GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4V4rgateMod == 3)
              {   *(here->BSIM4V4GEgePtr) += m * geltd;
                  *(here->BSIM4V4GEgmPtr) -= m * geltd;
                  *(here->BSIM4V4GMgePtr) -= m * geltd;
                  *(here->BSIM4V4GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4V4GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4V4GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4V4GMdpPtr) += m * gcrgd;
                  *(here->BSIM4V4GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4V4GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4V4GMgpPtr) += m * gcrgg;
                  *(here->BSIM4V4GMspPtr) += m * gcrgs;
                  *(here->BSIM4V4GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4V4GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4V4GMbpPtr) += m * gcrgb;
                  *(here->BSIM4V4GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4V4GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4V4DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4V4DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4V4GPgmPtr) -= m * gcrg;
                  *(here->BSIM4V4SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4V4SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4V4BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4V4BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4V4GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4V4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4V4GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4V4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4V4GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4V4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4V4GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4V4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4V4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4V4GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4V4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4V4GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4V4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4V4GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4V4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4V4GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4V4rdsMod)
              {   (*(here->BSIM4V4DgpPtr) += m * gdtotg);
                  (*(here->BSIM4V4DspPtr) += m * gdtots);
                  (*(here->BSIM4V4DbpPtr) += m * gdtotb);
                  (*(here->BSIM4V4SdpPtr) += m * gstotd);
                  (*(here->BSIM4V4SgpPtr) += m * gstotg);
                  (*(here->BSIM4V4SbpPtr) += m * gstotb);
              }

              *(here->BSIM4V4DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4V4DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4V4DPdpPtr) += m * (gdpr + gds + here->BSIM4V4gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4V4DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4V4DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4V4DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4V4DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4V4DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4V4DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4V4DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4V4DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4V4DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4V4DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4V4DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4V4DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4V4SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4V4SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4V4SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4V4SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4V4SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4V4SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4V4SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4V4SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4V4SPspPtr) += m * (gspr + gds + here->BSIM4V4gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4V4SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4V4SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4V4SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4V4SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4V4SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4V4SsPtr) += m * (gspr + gstot);

              *(here->BSIM4V4BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4V4BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4V4BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4V4BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4V4BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4V4BPgpPtr) -= m * (here->BSIM4V4gbgs + gIbtotg);
              *(here->BSIM4V4BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4V4BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4V4BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4V4BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4V4BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4V4BPbpPtr) += m * (gjbd + gjbs - here->BSIM4V4gbbs
				     - gIbtotb);
           ggidld = here->BSIM4V4ggidld;
           ggidlg = here->BSIM4V4ggidlg;
           ggidlb = here->BSIM4V4ggidlb;
           ggislg = here->BSIM4V4ggislg;
           ggisls = here->BSIM4V4ggisls;
           ggislb = here->BSIM4V4ggislb;

           /* stamp gidl */
           (*(here->BSIM4V4DPdpPtr) += m * ggidld);
           (*(here->BSIM4V4DPgpPtr) += m * ggidlg);
           (*(here->BSIM4V4DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4V4DPbpPtr) += m * ggidlb);
           (*(here->BSIM4V4BPdpPtr) -= m * ggidld);
           (*(here->BSIM4V4BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4V4BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4V4BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4V4SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4V4SPgpPtr) += m * ggislg);
           (*(here->BSIM4V4SPspPtr) += m * ggisls);
           (*(here->BSIM4V4SPbpPtr) += m * ggislb);
           (*(here->BSIM4V4BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4V4BPgpPtr) -= m * ggislg);
           (*(here->BSIM4V4BPspPtr) -= m * ggisls);
           (*(here->BSIM4V4BPbpPtr) -= m * ggislb);

              if (here->BSIM4V4rbodyMod)
              {   (*(here->BSIM4V4DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4V4DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4V4DPdbPtr) -= m * here->BSIM4V4gbd);
                  (*(here->BSIM4V4SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4V4SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4V4SPsbPtr) -= m * here->BSIM4V4gbs);

                  (*(here->BSIM4V4DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4V4DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4V4DBdpPtr) -= m * here->BSIM4V4gbd);
                  (*(here->BSIM4V4DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4V4DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4V4DBdbPtr) += m * (here->BSIM4V4gbd + here->BSIM4V4grbpd
                                          + here->BSIM4V4grbdb));
                  (*(here->BSIM4V4DBbpPtr) -= m * here->BSIM4V4grbpd);
                  (*(here->BSIM4V4DBbPtr) -= m * here->BSIM4V4grbdb);

                  (*(here->BSIM4V4BPdbPtr) -= m * here->BSIM4V4grbpd);
                  (*(here->BSIM4V4BPbPtr) -= m * here->BSIM4V4grbpb);
                  (*(here->BSIM4V4BPsbPtr) -= m * here->BSIM4V4grbps);
                  (*(here->BSIM4V4BPbpPtr) += m * (here->BSIM4V4grbpd + here->BSIM4V4grbps
					  + here->BSIM4V4grbpb));
                  /* WDL: (-here->BSIM4V4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4V4SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4V4SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4V4SBspPtr) -= m * here->BSIM4V4gbs);
                  (*(here->BSIM4V4SBbpPtr) -= m * here->BSIM4V4grbps);
                  (*(here->BSIM4V4SBbPtr) -= m * here->BSIM4V4grbsb);
                  (*(here->BSIM4V4SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4V4SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4V4SBsbPtr) += m * (here->BSIM4V4gbs
					  + here->BSIM4V4grbps + here->BSIM4V4grbsb));

                  (*(here->BSIM4V4BdbPtr) -= m * here->BSIM4V4grbdb);
                  (*(here->BSIM4V4BbpPtr) -= m * here->BSIM4V4grbpb);
                  (*(here->BSIM4V4BsbPtr) -= m * here->BSIM4V4grbsb);
                  (*(here->BSIM4V4BbPtr) += m * (here->BSIM4V4grbsb + here->BSIM4V4grbdb
                                        + here->BSIM4V4grbpb));
              }

              if (here->BSIM4V4acnqsMod)
              {   *(here->BSIM4V4QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4V4QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4V4QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4V4QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4V4QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4V4QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4V4QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4V4QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4V4QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4V4QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4V4GPqPtr) -= m * here->BSIM4V4gtau;
                  *(here->BSIM4V4DPqPtr) += m * dxpart * here->BSIM4V4gtau;
                  *(here->BSIM4V4SPqPtr) += m * sxpart * here->BSIM4V4gtau;

                  *(here->BSIM4V4QqPtr) += m * here->BSIM4V4gtau;
                  *(here->BSIM4V4QgpPtr) += m * xgtg;
                  *(here->BSIM4V4QdpPtr) += m * xgtd;
                  *(here->BSIM4V4QbpPtr) += m * xgtb;
                  *(here->BSIM4V4QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
