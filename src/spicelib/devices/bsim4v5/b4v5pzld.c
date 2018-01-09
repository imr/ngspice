/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim4v5def.h"
#include "ngspice/suffix.h"

int
BSIM4v5pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;

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
struct bsim4v5SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = BSIM4v5nextModel(model)) 
    {    for (here = BSIM4v5instances(model); here!= NULL;
              here = BSIM4v5nextInstance(here)) 
	       {
	            pParam = here->pParam;
              capbd = here->BSIM4v5capbd;
              capbs = here->BSIM4v5capbs;
              cgso = here->BSIM4v5cgso;
              cgdo = here->BSIM4v5cgdo;
              cgbo = pParam->BSIM4v5cgbo;

              if (here->BSIM4v5mode >= 0) 
              {   Gm = here->BSIM4v5gm;
                  Gmbs = here->BSIM4v5gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v5gbds);
                  gbbsp = here->BSIM4v5gbds + here->BSIM4v5gbgs + here->BSIM4v5gbbs;
                  gbdpg = here->BSIM4v5gbgs;
                  gbdpdp = here->BSIM4v5gbds;
                  gbdpb = here->BSIM4v5gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v5igcMod)
                  {   gIstotg = here->BSIM4v5gIgsg + here->BSIM4v5gIgcsg;
                      gIstotd = here->BSIM4v5gIgcsd;
                      gIstots = here->BSIM4v5gIgss + here->BSIM4v5gIgcss;
                      gIstotb = here->BSIM4v5gIgcsb;

                      gIdtotg = here->BSIM4v5gIgdg + here->BSIM4v5gIgcdg;
                      gIdtotd = here->BSIM4v5gIgdd + here->BSIM4v5gIgcdd;
                      gIdtots = here->BSIM4v5gIgcds;
                      gIdtotb = here->BSIM4v5gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v5igbMod)
                  {   gIbtotg = here->BSIM4v5gIgbg;
                      gIbtotd = here->BSIM4v5gIgbd;
                      gIbtots = here->BSIM4v5gIgbs;
                      gIbtotb = here->BSIM4v5gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v5igcMod != 0) || (model->BSIM4v5igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v5rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v5vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v5vgs);
                  else if (here->BSIM4v5rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v5vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v5vgs);
                  if (here->BSIM4v5rgateMod > 1)
                  {   gcrgd = here->BSIM4v5gcrgd * T0;
                      gcrgg = here->BSIM4v5gcrgg * T0;
                      gcrgs = here->BSIM4v5gcrgs * T0;
                      gcrgb = here->BSIM4v5gcrgb * T0;
                      gcrgg -= here->BSIM4v5gcrg;
                      gcrg = here->BSIM4v5gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v5acnqsMod == 0)
                  {   if (here->BSIM4v5rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v5cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v5cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v5cggb;
                          xcgdb = here->BSIM4v5cgdb;
                          xcgsb = here->BSIM4v5cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v5cdgb;
                          xcsgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                                + here->BSIM4v5cdgb);
                          xcbgb = here->BSIM4v5cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v5cggb + cgdo + cgso
                                + pParam->BSIM4v5cgbo;
                          xcgdb = here->BSIM4v5cgdb - cgdo;
                          xcgsb = here->BSIM4v5cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v5cdgb - cgdo;
                          xcsgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                                + here->BSIM4v5cdgb + cgso);
                          xcbgb = here->BSIM4v5cbgb - pParam->BSIM4v5cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v5cddb + here->BSIM4v5capbd + cgdo;
                      xcdsb = here->BSIM4v5cdsb;

                      xcsdb = -(here->BSIM4v5cgdb + here->BSIM4v5cbdb
                            + here->BSIM4v5cddb);
                      xcssb = here->BSIM4v5capbs + cgso - (here->BSIM4v5cgsb
                            + here->BSIM4v5cbsb + here->BSIM4v5cdsb);

                      if (!here->BSIM4v5rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v5cbdb - here->BSIM4v5capbd;
                          xcbsb = here->BSIM4v5cbsb - here->BSIM4v5capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v5cddb + here->BSIM4v5cdgb
                                 + here->BSIM4v5cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v5capbs;
                          xcbdb = here->BSIM4v5cbdb;
                          xcbsb = here->BSIM4v5cbsb;

                          xcdbdb = -here->BSIM4v5capbd;
                          xcsbsb = -here->BSIM4v5capbs;
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

		      xgtg = here->BSIM4v5gtg;
                      xgtd = here->BSIM4v5gtd;
                      xgts = here->BSIM4v5gts;
                      xgtb = here->BSIM4v5gtb;

                      xcqgb = here->BSIM4v5cqgb;
                      xcqdb = here->BSIM4v5cqdb;
                      xcqsb = here->BSIM4v5cqsb;
                      xcqbb = here->BSIM4v5cqbb;

		      CoxWL = model->BSIM4v5coxe * here->pParam->BSIM4v5weffCV
                            * here->BSIM4v5nf * here->pParam->BSIM4v5leffCV;
		      qcheq = -(here->BSIM4v5qgate + here->BSIM4v5qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v5xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v5xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v5qdrn / qcheq;
		          Cdd = here->BSIM4v5cddb;
		          Csd = -(here->BSIM4v5cgdb + here->BSIM4v5cddb
			      + here->BSIM4v5cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v5cdgb;
		          Csg = -(here->BSIM4v5cggb + here->BSIM4v5cdgb
			      + here->BSIM4v5cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v5cdsb;
		          Css = -(here->BSIM4v5cgsb + here->BSIM4v5cdsb
			      + here->BSIM4v5cbsb);
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
              {   Gm = -here->BSIM4v5gm;
                  Gmbs = -here->BSIM4v5gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v5gbds);
                  gbbdp = here->BSIM4v5gbds + here->BSIM4v5gbgs + here->BSIM4v5gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v5gbgs;
                  gbspsp = here->BSIM4v5gbds;
                  gbspb = here->BSIM4v5gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4v5igcMod)
                  {   gIstotg = here->BSIM4v5gIgsg + here->BSIM4v5gIgcdg;
                      gIstotd = here->BSIM4v5gIgcds;
                      gIstots = here->BSIM4v5gIgss + here->BSIM4v5gIgcdd;
                      gIstotb = here->BSIM4v5gIgcdb;

                      gIdtotg = here->BSIM4v5gIgdg + here->BSIM4v5gIgcsg;
                      gIdtotd = here->BSIM4v5gIgdd + here->BSIM4v5gIgcss;
                      gIdtots = here->BSIM4v5gIgcsd;
                      gIdtotb = here->BSIM4v5gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v5igbMod)
                  {   gIbtotg = here->BSIM4v5gIgbg;
                      gIbtotd = here->BSIM4v5gIgbs;
                      gIbtots = here->BSIM4v5gIgbd;
                      gIbtotb = here->BSIM4v5gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v5igcMod != 0) || (model->BSIM4v5igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v5rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v5vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v5vgs);
                  else if (here->BSIM4v5rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v5vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v5vgs);
                  if (here->BSIM4v5rgateMod > 1)
                  {   gcrgd = here->BSIM4v5gcrgs * T0;
                      gcrgg = here->BSIM4v5gcrgg * T0;
                      gcrgs = here->BSIM4v5gcrgd * T0;
                      gcrgb = here->BSIM4v5gcrgb * T0;
                      gcrgg -= here->BSIM4v5gcrg;
                      gcrg = here->BSIM4v5gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v5acnqsMod == 0)
                  {   if (here->BSIM4v5rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v5cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v5cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v5cggb;
                          xcgdb = here->BSIM4v5cgsb;
                          xcgsb = here->BSIM4v5cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                                + here->BSIM4v5cdgb);
                          xcsgb = here->BSIM4v5cdgb;
                          xcbgb = here->BSIM4v5cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v5cggb + cgdo + cgso
                                + pParam->BSIM4v5cgbo;
                          xcgdb = here->BSIM4v5cgsb - cgdo;
                          xcgsb = here->BSIM4v5cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v5cggb + here->BSIM4v5cbgb
                                + here->BSIM4v5cdgb + cgdo);
                          xcsgb = here->BSIM4v5cdgb - cgso;
                          xcbgb = here->BSIM4v5cbgb - pParam->BSIM4v5cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v5capbd + cgdo - (here->BSIM4v5cgsb
                            + here->BSIM4v5cbsb + here->BSIM4v5cdsb);
                      xcdsb = -(here->BSIM4v5cgdb + here->BSIM4v5cbdb
                            + here->BSIM4v5cddb);

                      xcsdb = here->BSIM4v5cdsb;
                      xcssb = here->BSIM4v5cddb + here->BSIM4v5capbs + cgso;

                      if (!here->BSIM4v5rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v5cbsb - here->BSIM4v5capbd;
                          xcbsb = here->BSIM4v5cbdb - here->BSIM4v5capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v5capbd;
                          xcsbb = -(here->BSIM4v5cddb + here->BSIM4v5cdgb
                                + here->BSIM4v5cdsb);
                          xcbdb = here->BSIM4v5cbsb;
                          xcbsb = here->BSIM4v5cbdb;
                          xcdbdb = -here->BSIM4v5capbd;
                          xcsbsb = -here->BSIM4v5capbs;
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

		      xgtg = here->BSIM4v5gtg;
                      xgtd = here->BSIM4v5gts;
                      xgts = here->BSIM4v5gtd;
                      xgtb = here->BSIM4v5gtb;

                      xcqgb = here->BSIM4v5cqgb;
                      xcqdb = here->BSIM4v5cqsb;
                      xcqsb = here->BSIM4v5cqdb;
                      xcqbb = here->BSIM4v5cqbb;

		      CoxWL = model->BSIM4v5coxe * here->pParam->BSIM4v5weffCV
                            * here->BSIM4v5nf * here->pParam->BSIM4v5leffCV;
		      qcheq = -(here->BSIM4v5qgate + here->BSIM4v5qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v5xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v5xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v5qdrn / qcheq;
		          Css = here->BSIM4v5cddb;
		          Cds = -(here->BSIM4v5cgdb + here->BSIM4v5cddb
			      + here->BSIM4v5cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v5cdgb;
		          Cdg = -(here->BSIM4v5cggb + here->BSIM4v5cdgb
			      + here->BSIM4v5cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v5cdsb;
		          Cdd = -(here->BSIM4v5cgsb + here->BSIM4v5cdsb
			      + here->BSIM4v5cbsb);
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

              if (model->BSIM4v5rdsMod == 1)
              {   gstot = here->BSIM4v5gstot;
                  gstotd = here->BSIM4v5gstotd;
                  gstotg = here->BSIM4v5gstotg;
                  gstots = here->BSIM4v5gstots - gstot;
                  gstotb = here->BSIM4v5gstotb;

                  gdtot = here->BSIM4v5gdtot;
                  gdtotd = here->BSIM4v5gdtotd - gdtot;
                  gdtotg = here->BSIM4v5gdtotg;
                  gdtots = here->BSIM4v5gdtots;
                  gdtotb = here->BSIM4v5gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v5qdef) * here->BSIM4v5gtau;
              gds = here->BSIM4v5gds;

              /*
               * Loading PZ matrix
               */

   	          m = here->BSIM4v5m;

              if (!model->BSIM4v5rdsMod)
              {   gdpr = here->BSIM4v5drainConductance;
                  gspr = here->BSIM4v5sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v5rbodyMod)
              {   gjbd = here->BSIM4v5gbd;
                  gjbs = here->BSIM4v5gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v5grgeltd;

              if (here->BSIM4v5rgateMod == 1)
              {   *(here->BSIM4v5GEgePtr) += m * geltd;
                  *(here->BSIM4v5GPgePtr) -= m * geltd;
                  *(here->BSIM4v5GEgpPtr) -= m * geltd;

                  *(here->BSIM4v5GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v5GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4v5GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v5GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v5GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v5GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v5GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v5GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4v5rgateMod == 2)
              {   *(here->BSIM4v5GEgePtr) += m * gcrg;
                  *(here->BSIM4v5GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v5GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v5GEspPtr) += m * gcrgs;
                  *(here->BSIM4v5GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v5GPgePtr) -= m * gcrg;
                  *(here->BSIM4v5GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v5GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v5GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v5GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v5GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v5GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v5GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4v5GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4v5rgateMod == 3)
              {   *(here->BSIM4v5GEgePtr) += m * geltd;
                  *(here->BSIM4v5GEgmPtr) -= m * geltd;
                  *(here->BSIM4v5GMgePtr) -= m * geltd;
                  *(here->BSIM4v5GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v5GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4v5GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4v5GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v5GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4v5GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4v5GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v5GMspPtr) += m * gcrgs;
                  *(here->BSIM4v5GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4v5GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4v5GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v5GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4v5GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4v5DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4v5DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4v5GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v5SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4v5SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4v5BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4v5BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4v5GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v5GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v5GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v5GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v5GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v5GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v5GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4v5GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v5GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v5GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v5GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4v5GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4v5GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v5GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v5GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v5GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4v5rdsMod)
              {   (*(here->BSIM4v5DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v5DspPtr) += m * gdtots);
                  (*(here->BSIM4v5DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v5SdpPtr) += m * gstotd);
                  (*(here->BSIM4v5SgpPtr) += m * gstotg);
                  (*(here->BSIM4v5SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v5DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4v5DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4v5DPdpPtr) += m * (gdpr + gds + here->BSIM4v5gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4v5DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v5DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4v5DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4v5DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4v5DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4v5DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4v5DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4v5DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4v5DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4v5DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4v5DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v5DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v5SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4v5SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4v5SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4v5SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4v5SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4v5SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4v5SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4v5SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4v5SPspPtr) += m * (gspr + gds + here->BSIM4v5gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4v5SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v5SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4v5SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4v5SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4v5SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v5SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v5BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4v5BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4v5BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v5BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4v5BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4v5BPgpPtr) -= m * (here->BSIM4v5gbgs + gIbtotg);
              *(here->BSIM4v5BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4v5BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4v5BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v5BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4v5BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4v5BPbpPtr) += m * (gjbd + gjbs - here->BSIM4v5gbbs
				     - gIbtotb);
           ggidld = here->BSIM4v5ggidld;
           ggidlg = here->BSIM4v5ggidlg;
           ggidlb = here->BSIM4v5ggidlb;
           ggislg = here->BSIM4v5ggislg;
           ggisls = here->BSIM4v5ggisls;
           ggislb = here->BSIM4v5ggislb;

           /* stamp gidl */
           (*(here->BSIM4v5DPdpPtr) += m * ggidld);
           (*(here->BSIM4v5DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v5DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v5DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v5BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v5BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v5BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v5BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v5SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4v5SPgpPtr) += m * ggislg);
           (*(here->BSIM4v5SPspPtr) += m * ggisls);
           (*(here->BSIM4v5SPbpPtr) += m * ggislb);
           (*(here->BSIM4v5BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4v5BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v5BPspPtr) -= m * ggisls);
           (*(here->BSIM4v5BPbpPtr) -= m * ggislb);

              if (here->BSIM4v5rbodyMod)
              {   (*(here->BSIM4v5DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v5DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v5DPdbPtr) -= m * here->BSIM4v5gbd);
                  (*(here->BSIM4v5SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v5SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v5SPsbPtr) -= m * here->BSIM4v5gbs);

                  (*(here->BSIM4v5DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v5DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v5DBdpPtr) -= m * here->BSIM4v5gbd);
                  (*(here->BSIM4v5DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4v5DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4v5DBdbPtr) += m * (here->BSIM4v5gbd + here->BSIM4v5grbpd
                                          + here->BSIM4v5grbdb));
                  (*(here->BSIM4v5DBbpPtr) -= m * here->BSIM4v5grbpd);
                  (*(here->BSIM4v5DBbPtr) -= m * here->BSIM4v5grbdb);

                  (*(here->BSIM4v5BPdbPtr) -= m * here->BSIM4v5grbpd);
                  (*(here->BSIM4v5BPbPtr) -= m * here->BSIM4v5grbpb);
                  (*(here->BSIM4v5BPsbPtr) -= m * here->BSIM4v5grbps);
                  (*(here->BSIM4v5BPbpPtr) += m * (here->BSIM4v5grbpd + here->BSIM4v5grbps
					  + here->BSIM4v5grbpb));
                  /* WDL: (-here->BSIM4v5gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v5SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v5SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v5SBspPtr) -= m * here->BSIM4v5gbs);
                  (*(here->BSIM4v5SBbpPtr) -= m * here->BSIM4v5grbps);
                  (*(here->BSIM4v5SBbPtr) -= m * here->BSIM4v5grbsb);
                  (*(here->BSIM4v5SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4v5SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4v5SBsbPtr) += m * (here->BSIM4v5gbs
					  + here->BSIM4v5grbps + here->BSIM4v5grbsb));

                  (*(here->BSIM4v5BdbPtr) -= m * here->BSIM4v5grbdb);
                  (*(here->BSIM4v5BbpPtr) -= m * here->BSIM4v5grbpb);
                  (*(here->BSIM4v5BsbPtr) -= m * here->BSIM4v5grbsb);
                  (*(here->BSIM4v5BbPtr) += m * (here->BSIM4v5grbsb + here->BSIM4v5grbdb
                                        + here->BSIM4v5grbpb));
              }

              if (here->BSIM4v5acnqsMod)
              {   *(here->BSIM4v5QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4v5QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4v5QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4v5QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4v5QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4v5QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4v5QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4v5QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4v5QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4v5QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4v5GPqPtr) -= m * here->BSIM4v5gtau;
                  *(here->BSIM4v5DPqPtr) += m * dxpart * here->BSIM4v5gtau;
                  *(here->BSIM4v5SPqPtr) += m * sxpart * here->BSIM4v5gtau;

                  *(here->BSIM4v5QqPtr) += m * here->BSIM4v5gtau;
                  *(here->BSIM4v5QgpPtr) += m * xgtg;
                  *(here->BSIM4v5QdpPtr) += m * xgtd;
                  *(here->BSIM4v5QbpPtr) += m * xgtb;
                  *(here->BSIM4v5QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
