/**** BSIM4.3.0 Released by Xuemei (Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3pzld.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim4v3def.h"

int
BSIM4v3pzLoad(inModel,ckt,s)
GENmodel *inModel;
CKTcircuit *ckt;
SPcomplex *s;
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gds, gbd, gbs, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double xcdbdb, xcsbsb, xcgmgmb, xcgmdb, xcgmsb, xcdgmb, xcsgmb;
double xcgmbb, xcbgmb;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb, xcqdb, xcqsb, xcqbb;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T0, T1, CoxWL, qcheq, Cdg, Cdd, Cds, Cdb, Csg, Csd, Css, Csb;
double ScalingFactor = 1.0e-9;
struct bsim4v3SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb,ggisld, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = model->BSIM4v3nextModel) 
    {    for (here = model->BSIM4v3instances; here!= NULL;
              here = here->BSIM4v3nextInstance) 
         {    if (here->BSIM4v3owner != ARCHme) continue;
	      pParam = here->pParam;
              capbd = here->BSIM4v3capbd;
              capbs = here->BSIM4v3capbs;
              cgso = here->BSIM4v3cgso;
              cgdo = here->BSIM4v3cgdo;
              cgbo = pParam->BSIM4v3cgbo;

              if (here->BSIM4v3mode >= 0) 
              {   Gm = here->BSIM4v3gm;
                  Gmbs = here->BSIM4v3gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v3gbds);
                  gbbsp = here->BSIM4v3gbds + here->BSIM4v3gbgs + here->BSIM4v3gbbs;
                  gbdpg = here->BSIM4v3gbgs;
                  gbdpdp = here->BSIM4v3gbds;
                  gbdpb = here->BSIM4v3gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v3igcMod)
                  {   gIstotg = here->BSIM4v3gIgsg + here->BSIM4v3gIgcsg;
                      gIstotd = here->BSIM4v3gIgcsd;
                      gIstots = here->BSIM4v3gIgss + here->BSIM4v3gIgcss;
                      gIstotb = here->BSIM4v3gIgcsb;

                      gIdtotg = here->BSIM4v3gIgdg + here->BSIM4v3gIgcdg;
                      gIdtotd = here->BSIM4v3gIgdd + here->BSIM4v3gIgcdd;
                      gIdtots = here->BSIM4v3gIgcds;
                      gIdtotb = here->BSIM4v3gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v3igbMod)
                  {   gIbtotg = here->BSIM4v3gIgbg;
                      gIbtotd = here->BSIM4v3gIgbd;
                      gIbtots = here->BSIM4v3gIgbs;
                      gIbtotb = here->BSIM4v3gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v3igcMod != 0) || (model->BSIM4v3igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v3rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v3vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v3vgs);
                  else if (here->BSIM4v3rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v3vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v3vgs);
                  if (here->BSIM4v3rgateMod > 1)
                  {   gcrgd = here->BSIM4v3gcrgd * T0;
                      gcrgg = here->BSIM4v3gcrgg * T0;
                      gcrgs = here->BSIM4v3gcrgs * T0;
                      gcrgb = here->BSIM4v3gcrgb * T0;
                      gcrgg -= here->BSIM4v3gcrg;
                      gcrg = here->BSIM4v3gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v3acnqsMod == 0)
                  {   if (here->BSIM4v3rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v3cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v3cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v3cggb;
                          xcgdb = here->BSIM4v3cgdb;
                          xcgsb = here->BSIM4v3cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v3cdgb;
                          xcsgb = -(here->BSIM4v3cggb + here->BSIM4v3cbgb
                                + here->BSIM4v3cdgb);
                          xcbgb = here->BSIM4v3cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v3cggb + cgdo + cgso
                                + pParam->BSIM4v3cgbo;
                          xcgdb = here->BSIM4v3cgdb - cgdo;
                          xcgsb = here->BSIM4v3cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v3cdgb - cgdo;
                          xcsgb = -(here->BSIM4v3cggb + here->BSIM4v3cbgb
                                + here->BSIM4v3cdgb + cgso);
                          xcbgb = here->BSIM4v3cbgb - pParam->BSIM4v3cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v3cddb + here->BSIM4v3capbd + cgdo;
                      xcdsb = here->BSIM4v3cdsb;

                      xcsdb = -(here->BSIM4v3cgdb + here->BSIM4v3cbdb
                            + here->BSIM4v3cddb);
                      xcssb = here->BSIM4v3capbs + cgso - (here->BSIM4v3cgsb
                            + here->BSIM4v3cbsb + here->BSIM4v3cdsb);

                      if (!here->BSIM4v3rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v3cbdb - here->BSIM4v3capbd;
                          xcbsb = here->BSIM4v3cbsb - here->BSIM4v3capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v3cddb + here->BSIM4v3cdgb
                                 + here->BSIM4v3cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v3capbs;
                          xcbdb = here->BSIM4v3cbdb;
                          xcbsb = here->BSIM4v3cbsb;

                          xcdbdb = -here->BSIM4v3capbd;
                          xcsbsb = -here->BSIM4v3capbs;
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

		      xgtg = here->BSIM4v3gtg;
                      xgtd = here->BSIM4v3gtd;
                      xgts = here->BSIM4v3gts;
                      xgtb = here->BSIM4v3gtb;

                      xcqgb = here->BSIM4v3cqgb;
                      xcqdb = here->BSIM4v3cqdb;
                      xcqsb = here->BSIM4v3cqsb;
                      xcqbb = here->BSIM4v3cqbb;

		      CoxWL = model->BSIM4v3coxe * here->pParam->BSIM4v3weffCV
                            * here->BSIM4v3nf * here->pParam->BSIM4v3leffCV;
		      qcheq = -(here->BSIM4v3qgate + here->BSIM4v3qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v3xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v3xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v3qdrn / qcheq;
		          Cdd = here->BSIM4v3cddb;
		          Csd = -(here->BSIM4v3cgdb + here->BSIM4v3cddb
			      + here->BSIM4v3cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v3cdgb;
		          Csg = -(here->BSIM4v3cggb + here->BSIM4v3cdgb
			      + here->BSIM4v3cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v3cdsb;
		          Css = -(here->BSIM4v3cgsb + here->BSIM4v3cdsb
			      + here->BSIM4v3cbsb);
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
              {   Gm = -here->BSIM4v3gm;
                  Gmbs = -here->BSIM4v3gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v3gbds);
                  gbbdp = here->BSIM4v3gbds + here->BSIM4v3gbgs + here->BSIM4v3gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v3gbgs;
                  gbspsp = here->BSIM4v3gbds;
                  gbspb = here->BSIM4v3gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4v3igcMod)
                  {   gIstotg = here->BSIM4v3gIgsg + here->BSIM4v3gIgcdg;
                      gIstotd = here->BSIM4v3gIgcds;
                      gIstots = here->BSIM4v3gIgss + here->BSIM4v3gIgcdd;
                      gIstotb = here->BSIM4v3gIgcdb;

                      gIdtotg = here->BSIM4v3gIgdg + here->BSIM4v3gIgcsg;
                      gIdtotd = here->BSIM4v3gIgdd + here->BSIM4v3gIgcss;
                      gIdtots = here->BSIM4v3gIgcsd;
                      gIdtotb = here->BSIM4v3gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v3igbMod)
                  {   gIbtotg = here->BSIM4v3gIgbg;
                      gIbtotd = here->BSIM4v3gIgbs;
                      gIbtots = here->BSIM4v3gIgbd;
                      gIbtotb = here->BSIM4v3gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v3igcMod != 0) || (model->BSIM4v3igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v3rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v3vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v3vgs);
                  else if (here->BSIM4v3rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v3vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v3vgs);
                  if (here->BSIM4v3rgateMod > 1)
                  {   gcrgd = here->BSIM4v3gcrgs * T0;
                      gcrgg = here->BSIM4v3gcrgg * T0;
                      gcrgs = here->BSIM4v3gcrgd * T0;
                      gcrgb = here->BSIM4v3gcrgb * T0;
                      gcrgg -= here->BSIM4v3gcrg;
                      gcrg = here->BSIM4v3gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v3acnqsMod == 0)
                  {   if (here->BSIM4v3rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v3cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v3cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v3cggb;
                          xcgdb = here->BSIM4v3cgsb;
                          xcgsb = here->BSIM4v3cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v3cggb + here->BSIM4v3cbgb
                                + here->BSIM4v3cdgb);
                          xcsgb = here->BSIM4v3cdgb;
                          xcbgb = here->BSIM4v3cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v3cggb + cgdo + cgso
                                + pParam->BSIM4v3cgbo;
                          xcgdb = here->BSIM4v3cgsb - cgdo;
                          xcgsb = here->BSIM4v3cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v3cggb + here->BSIM4v3cbgb
                                + here->BSIM4v3cdgb + cgdo);
                          xcsgb = here->BSIM4v3cdgb - cgso;
                          xcbgb = here->BSIM4v3cbgb - pParam->BSIM4v3cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v3capbd + cgdo - (here->BSIM4v3cgsb
                            + here->BSIM4v3cbsb + here->BSIM4v3cdsb);
                      xcdsb = -(here->BSIM4v3cgdb + here->BSIM4v3cbdb
                            + here->BSIM4v3cddb);

                      xcsdb = here->BSIM4v3cdsb;
                      xcssb = here->BSIM4v3cddb + here->BSIM4v3capbs + cgso;

                      if (!here->BSIM4v3rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v3cbsb - here->BSIM4v3capbd;
                          xcbsb = here->BSIM4v3cbdb - here->BSIM4v3capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v3capbd;
                          xcsbb = -(here->BSIM4v3cddb + here->BSIM4v3cdgb
                                + here->BSIM4v3cdsb);
                          xcbdb = here->BSIM4v3cbsb;
                          xcbsb = here->BSIM4v3cbdb;
                          xcdbdb = -here->BSIM4v3capbd;
                          xcsbsb = -here->BSIM4v3capbs;
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

		      xgtg = here->BSIM4v3gtg;
                      xgtd = here->BSIM4v3gts;
                      xgts = here->BSIM4v3gtd;
                      xgtb = here->BSIM4v3gtb;

                      xcqgb = here->BSIM4v3cqgb;
                      xcqdb = here->BSIM4v3cqsb;
                      xcqsb = here->BSIM4v3cqdb;
                      xcqbb = here->BSIM4v3cqbb;

		      CoxWL = model->BSIM4v3coxe * here->pParam->BSIM4v3weffCV
                            * here->BSIM4v3nf * here->pParam->BSIM4v3leffCV;
		      qcheq = -(here->BSIM4v3qgate + here->BSIM4v3qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v3xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v3xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v3qdrn / qcheq;
		          Css = here->BSIM4v3cddb;
		          Cds = -(here->BSIM4v3cgdb + here->BSIM4v3cddb
			      + here->BSIM4v3cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v3cdgb;
		          Cdg = -(here->BSIM4v3cggb + here->BSIM4v3cdgb
			      + here->BSIM4v3cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v3cdsb;
		          Cdd = -(here->BSIM4v3cgsb + here->BSIM4v3cdsb
			      + here->BSIM4v3cbsb);
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

              if (model->BSIM4v3rdsMod == 1)
              {   gstot = here->BSIM4v3gstot;
                  gstotd = here->BSIM4v3gstotd;
                  gstotg = here->BSIM4v3gstotg;
                  gstots = here->BSIM4v3gstots - gstot;
                  gstotb = here->BSIM4v3gstotb;

                  gdtot = here->BSIM4v3gdtot;
                  gdtotd = here->BSIM4v3gdtotd - gdtot;
                  gdtotg = here->BSIM4v3gdtotg;
                  gdtots = here->BSIM4v3gdtots;
                  gdtotb = here->BSIM4v3gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v3qdef) * here->BSIM4v3gtau;
              gds = here->BSIM4v3gds;

              /*
               * Loading PZ matrix
               */

              m = here->BSIM4v3m;

              if (!model->BSIM4v3rdsMod)
              {   gdpr = here->BSIM4v3drainConductance;
                  gspr = here->BSIM4v3sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v3rbodyMod)
              {   gjbd = here->BSIM4v3gbd;
                  gjbs = here->BSIM4v3gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v3grgeltd;

              if (here->BSIM4v3rgateMod == 1)
              {   *(here->BSIM4v3GEgePtr) += m * geltd;
                  *(here->BSIM4v3GPgePtr) -= m * geltd;
                  *(here->BSIM4v3GEgpPtr) -= m * geltd;

                  *(here->BSIM4v3GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v3GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4v3GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v3GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v3GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v3GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v3GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v3GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4v3rgateMod == 2)
              {   *(here->BSIM4v3GEgePtr) += m * gcrg;
                  *(here->BSIM4v3GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v3GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v3GEspPtr) += m * gcrgs;
                  *(here->BSIM4v3GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v3GPgePtr) -= m * gcrg;
                  *(here->BSIM4v3GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v3GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v3GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v3GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v3GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v3GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v3GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4v3GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4v3rgateMod == 3)
              {   *(here->BSIM4v3GEgePtr) += m * geltd;
                  *(here->BSIM4v3GEgmPtr) -= m * geltd;
                  *(here->BSIM4v3GMgePtr) -= m * geltd;
                  *(here->BSIM4v3GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v3GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4v3GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4v3GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v3GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4v3GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4v3GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v3GMspPtr) += m * gcrgs;
                  *(here->BSIM4v3GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4v3GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4v3GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v3GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4v3GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4v3DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4v3DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4v3GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v3SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4v3SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4v3BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4v3BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4v3GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v3GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v3GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v3GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v3GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v3GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v3GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4v3GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v3GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v3GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v3GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4v3GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4v3GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v3GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v3GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v3GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4v3rdsMod)
              {   (*(here->BSIM4v3DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v3DspPtr) += m * gdtots);
                  (*(here->BSIM4v3DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v3SdpPtr) += m * gstotd);
                  (*(here->BSIM4v3SgpPtr) += m * gstotg);
                  (*(here->BSIM4v3SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v3DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4v3DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4v3DPdpPtr) += m * (gdpr + gds + here->BSIM4v3gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4v3DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v3DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4v3DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4v3DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4v3DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4v3DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4v3DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4v3DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4v3DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4v3DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4v3DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v3DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v3SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4v3SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4v3SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4v3SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4v3SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4v3SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4v3SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4v3SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4v3SPspPtr) += m * (gspr + gds + here->BSIM4v3gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4v3SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v3SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4v3SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4v3SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4v3SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v3SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v3BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4v3BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4v3BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v3BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4v3BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4v3BPgpPtr) -= m * (here->BSIM4v3gbgs + gIbtotg);
              *(here->BSIM4v3BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4v3BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4v3BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v3BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4v3BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4v3BPbpPtr) += m * (gjbd + gjbs - here->BSIM4v3gbbs
				     - gIbtotb);
           ggidld = here->BSIM4v3ggidld;
           ggidlg = here->BSIM4v3ggidlg;
           ggidlb = here->BSIM4v3ggidlb;
           ggislg = here->BSIM4v3ggislg;
           ggisls = here->BSIM4v3ggisls;
           ggislb = here->BSIM4v3ggislb;

           /* stamp gidl */
           (*(here->BSIM4v3DPdpPtr) += m * ggidld);
           (*(here->BSIM4v3DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v3DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v3DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v3BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v3BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v3BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v3BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v3SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4v3SPgpPtr) += m * ggislg);
           (*(here->BSIM4v3SPspPtr) += m * ggisls);
           (*(here->BSIM4v3SPbpPtr) += m * ggislb);
           (*(here->BSIM4v3BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4v3BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v3BPspPtr) -= m * ggisls);
           (*(here->BSIM4v3BPbpPtr) -= m * ggislb);

              if (here->BSIM4v3rbodyMod)
              {   (*(here->BSIM4v3DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v3DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v3DPdbPtr) -= m * here->BSIM4v3gbd);
                  (*(here->BSIM4v3SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v3SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v3SPsbPtr) -= m * here->BSIM4v3gbs);

                  (*(here->BSIM4v3DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v3DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v3DBdpPtr) -= m * here->BSIM4v3gbd);
                  (*(here->BSIM4v3DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4v3DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4v3DBdbPtr) += m * (here->BSIM4v3gbd + here->BSIM4v3grbpd
                                          + here->BSIM4v3grbdb));
                  (*(here->BSIM4v3DBbpPtr) -= m * here->BSIM4v3grbpd);
                  (*(here->BSIM4v3DBbPtr) -= m * here->BSIM4v3grbdb);

                  (*(here->BSIM4v3BPdbPtr) -= m * here->BSIM4v3grbpd);
                  (*(here->BSIM4v3BPbPtr) -= m * here->BSIM4v3grbpb);
                  (*(here->BSIM4v3BPsbPtr) -= m * here->BSIM4v3grbps);
                  (*(here->BSIM4v3BPbpPtr) += m * (here->BSIM4v3grbpd + here->BSIM4v3grbps
					  + here->BSIM4v3grbpb));
                  /* WDL: (-here->BSIM4v3gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v3SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v3SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v3SBspPtr) -= m * here->BSIM4v3gbs);
                  (*(here->BSIM4v3SBbpPtr) -= m * here->BSIM4v3grbps);
                  (*(here->BSIM4v3SBbPtr) -= m * here->BSIM4v3grbsb);
                  (*(here->BSIM4v3SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4v3SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4v3SBsbPtr) += m * (here->BSIM4v3gbs
					  + here->BSIM4v3grbps + here->BSIM4v3grbsb));

                  (*(here->BSIM4v3BdbPtr) -= m * here->BSIM4v3grbdb);
                  (*(here->BSIM4v3BbpPtr) -= m * here->BSIM4v3grbpb);
                  (*(here->BSIM4v3BsbPtr) -= m * here->BSIM4v3grbsb);
                  (*(here->BSIM4v3BbPtr) += m * (here->BSIM4v3grbsb + here->BSIM4v3grbdb
                                        + here->BSIM4v3grbpb));
              }

              if (here->BSIM4v3acnqsMod)
              {   *(here->BSIM4v3QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4v3QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4v3QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4v3QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4v3QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4v3QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4v3QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4v3QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4v3QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4v3QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4v3GPqPtr) -= m * here->BSIM4v3gtau;
                  *(here->BSIM4v3DPqPtr) += m * dxpart * here->BSIM4v3gtau;
                  *(here->BSIM4v3SPqPtr) += m * sxpart * here->BSIM4v3gtau;

                  *(here->BSIM4v3QqPtr) += m * here->BSIM4v3gtau;
                  *(here->BSIM4v3QgpPtr) += m * xgtg;
                  *(here->BSIM4v3QdpPtr) += m * xgtd;
                  *(here->BSIM4v3QbpPtr) += m * xgtb;
                  *(here->BSIM4v3QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
