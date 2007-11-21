/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.

 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "bsim4v2def.h"

int
BSIM4v2pzLoad(inModel,ckt,s)
GENmodel *inModel;
CKTcircuit *ckt;
SPcomplex *s;
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;

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
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb,ggisld, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = model->BSIM4v2nextModel) 
    {    for (here = model->BSIM4v2instances; here!= NULL;
              here = here->BSIM4v2nextInstance) 
	 {     if (here->BSIM4v2owner != ARCHme) continue;
	      pParam = here->pParam;
              capbd = here->BSIM4v2capbd;
              capbs = here->BSIM4v2capbs;
              cgso = here->BSIM4v2cgso;
              cgdo = here->BSIM4v2cgdo;
              cgbo = pParam->BSIM4v2cgbo;

              if (here->BSIM4v2mode >= 0) 
              {   Gm = here->BSIM4v2gm;
                  Gmbs = here->BSIM4v2gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v2gbds);
                  gbbsp = here->BSIM4v2gbds + here->BSIM4v2gbgs + here->BSIM4v2gbbs;
                  gbdpg = here->BSIM4v2gbgs;
                  gbdpdp = here->BSIM4v2gbds;
                  gbdpb = here->BSIM4v2gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v2igcMod)
                  {   gIstotg = here->BSIM4v2gIgsg + here->BSIM4v2gIgcsg;
                      gIstotd = here->BSIM4v2gIgcsd;
                      gIstots = here->BSIM4v2gIgss + here->BSIM4v2gIgcss;
                      gIstotb = here->BSIM4v2gIgcsb;

                      gIdtotg = here->BSIM4v2gIgdg + here->BSIM4v2gIgcdg;
                      gIdtotd = here->BSIM4v2gIgdd + here->BSIM4v2gIgcdd;
                      gIdtots = here->BSIM4v2gIgcds;
                      gIdtotb = here->BSIM4v2gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v2igbMod)
                  {   gIbtotg = here->BSIM4v2gIgbg;
                      gIbtotd = here->BSIM4v2gIgbd;
                      gIbtots = here->BSIM4v2gIgbs;
                      gIbtotb = here->BSIM4v2gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v2igcMod != 0) || (model->BSIM4v2igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v2rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v2vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v2vgs);
                  else if (here->BSIM4v2rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v2vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v2vgs);
                  if (here->BSIM4v2rgateMod > 1)
                  {   gcrgd = here->BSIM4v2gcrgd * T0;
                      gcrgg = here->BSIM4v2gcrgg * T0;
                      gcrgs = here->BSIM4v2gcrgs * T0;
                      gcrgb = here->BSIM4v2gcrgb * T0;
                      gcrgg -= here->BSIM4v2gcrg;
                      gcrg = here->BSIM4v2gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v2acnqsMod == 0)
                  {   if (here->BSIM4v2rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v2cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v2cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v2cggb;
                          xcgdb = here->BSIM4v2cgdb;
                          xcgsb = here->BSIM4v2cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v2cdgb;
                          xcsgb = -(here->BSIM4v2cggb + here->BSIM4v2cbgb
                                + here->BSIM4v2cdgb);
                          xcbgb = here->BSIM4v2cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v2cggb + cgdo + cgso
                                + pParam->BSIM4v2cgbo;
                          xcgdb = here->BSIM4v2cgdb - cgdo;
                          xcgsb = here->BSIM4v2cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v2cdgb - cgdo;
                          xcsgb = -(here->BSIM4v2cggb + here->BSIM4v2cbgb
                                + here->BSIM4v2cdgb + cgso);
                          xcbgb = here->BSIM4v2cbgb - pParam->BSIM4v2cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v2cddb + here->BSIM4v2capbd + cgdo;
                      xcdsb = here->BSIM4v2cdsb;

                      xcsdb = -(here->BSIM4v2cgdb + here->BSIM4v2cbdb
                            + here->BSIM4v2cddb);
                      xcssb = here->BSIM4v2capbs + cgso - (here->BSIM4v2cgsb
                            + here->BSIM4v2cbsb + here->BSIM4v2cdsb);

                      if (!here->BSIM4v2rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v2cbdb - here->BSIM4v2capbd;
                          xcbsb = here->BSIM4v2cbsb - here->BSIM4v2capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v2cddb + here->BSIM4v2cdgb
                                 + here->BSIM4v2cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v2capbs;
                          xcbdb = here->BSIM4v2cbdb;
                          xcbsb = here->BSIM4v2cbsb;

                          xcdbdb = -here->BSIM4v2capbd;
                          xcsbsb = -here->BSIM4v2capbs;
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

		      xgtg = here->BSIM4v2gtg;
                      xgtd = here->BSIM4v2gtd;
                      xgts = here->BSIM4v2gts;
                      xgtb = here->BSIM4v2gtb;

                      xcqgb = here->BSIM4v2cqgb;
                      xcqdb = here->BSIM4v2cqdb;
                      xcqsb = here->BSIM4v2cqsb;
                      xcqbb = here->BSIM4v2cqbb;

		      CoxWL = model->BSIM4v2coxe * here->pParam->BSIM4v2weffCV
                            * here->BSIM4v2nf * here->pParam->BSIM4v2leffCV;
		      qcheq = -(here->BSIM4v2qgate + here->BSIM4v2qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v2xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v2xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v2qdrn / qcheq;
		          Cdd = here->BSIM4v2cddb;
		          Csd = -(here->BSIM4v2cgdb + here->BSIM4v2cddb
			      + here->BSIM4v2cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v2cdgb;
		          Csg = -(here->BSIM4v2cggb + here->BSIM4v2cdgb
			      + here->BSIM4v2cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v2cdsb;
		          Css = -(here->BSIM4v2cgsb + here->BSIM4v2cdsb
			      + here->BSIM4v2cbsb);
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
              {   Gm = -here->BSIM4v2gm;
                  Gmbs = -here->BSIM4v2gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v2gbds);
                  gbbdp = here->BSIM4v2gbds + here->BSIM4v2gbgs + here->BSIM4v2gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v2gbgs;
                  gbspsp = here->BSIM4v2gbds;
                  gbspb = here->BSIM4v2gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4v2igcMod)
                  {   gIstotg = here->BSIM4v2gIgsg + here->BSIM4v2gIgcdg;
                      gIstotd = here->BSIM4v2gIgcds;
                      gIstots = here->BSIM4v2gIgss + here->BSIM4v2gIgcdd;
                      gIstotb = here->BSIM4v2gIgcdb;

                      gIdtotg = here->BSIM4v2gIgdg + here->BSIM4v2gIgcsg;
                      gIdtotd = here->BSIM4v2gIgdd + here->BSIM4v2gIgcss;
                      gIdtots = here->BSIM4v2gIgcsd;
                      gIdtotb = here->BSIM4v2gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v2igbMod)
                  {   gIbtotg = here->BSIM4v2gIgbg;
                      gIbtotd = here->BSIM4v2gIgbs;
                      gIbtots = here->BSIM4v2gIgbd;
                      gIbtotb = here->BSIM4v2gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v2igcMod != 0) || (model->BSIM4v2igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v2rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v2vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v2vgs);
                  else if (here->BSIM4v2rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v2vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v2vgs);
                  if (here->BSIM4v2rgateMod > 1)
                  {   gcrgd = here->BSIM4v2gcrgs * T0;
                      gcrgg = here->BSIM4v2gcrgg * T0;
                      gcrgs = here->BSIM4v2gcrgd * T0;
                      gcrgb = here->BSIM4v2gcrgb * T0;
                      gcrgg -= here->BSIM4v2gcrg;
                      gcrg = here->BSIM4v2gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v2acnqsMod == 0)
                  {   if (here->BSIM4v2rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v2cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v2cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v2cggb;
                          xcgdb = here->BSIM4v2cgsb;
                          xcgsb = here->BSIM4v2cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v2cggb + here->BSIM4v2cbgb
                                + here->BSIM4v2cdgb);
                          xcsgb = here->BSIM4v2cdgb;
                          xcbgb = here->BSIM4v2cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v2cggb + cgdo + cgso
                                + pParam->BSIM4v2cgbo;
                          xcgdb = here->BSIM4v2cgsb - cgdo;
                          xcgsb = here->BSIM4v2cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v2cggb + here->BSIM4v2cbgb
                                + here->BSIM4v2cdgb + cgdo);
                          xcsgb = here->BSIM4v2cdgb - cgso;
                          xcbgb = here->BSIM4v2cbgb - pParam->BSIM4v2cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v2capbd + cgdo - (here->BSIM4v2cgsb
                            + here->BSIM4v2cbsb + here->BSIM4v2cdsb);
                      xcdsb = -(here->BSIM4v2cgdb + here->BSIM4v2cbdb
                            + here->BSIM4v2cddb);

                      xcsdb = here->BSIM4v2cdsb;
                      xcssb = here->BSIM4v2cddb + here->BSIM4v2capbs + cgso;

                      if (!here->BSIM4v2rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v2cbsb - here->BSIM4v2capbd;
                          xcbsb = here->BSIM4v2cbdb - here->BSIM4v2capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v2capbd;
                          xcsbb = -(here->BSIM4v2cddb + here->BSIM4v2cdgb
                                + here->BSIM4v2cdsb);
                          xcbdb = here->BSIM4v2cbsb;
                          xcbsb = here->BSIM4v2cbdb;
                          xcdbdb = -here->BSIM4v2capbd;
                          xcsbsb = -here->BSIM4v2capbs;
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

		      xgtg = here->BSIM4v2gtg;
                      xgtd = here->BSIM4v2gts;
                      xgts = here->BSIM4v2gtd;
                      xgtb = here->BSIM4v2gtb;

                      xcqgb = here->BSIM4v2cqgb;
                      xcqdb = here->BSIM4v2cqsb;
                      xcqsb = here->BSIM4v2cqdb;
                      xcqbb = here->BSIM4v2cqbb;

		      CoxWL = model->BSIM4v2coxe * here->pParam->BSIM4v2weffCV
                            * here->BSIM4v2nf * here->pParam->BSIM4v2leffCV;
		      qcheq = -(here->BSIM4v2qgate + here->BSIM4v2qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v2xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v2xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v2qdrn / qcheq;
		          Css = here->BSIM4v2cddb;
		          Cds = -(here->BSIM4v2cgdb + here->BSIM4v2cddb
			      + here->BSIM4v2cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v2cdgb;
		          Cdg = -(here->BSIM4v2cggb + here->BSIM4v2cdgb
			      + here->BSIM4v2cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v2cdsb;
		          Cdd = -(here->BSIM4v2cgsb + here->BSIM4v2cdsb
			      + here->BSIM4v2cbsb);
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

              if (model->BSIM4v2rdsMod == 1)
              {   gstot = here->BSIM4v2gstot;
                  gstotd = here->BSIM4v2gstotd;
                  gstotg = here->BSIM4v2gstotg;
                  gstots = here->BSIM4v2gstots - gstot;
                  gstotb = here->BSIM4v2gstotb;

                  gdtot = here->BSIM4v2gdtot;
                  gdtotd = here->BSIM4v2gdtotd - gdtot;
                  gdtotg = here->BSIM4v2gdtotg;
                  gdtots = here->BSIM4v2gdtots;
                  gdtotb = here->BSIM4v2gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v2qdef) * here->BSIM4v2gtau;
              gds = here->BSIM4v2gds;

              /*
               * Loading PZ matrix
               */

              m = here->BSIM4v2m;

              if (!model->BSIM4v2rdsMod)
              {   gdpr = here->BSIM4v2drainConductance;
                  gspr = here->BSIM4v2sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v2rbodyMod)
              {   gjbd = here->BSIM4v2gbd;
                  gjbs = here->BSIM4v2gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v2grgeltd;

              if (here->BSIM4v2rgateMod == 1)
              {   *(here->BSIM4v2GEgePtr) += m * geltd;
                  *(here->BSIM4v2GPgePtr) -= m * geltd;
                  *(here->BSIM4v2GEgpPtr) -= m * geltd;

                  *(here->BSIM4v2GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v2GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v2GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4v2GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v2GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v2GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v2GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v2GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v2GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v2GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v2GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v2GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4v2rgateMod == 2)
              {   *(here->BSIM4v2GEgePtr) += m * gcrg;
                  *(here->BSIM4v2GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v2GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v2GEspPtr) += m * gcrgs;
                  *(here->BSIM4v2GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v2GPgePtr) -= m * gcrg;
                  *(here->BSIM4v2GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v2GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v2GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v2GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v2GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v2GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v2GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v2GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v2GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v2GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v2GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4v2GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4v2rgateMod == 3)
              {   *(here->BSIM4v2GEgePtr) += m * geltd;
                  *(here->BSIM4v2GEgmPtr) -= m * geltd;
                  *(here->BSIM4v2GMgePtr) -= m * geltd;
                  *(here->BSIM4v2GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v2GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4v2GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4v2GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v2GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4v2GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4v2GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v2GMspPtr) += m * gcrgs;
                  *(here->BSIM4v2GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4v2GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4v2GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v2GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4v2GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4v2DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4v2DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4v2GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v2SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4v2SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4v2BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4v2BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4v2GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v2GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v2GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v2GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v2GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v2GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v2GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v2GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v2GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v2GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4v2GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v2GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v2GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v2GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v2GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v2GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v2GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4v2GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4v2GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v2GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v2GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v2GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v2GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v2GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4v2rdsMod)
              {   (*(here->BSIM4v2DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v2DspPtr) += m * gdtots);
                  (*(here->BSIM4v2DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v2SdpPtr) += m * gstotd);
                  (*(here->BSIM4v2SgpPtr) += m * gstotg);
                  (*(here->BSIM4v2SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v2DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4v2DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4v2DPdpPtr) += m * (gdpr + gds + here->BSIM4v2gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4v2DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v2DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4v2DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4v2DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4v2DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4v2DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4v2DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4v2DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4v2DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4v2DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4v2DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v2DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v2SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4v2SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4v2SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4v2SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4v2SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4v2SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4v2SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4v2SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4v2SPspPtr) += m * (gspr + gds + here->BSIM4v2gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4v2SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v2SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4v2SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4v2SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4v2SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v2SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v2BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4v2BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4v2BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v2BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4v2BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4v2BPgpPtr) -= m * (here->BSIM4v2gbgs + gIbtotg);
              *(here->BSIM4v2BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4v2BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4v2BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v2BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4v2BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4v2BPbpPtr) += m * (gjbd + gjbs - here->BSIM4v2gbbs
				     - gIbtotb);
           ggidld = here->BSIM4v2ggidld;
           ggidlg = here->BSIM4v2ggidlg;
           ggidlb = here->BSIM4v2ggidlb;
           ggislg = here->BSIM4v2ggislg;
           ggisls = here->BSIM4v2ggisls;
           ggislb = here->BSIM4v2ggislb;

           /* stamp gidl */
           (*(here->BSIM4v2DPdpPtr) += m * ggidld);
           (*(here->BSIM4v2DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v2DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v2DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v2BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v2BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v2BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v2BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v2SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4v2SPgpPtr) += m * ggislg);
           (*(here->BSIM4v2SPspPtr) += m * ggisls);
           (*(here->BSIM4v2SPbpPtr) += m * ggislb);
           (*(here->BSIM4v2BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4v2BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v2BPspPtr) -= m * ggisls);
           (*(here->BSIM4v2BPbpPtr) -= m * ggislb);

              if (here->BSIM4v2rbodyMod)
              {   (*(here->BSIM4v2DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v2DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v2DPdbPtr) -= m * here->BSIM4v2gbd);
                  (*(here->BSIM4v2SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v2SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v2SPsbPtr) -= m * here->BSIM4v2gbs);

                  (*(here->BSIM4v2DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v2DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v2DBdpPtr) -= m * here->BSIM4v2gbd);
                  (*(here->BSIM4v2DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4v2DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4v2DBdbPtr) += m * (here->BSIM4v2gbd + here->BSIM4v2grbpd
                                          + here->BSIM4v2grbdb));
                  (*(here->BSIM4v2DBbpPtr) -= m * here->BSIM4v2grbpd);
                  (*(here->BSIM4v2DBbPtr) -= m * here->BSIM4v2grbdb);

                  (*(here->BSIM4v2BPdbPtr) -= m * here->BSIM4v2grbpd);
                  (*(here->BSIM4v2BPbPtr) -= m * here->BSIM4v2grbpb);
                  (*(here->BSIM4v2BPsbPtr) -= m * here->BSIM4v2grbps);
                  (*(here->BSIM4v2BPbpPtr) += m * (here->BSIM4v2grbpd + here->BSIM4v2grbps
					  + here->BSIM4v2grbpb));
                  /* WDL: (-here->BSIM4v2gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v2SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v2SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v2SBspPtr) -= m * here->BSIM4v2gbs);
                  (*(here->BSIM4v2SBbpPtr) -= m * here->BSIM4v2grbps);
                  (*(here->BSIM4v2SBbPtr) -= m * here->BSIM4v2grbsb);
                  (*(here->BSIM4v2SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4v2SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4v2SBsbPtr) += m * (here->BSIM4v2gbs
					  + here->BSIM4v2grbps + here->BSIM4v2grbsb));

                  (*(here->BSIM4v2BdbPtr) -= m * here->BSIM4v2grbdb);
                  (*(here->BSIM4v2BbpPtr) -= m * here->BSIM4v2grbpb);
                  (*(here->BSIM4v2BsbPtr) -= m * here->BSIM4v2grbsb);
                  (*(here->BSIM4v2BbPtr) += m * (here->BSIM4v2grbsb + here->BSIM4v2grbdb
                                        + here->BSIM4v2grbpb));
              }

              if (here->BSIM4v2acnqsMod)
              {   *(here->BSIM4v2QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4v2QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4v2QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4v2QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4v2QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4v2QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4v2QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4v2QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4v2QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4v2QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4v2GPqPtr) -= m * here->BSIM4v2gtau;
                  *(here->BSIM4v2DPqPtr) += m * dxpart * here->BSIM4v2gtau;
                  *(here->BSIM4v2SPqPtr) += m * sxpart * here->BSIM4v2gtau;

                  *(here->BSIM4v2QqPtr) += m * here->BSIM4v2gtau;
                  *(here->BSIM4v2QgpPtr) += m * xgtg;
                  *(here->BSIM4v2QdpPtr) += m * xgtd;
                  *(here->BSIM4v2QbpPtr) += m * xgtb;
                  *(here->BSIM4v2QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
