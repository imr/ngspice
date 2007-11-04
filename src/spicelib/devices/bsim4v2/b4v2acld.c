/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Xuemei Xi 10/05/2001
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim4v2def.h"
#include "sperror.h"



int
BSIM4v2acLoad(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4v2model *model = (BSIM4v2model*)inModel;
BSIM4v2instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcbgb, xcbdb, xcbsb, xcbbb;
double xcggbr, xcgdbr, xcgsbr, xcgbbr, xcggbi, xcgdbi, xcgsbi, xcgbbi;
double Cggr, Cgdr, Cgsr, Cgbr, Cggi, Cgdi, Cgsi, Cgbi;
double xcddbr, xcdgbr, xcdsbr, xcdbbr, xcsdbr, xcsgbr, xcssbr, xcsbbr;
double xcddbi, xcdgbi, xcdsbi, xcdbbi, xcsdbi, xcsgbi, xcssbi, xcsbbi;
double xcdbdb, xcsbsb, xcgmgmb, xcgmdb, xcgmsb, xcdgmb, xcsgmb;
double xcgmbb, xcbgmb;
double capbd, capbs, omega;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11;
double Csg, Csd, Css, Csb;
double Cdgr, Cddr, Cdsr, Cdbr, Csgr, Csdr, Cssr, Csbr;
double Cdgi, Cddi, Cdsi, Cdbi, Csgi, Csdi, Cssi, Csbi;
double gmr, gmi, gmbsr, gmbsi, gdsr, gdsi;
double FwdSumr, RevSumr, Gmr, Gmbsr, Gdsr;
double FwdSumi, RevSumi, Gmi, Gmbsi, Gdsi;
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb,ggisld, ggislg, ggislb, ggisls;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM4v2nextModel) 
    {    for (here = model->BSIM4v2instances; here!= NULL;
              here = here->BSIM4v2nextInstance) 
	 {        if (here->BSIM4v2owner != ARCHme) continue;
	          pParam = here->pParam;
              capbd = here->BSIM4v2capbd;
              capbs = here->BSIM4v2capbs;
              cgso = here->BSIM4v2cgso;
              cgdo = here->BSIM4v2cgdo;
              cgbo = pParam->BSIM4v2cgbo;

              Csd = -(here->BSIM4v2cddb + here->BSIM4v2cgdb + here->BSIM4v2cbdb);
              Csg = -(here->BSIM4v2cdgb + here->BSIM4v2cggb + here->BSIM4v2cbgb);
              Css = -(here->BSIM4v2cdsb + here->BSIM4v2cgsb + here->BSIM4v2cbsb);

              if (here->BSIM4v2acnqsMod)
              {   T0 = omega * here->BSIM4v2taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v2gm * T2;
                  gmbsr = here->BSIM4v2gmbs * T2;
                  gdsr = here->BSIM4v2gds * T2;

                  gmi = -here->BSIM4v2gm * T3;
                  gmbsi = -here->BSIM4v2gmbs * T3;
                  gdsi = -here->BSIM4v2gds * T3;

                  Cddr = here->BSIM4v2cddb * T2;
                  Cdgr = here->BSIM4v2cdgb * T2;
                  Cdsr = here->BSIM4v2cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v2cddb * T3 * omega;
                  Cdgi = here->BSIM4v2cdgb * T3 * omega;
                  Cdsi = here->BSIM4v2cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4v2cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4v2cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4v2cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v2gm;
                  gmbsr = here->BSIM4v2gmbs;
                  gdsr = here->BSIM4v2gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v2cddb;
                  Cdgr = here->BSIM4v2cdgb;
                  Cdsr = here->BSIM4v2cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v2cgdb;
                  Cggr = here->BSIM4v2cggb;
                  Cgsr = here->BSIM4v2cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v2mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4v2rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v2cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v2cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v2cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v2cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v2cbgb - pParam->BSIM4v2cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v2capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v2capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v2rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v2cbdb - here->BSIM4v2capbd) * omega;
                      xcbsb = (here->BSIM4v2cbsb - here->BSIM4v2capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4v2capbs * omega;

                      xcbdb = here->BSIM4v2cbdb * omega;
                      xcbsb = here->BSIM4v2cbsb * omega;
    
                      xcdbdb = -here->BSIM4v2capbd * omega;
                      xcsbsb = -here->BSIM4v2capbs * omega;
                  }
                  xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

                  xcdgbi = Cdgi;
                  xcsgbi = Csgi;
                  xcddbi = Cddi;
                  xcdsbi = Cdsi;
                  xcsdbi = Csdi;
                  xcssbi = Cssi;
                  xcdbbi = Cdbi;
                  xcsbbi = Csbi;
                  xcggbi = Cggi;
                  xcgdbi = Cgdi;
                  xcgsbi = Cgsi;
                  xcgbbi = Cgbi;
              } 
              else /* Reverse mode */
              {   Gmr = -gmr;
                  Gmbsr = -gmbsr;
                  FwdSumr = 0.0;
                  RevSumr = -(Gmr + Gmbsr);
                  Gmi = -gmi;
                  Gmbsi = -gmbsi;
                  FwdSumi = 0.0;
                  RevSumi = -(Gmi + Gmbsi);

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

                  if (here->BSIM4v2rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v2cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v2cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v2cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v2cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v2cbgb - pParam->BSIM4v2cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v2capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v2capbs + cgso) * omega;
    
                  if (!here->BSIM4v2rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v2cbsb - here->BSIM4v2capbd) * omega;
                      xcbsb = (here->BSIM4v2cbdb - here->BSIM4v2capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v2capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v2cbsb * omega;
                      xcbsb = here->BSIM4v2cbdb * omega;
                      xcdbdb = -here->BSIM4v2capbd * omega;
                      xcsbsb = -here->BSIM4v2capbs * omega;
                  }
                  xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

                  xcdgbi = Csgi;
                  xcsgbi = Cdgi;
                  xcddbi = Cssi;
                  xcdsbi = Csdi;
                  xcsdbi = Cdsi;
                  xcssbi = Cddi;
                  xcdbbi = Csbi;
                  xcsbbi = Cdbi;
                  xcggbi = Cggi;
                  xcgdbi = Cgsi;
                  xcgsbi = Cgdi;
                  xcgbbi = Cgbi;
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


              /*
               * Loading AC matrix
               */

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
              {   *(here->BSIM4v2GEgePtr) += geltd;
                  *(here->BSIM4v2GPgePtr) -= geltd;
                  *(here->BSIM4v2GEgpPtr) -= geltd;

                  *(here->BSIM4v2GPgpPtr +1) += xcggbr;
		  *(here->BSIM4v2GPgpPtr) += geltd + xcggbi + gIgtotg;
                  *(here->BSIM4v2GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v2GPdpPtr) += xcgdbi + gIgtotd;
                  *(here->BSIM4v2GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v2GPspPtr) += xcgsbi + gIgtots;
                  *(here->BSIM4v2GPbpPtr +1) += xcgbbr;
                  *(here->BSIM4v2GPbpPtr) += xcgbbi + gIgtotb;
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v2rgateMod == 2)
              {   *(here->BSIM4v2GEgePtr) += gcrg;
                  *(here->BSIM4v2GEgpPtr) += gcrgg;
                  *(here->BSIM4v2GEdpPtr) += gcrgd;
                  *(here->BSIM4v2GEspPtr) += gcrgs;
                  *(here->BSIM4v2GEbpPtr) += gcrgb;

                  *(here->BSIM4v2GPgePtr) -= gcrg;
                  *(here->BSIM4v2GPgpPtr +1) += xcggbr;
		  *(here->BSIM4v2GPgpPtr) -= gcrgg - xcggbi - gIgtotg;
                  *(here->BSIM4v2GPdpPtr +1) += xcgdbr;
		  *(here->BSIM4v2GPdpPtr) -= gcrgd - xcgdbi - gIgtotd;
                  *(here->BSIM4v2GPspPtr +1) += xcgsbr;
		  *(here->BSIM4v2GPspPtr) -= gcrgs - xcgsbi - gIgtots;
                  *(here->BSIM4v2GPbpPtr +1) += xcgbbr;
		  *(here->BSIM4v2GPbpPtr) -= gcrgb - xcgbbi - gIgtotb;
              }
              else if (here->BSIM4v2rgateMod == 3)
              {   *(here->BSIM4v2GEgePtr) += geltd;
                  *(here->BSIM4v2GEgmPtr) -= geltd;
                  *(here->BSIM4v2GMgePtr) -= geltd;
                  *(here->BSIM4v2GMgmPtr) += geltd + gcrg;
                  *(here->BSIM4v2GMgmPtr +1) += xcgmgmb;
   
                  *(here->BSIM4v2GMdpPtr) += gcrgd;
                  *(here->BSIM4v2GMdpPtr +1) += xcgmdb;
                  *(here->BSIM4v2GMgpPtr) += gcrgg;
                  *(here->BSIM4v2GMspPtr) += gcrgs;
                  *(here->BSIM4v2GMspPtr +1) += xcgmsb;
                  *(here->BSIM4v2GMbpPtr) += gcrgb;
                  *(here->BSIM4v2GMbpPtr +1) += xcgmbb;
   
                  *(here->BSIM4v2DPgmPtr +1) += xcdgmb;
                  *(here->BSIM4v2GPgmPtr) -= gcrg;
                  *(here->BSIM4v2SPgmPtr +1) += xcsgmb;
                  *(here->BSIM4v2BPgmPtr +1) += xcbgmb;
   
                  *(here->BSIM4v2GPgpPtr) -= gcrgg - xcggbi - gIgtotg;
                  *(here->BSIM4v2GPgpPtr +1) += xcggbr;
                  *(here->BSIM4v2GPdpPtr) -= gcrgd - xcgdbi - gIgtotd;
                  *(here->BSIM4v2GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v2GPspPtr) -= gcrgs - xcgsbi - gIgtots;
                  *(here->BSIM4v2GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v2GPbpPtr) -= gcrgb - xcgbbi - gIgtotb;
                  *(here->BSIM4v2GPbpPtr +1) += xcgbbr;
              }
              else
              {   *(here->BSIM4v2GPgpPtr +1) += xcggbr;
                  *(here->BSIM4v2GPgpPtr) += xcggbi + gIgtotg;
                  *(here->BSIM4v2GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v2GPdpPtr) += xcgdbi + gIgtotd;
                  *(here->BSIM4v2GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v2GPspPtr) += xcgsbi + gIgtots;
                  *(here->BSIM4v2GPbpPtr +1) += xcgbbr;
                  *(here->BSIM4v2GPbpPtr) += xcgbbi + gIgtotb;
              }

              if (model->BSIM4v2rdsMod)
              {   (*(here->BSIM4v2DgpPtr) += gdtotg);
                  (*(here->BSIM4v2DspPtr) += gdtots);
                  (*(here->BSIM4v2DbpPtr) += gdtotb);
                  (*(here->BSIM4v2SdpPtr) += gstotd);
                  (*(here->BSIM4v2SgpPtr) += gstotg);
                  (*(here->BSIM4v2SbpPtr) += gstotb);
              }

              *(here->BSIM4v2DPdpPtr +1) += xcddbr + gdsi + RevSumi;
              *(here->BSIM4v2DPdpPtr) += gdpr + xcddbi + gdsr + here->BSIM4v2gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd;
              *(here->BSIM4v2DPdPtr) -= gdpr + gdtot;
              *(here->BSIM4v2DPgpPtr +1) += xcdgbr + Gmi;
              *(here->BSIM4v2DPgpPtr) += Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg;
              *(here->BSIM4v2DPspPtr +1) += xcdsbr - gdsi - FwdSumi;
              *(here->BSIM4v2DPspPtr) -= gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots;
              *(here->BSIM4v2DPbpPtr +1) += xcdbbr + Gmbsi;
              *(here->BSIM4v2DPbpPtr) -= gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb;

              *(here->BSIM4v2DdpPtr) -= gdpr - gdtotd;
              *(here->BSIM4v2DdPtr) += gdpr + gdtot;

              *(here->BSIM4v2SPdpPtr +1) += xcsdbr - gdsi - RevSumi;
              *(here->BSIM4v2SPdpPtr) -= gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd;
              *(here->BSIM4v2SPgpPtr +1) += xcsgbr - Gmi;
              *(here->BSIM4v2SPgpPtr) -= Gmr - xcsgbi + gstotg - gbspg + gIstotg;
              *(here->BSIM4v2SPspPtr +1) += xcssbr + gdsi + FwdSumi;
              *(here->BSIM4v2SPspPtr) += gspr + xcssbi + gdsr + here->BSIM4v2gbs
				     - gstots + FwdSumr + gbspsp - gIstots;
              *(here->BSIM4v2SPsPtr) -= gspr + gstot;
              *(here->BSIM4v2SPbpPtr +1) += xcsbbr - Gmbsi;
              *(here->BSIM4v2SPbpPtr) -= gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb;

              *(here->BSIM4v2SspPtr) -= gspr - gstots;
              *(here->BSIM4v2SsPtr) += gspr + gstot;

              *(here->BSIM4v2BPdpPtr +1) += xcbdb;
              *(here->BSIM4v2BPdpPtr) -= gjbd - gbbdp + gIbtotd;
              *(here->BSIM4v2BPgpPtr +1) += xcbgb;
              *(here->BSIM4v2BPgpPtr) -= here->BSIM4v2gbgs + gIbtotg;
              *(here->BSIM4v2BPspPtr +1) += xcbsb;
              *(here->BSIM4v2BPspPtr) -= gjbs - gbbsp + gIbtots;
              *(here->BSIM4v2BPbpPtr +1) += xcbbb;
              *(here->BSIM4v2BPbpPtr) += gjbd + gjbs - here->BSIM4v2gbbs
				     - gIbtotb;
           ggidld = here->BSIM4v2ggidld;
           ggidlg = here->BSIM4v2ggidlg;
           ggidlb = here->BSIM4v2ggidlb;
           ggislg = here->BSIM4v2ggislg;
           ggisls = here->BSIM4v2ggisls;
           ggislb = here->BSIM4v2ggislb;

           /* stamp gidl */
           (*(here->BSIM4v2DPdpPtr) += ggidld);
           (*(here->BSIM4v2DPgpPtr) += ggidlg);
           (*(here->BSIM4v2DPspPtr) -= (ggidlg + ggidld) + ggidlb);
           (*(here->BSIM4v2DPbpPtr) += ggidlb);
           (*(here->BSIM4v2BPdpPtr) -= ggidld);
           (*(here->BSIM4v2BPgpPtr) -= ggidlg);
           (*(here->BSIM4v2BPspPtr) += (ggidlg + ggidld) + ggidlb);
           (*(here->BSIM4v2BPbpPtr) -= ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v2SPdpPtr) -= (ggisls + ggislg) + ggislb);
           (*(here->BSIM4v2SPgpPtr) += ggislg);
           (*(here->BSIM4v2SPspPtr) += ggisls);
           (*(here->BSIM4v2SPbpPtr) += ggislb);
           (*(here->BSIM4v2BPdpPtr) += (ggislg + ggisls) + ggislb);
           (*(here->BSIM4v2BPgpPtr) -= ggislg);
           (*(here->BSIM4v2BPspPtr) -= ggisls);
           (*(here->BSIM4v2BPbpPtr) -= ggislb);

              if (here->BSIM4v2rbodyMod)
              {   (*(here->BSIM4v2DPdbPtr +1) += xcdbdb);
                  (*(here->BSIM4v2DPdbPtr) -= here->BSIM4v2gbd);
                  (*(here->BSIM4v2SPsbPtr +1) += xcsbsb);
                  (*(here->BSIM4v2SPsbPtr) -= here->BSIM4v2gbs);

                  (*(here->BSIM4v2DBdpPtr +1) += xcdbdb);
                  (*(here->BSIM4v2DBdpPtr) -= here->BSIM4v2gbd);
                  (*(here->BSIM4v2DBdbPtr +1) -= xcdbdb);
                  (*(here->BSIM4v2DBdbPtr) += here->BSIM4v2gbd + here->BSIM4v2grbpd 
                                          + here->BSIM4v2grbdb);
                  (*(here->BSIM4v2DBbpPtr) -= here->BSIM4v2grbpd);
                  (*(here->BSIM4v2DBbPtr) -= here->BSIM4v2grbdb);

                  (*(here->BSIM4v2BPdbPtr) -= here->BSIM4v2grbpd);
                  (*(here->BSIM4v2BPbPtr) -= here->BSIM4v2grbpb);
                  (*(here->BSIM4v2BPsbPtr) -= here->BSIM4v2grbps);
                  (*(here->BSIM4v2BPbpPtr) += here->BSIM4v2grbpd + here->BSIM4v2grbps 
					  + here->BSIM4v2grbpb);
		  /* WDLiu: (-here->BSIM4v2gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v2SBspPtr +1) += xcsbsb);
                  (*(here->BSIM4v2SBspPtr) -= here->BSIM4v2gbs);
                  (*(here->BSIM4v2SBbpPtr) -= here->BSIM4v2grbps);
                  (*(here->BSIM4v2SBbPtr) -= here->BSIM4v2grbsb);
                  (*(here->BSIM4v2SBsbPtr +1) -= xcsbsb);
                  (*(here->BSIM4v2SBsbPtr) += here->BSIM4v2gbs
					  + here->BSIM4v2grbps + here->BSIM4v2grbsb);

                  (*(here->BSIM4v2BdbPtr) -= here->BSIM4v2grbdb);
                  (*(here->BSIM4v2BbpPtr) -= here->BSIM4v2grbpb);
                  (*(here->BSIM4v2BsbPtr) -= here->BSIM4v2grbsb);
                  (*(here->BSIM4v2BbPtr) += here->BSIM4v2grbsb + here->BSIM4v2grbdb
                                        + here->BSIM4v2grbpb);
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4v2trnqsMod)
           {   (*(here->BSIM4v2QqPtr) += 1.0);
               (*(here->BSIM4v2QgpPtr) += 0.0);
               (*(here->BSIM4v2QdpPtr) += 0.0);
               (*(here->BSIM4v2QspPtr) += 0.0);
               (*(here->BSIM4v2QbpPtr) += 0.0);

               (*(here->BSIM4v2DPqPtr) += 0.0);
               (*(here->BSIM4v2SPqPtr) += 0.0);
               (*(here->BSIM4v2GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
