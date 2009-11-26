/**** BSIM4.3.0 Released by Xuemei (Jane) Xi  05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3acld.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4v3def.h"
#include "sperror.h"


int
BSIM4v3acLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v3model *model = (BSIM4v3model*)inModel;
BSIM4v3instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcbgb, xcbdb, xcbsb, xcbbb;
double xcggbr, xcgdbr, xcgsbr, xcgbbr, xcggbi, xcgdbi, xcgsbi, xcgbbi;
double Cggr, Cgdr, Cgsr, Cgbr, Cggi, Cgdi, Cgsi, Cgbi;
double xcddbr, xcdgbr, xcdsbr, xcdbbr, xcsdbr, xcsgbr, xcssbr, xcsbbr;
double xcddbi, xcdgbi, xcdsbi, xcdbbi, xcsdbi, xcsgbi, xcssbi, xcsbbi;
double xcdbdb, xcsbsb=0.0, xcgmgmb=0.0, xcgmdb=0.0, xcgmsb=0.0, xcdgmb, xcsgmb;
double xcgmbb=0.0, xcbgmb;
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
double T0=0.0, T1, T2, T3;
double Csg, Csd, Css;
double Cdgr, Cddr, Cdsr, Cdbr, Csgr, Csdr, Cssr, Csbr;
double Cdgi, Cddi, Cdsi, Cdbi, Csgi, Csdi, Cssi, Csbi;
double gmr, gmi, gmbsr, gmbsi, gdsr, gdsi;
double FwdSumr, RevSumr, Gmr, Gmbsr;
double FwdSumi, RevSumi, Gmi, Gmbsi;
struct bsim4v3SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM4v3nextModel) 
    {    for (here = model->BSIM4v3instances; here!= NULL;
              here = here->BSIM4v3nextInstance) 
         {        if (here->BSIM4v3owner != ARCHme) continue;
                  pParam = here->pParam;
              capbd = here->BSIM4v3capbd;
              capbs = here->BSIM4v3capbs;
              cgso = here->BSIM4v3cgso;
              cgdo = here->BSIM4v3cgdo;
              cgbo = pParam->BSIM4v3cgbo;

              Csd = -(here->BSIM4v3cddb + here->BSIM4v3cgdb + here->BSIM4v3cbdb);
              Csg = -(here->BSIM4v3cdgb + here->BSIM4v3cggb + here->BSIM4v3cbgb);
              Css = -(here->BSIM4v3cdsb + here->BSIM4v3cgsb + here->BSIM4v3cbsb);

              if (here->BSIM4v3acnqsMod)
              {   T0 = omega * here->BSIM4v3taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v3gm * T2;
                  gmbsr = here->BSIM4v3gmbs * T2;
                  gdsr = here->BSIM4v3gds * T2;

                  gmi = -here->BSIM4v3gm * T3;
                  gmbsi = -here->BSIM4v3gmbs * T3;
                  gdsi = -here->BSIM4v3gds * T3;

                  Cddr = here->BSIM4v3cddb * T2;
                  Cdgr = here->BSIM4v3cdgb * T2;
                  Cdsr = here->BSIM4v3cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v3cddb * T3 * omega;
                  Cdgi = here->BSIM4v3cdgb * T3 * omega;
                  Cdsi = here->BSIM4v3cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4v3cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4v3cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4v3cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v3gm;
                  gmbsr = here->BSIM4v3gmbs;
                  gdsr = here->BSIM4v3gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v3cddb;
                  Cdgr = here->BSIM4v3cdgb;
                  Cdsr = here->BSIM4v3cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v3cgdb;
                  Cggr = here->BSIM4v3cggb;
                  Cgsr = here->BSIM4v3cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v3mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4v3rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v3cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v3cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v3cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v3cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v3cbgb - pParam->BSIM4v3cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v3capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v3capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v3rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v3cbdb - here->BSIM4v3capbd) * omega;
                      xcbsb = (here->BSIM4v3cbsb - here->BSIM4v3capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4v3capbs * omega;

                      xcbdb = here->BSIM4v3cbdb * omega;
                      xcbsb = here->BSIM4v3cbsb * omega;
    
                      xcdbdb = -here->BSIM4v3capbd * omega;
                      xcsbsb = -here->BSIM4v3capbs * omega;
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

                  if (here->BSIM4v3rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v3cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v3cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v3cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v3cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v3cbgb - pParam->BSIM4v3cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v3capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v3capbs + cgso) * omega;
    
                  if (!here->BSIM4v3rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v3cbsb - here->BSIM4v3capbd) * omega;
                      xcbsb = (here->BSIM4v3cbdb - here->BSIM4v3capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v3capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v3cbsb * omega;
                      xcbsb = here->BSIM4v3cbdb * omega;
                      xcdbdb = -here->BSIM4v3capbd * omega;
                      xcsbsb = -here->BSIM4v3capbs * omega;
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


              /*
               * Loading AC matrix
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

                  *(here->BSIM4v3GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v3GPgpPtr) += m * (geltd + xcggbi + gIgtotg);
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v3GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v3GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v3GPbpPtr) += m * (xcgbbi + gIgtotb);
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v3rgateMod == 2)
              {   *(here->BSIM4v3GEgePtr) += m * gcrg;
                  *(here->BSIM4v3GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v3GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v3GEspPtr) += m * gcrgs;
                  *(here->BSIM4v3GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v3GPgePtr) -= m * gcrg;
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v3GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdbr;
		  *(here->BSIM4v3GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsbr;
		  *(here->BSIM4v3GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbbr;
		  *(here->BSIM4v3GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
              }
              else if (here->BSIM4v3rgateMod == 3)
              {   *(here->BSIM4v3GEgePtr) += m * geltd;
                  *(here->BSIM4v3GEgmPtr) -= m * geltd;
                  *(here->BSIM4v3GMgePtr) -= m * geltd;
                  *(here->BSIM4v3GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v3GMgmPtr +1) += m * xcgmgmb;
   
                  *(here->BSIM4v3GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v3GMdpPtr +1) += m * xcgmdb;
                  *(here->BSIM4v3GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v3GMspPtr) += m * gcrgs;
                  *(here->BSIM4v3GMspPtr +1) += m * xcgmsb;
                  *(here->BSIM4v3GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v3GMbpPtr +1) += m * xcgmbb;
   
                  *(here->BSIM4v3DPgmPtr +1) += m * xcdgmb;
                  *(here->BSIM4v3GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v3SPgmPtr +1) += m * xcsgmb;
                  *(here->BSIM4v3BPgmPtr +1) += m * xcbgmb;
   
                  *(here->BSIM4v3GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v3GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v3GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v3GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v3GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbbr;
              }
              else
              {   *(here->BSIM4v3GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v3GPgpPtr) += m * (xcggbi + gIgtotg);
                  *(here->BSIM4v3GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v3GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v3GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v3GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v3GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v3GPbpPtr) += m * (xcgbbi + gIgtotb);
              }

              if (model->BSIM4v3rdsMod)
              {   (*(here->BSIM4v3DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v3DspPtr) += m * gdtots);
                  (*(here->BSIM4v3DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v3SdpPtr) += m * gstotd);
                  (*(here->BSIM4v3SgpPtr) += m * gstotg);
                  (*(here->BSIM4v3SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v3DPdpPtr +1) += m * (xcddbr + gdsi + RevSumi);
              *(here->BSIM4v3DPdpPtr) += m * (gdpr + xcddbi + gdsr + here->BSIM4v3gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd);
              *(here->BSIM4v3DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v3DPgpPtr +1) += m * (xcdgbr + Gmi);
              *(here->BSIM4v3DPgpPtr) += m * (Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg);
              *(here->BSIM4v3DPspPtr +1) += m * (xcdsbr - gdsi - FwdSumi);
              *(here->BSIM4v3DPspPtr) -= m * (gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots);
              *(here->BSIM4v3DPbpPtr +1) += m * (xcdbbr + Gmbsi);
              *(here->BSIM4v3DPbpPtr) -= m * (gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb);

              *(here->BSIM4v3DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v3DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v3SPdpPtr +1) += m * (xcsdbr - gdsi - RevSumi);
              *(here->BSIM4v3SPdpPtr) -= m * (gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd);
              *(here->BSIM4v3SPgpPtr +1) += m * (xcsgbr - Gmi);
              *(here->BSIM4v3SPgpPtr) -= m * (Gmr - xcsgbi + gstotg - gbspg + gIstotg);
              *(here->BSIM4v3SPspPtr +1) += m * (xcssbr + gdsi + FwdSumi);
              *(here->BSIM4v3SPspPtr) += m * (gspr + xcssbi + gdsr + here->BSIM4v3gbs
				     - gstots + FwdSumr + gbspsp - gIstots);
              *(here->BSIM4v3SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v3SPbpPtr +1) += m * (xcsbbr - Gmbsi);
              *(here->BSIM4v3SPbpPtr) -= m * (gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb);

              *(here->BSIM4v3SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v3SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v3BPdpPtr +1) += m * xcbdb;
              *(here->BSIM4v3BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v3BPgpPtr +1) += m * xcbgb;
              *(here->BSIM4v3BPgpPtr) -= m * (here->BSIM4v3gbgs + gIbtotg);
              *(here->BSIM4v3BPspPtr +1) += m * xcbsb;
              *(here->BSIM4v3BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v3BPbpPtr +1) += m * xcbbb;
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
              {   (*(here->BSIM4v3DPdbPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v3DPdbPtr) -= m * here->BSIM4v3gbd);
                  (*(here->BSIM4v3SPsbPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v3SPsbPtr) -= m * here->BSIM4v3gbs);

                  (*(here->BSIM4v3DBdpPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v3DBdpPtr) -= m * here->BSIM4v3gbd);
                  (*(here->BSIM4v3DBdbPtr +1) -= m * xcdbdb);
                  (*(here->BSIM4v3DBdbPtr) += m * (here->BSIM4v3gbd + here->BSIM4v3grbpd 
                                          + here->BSIM4v3grbdb));
                  (*(here->BSIM4v3DBbpPtr) -= m * here->BSIM4v3grbpd);
                  (*(here->BSIM4v3DBbPtr) -= m * here->BSIM4v3grbdb);

                  (*(here->BSIM4v3BPdbPtr) -= m * here->BSIM4v3grbpd);
                  (*(here->BSIM4v3BPbPtr) -= m * here->BSIM4v3grbpb);
                  (*(here->BSIM4v3BPsbPtr) -= m * here->BSIM4v3grbps);
                  (*(here->BSIM4v3BPbpPtr) += m * (here->BSIM4v3grbpd + here->BSIM4v3grbps 
					  + here->BSIM4v3grbpb));
		  /* WDLiu: (-here->BSIM4v3gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v3SBspPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v3SBspPtr) -= m * here->BSIM4v3gbs);
                  (*(here->BSIM4v3SBbpPtr) -= m * here->BSIM4v3grbps);
                  (*(here->BSIM4v3SBbPtr) -= m * here->BSIM4v3grbsb);
                  (*(here->BSIM4v3SBsbPtr +1) -= m * xcsbsb);
                  (*(here->BSIM4v3SBsbPtr) += m * (here->BSIM4v3gbs
					  + here->BSIM4v3grbps + here->BSIM4v3grbsb));

                  (*(here->BSIM4v3BdbPtr) -= m * here->BSIM4v3grbdb);
                  (*(here->BSIM4v3BbpPtr) -= m * here->BSIM4v3grbpb);
                  (*(here->BSIM4v3BsbPtr) -= m * here->BSIM4v3grbsb);
                  (*(here->BSIM4v3BbPtr) += m * (here->BSIM4v3grbsb + here->BSIM4v3grbdb
                                        + here->BSIM4v3grbpb));
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4v3trnqsMod)
           {   (*(here->BSIM4v3QqPtr) += m * 1.0);
               (*(here->BSIM4v3QgpPtr) += 0.0);
               (*(here->BSIM4v3QdpPtr) += 0.0);
               (*(here->BSIM4v3QspPtr) += 0.0);
               (*(here->BSIM4v3QbpPtr) += 0.0);

               (*(here->BSIM4v3DPqPtr) += 0.0);
               (*(here->BSIM4v3SPqPtr) += 0.0);
               (*(here->BSIM4v3GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
