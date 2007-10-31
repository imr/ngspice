/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include "cktdefs.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM4V4acLoad(inModel,ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
BSIM4V4model *model = (BSIM4V4model*)inModel;
BSIM4V4instance *here;

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
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    omega = ckt->CKTomega;
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

              Csd = -(here->BSIM4V4cddb + here->BSIM4V4cgdb + here->BSIM4V4cbdb);
              Csg = -(here->BSIM4V4cdgb + here->BSIM4V4cggb + here->BSIM4V4cbgb);
              Css = -(here->BSIM4V4cdsb + here->BSIM4V4cgsb + here->BSIM4V4cbsb);

              if (here->BSIM4V4acnqsMod)
              {   T0 = omega * here->BSIM4V4taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4V4gm * T2;
                  gmbsr = here->BSIM4V4gmbs * T2;
                  gdsr = here->BSIM4V4gds * T2;

                  gmi = -here->BSIM4V4gm * T3;
                  gmbsi = -here->BSIM4V4gmbs * T3;
                  gdsi = -here->BSIM4V4gds * T3;

                  Cddr = here->BSIM4V4cddb * T2;
                  Cdgr = here->BSIM4V4cdgb * T2;
                  Cdsr = here->BSIM4V4cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4V4cddb * T3 * omega;
                  Cdgi = here->BSIM4V4cdgb * T3 * omega;
                  Cdsi = here->BSIM4V4cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4V4cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4V4cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4V4cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4V4gm;
                  gmbsr = here->BSIM4V4gmbs;
                  gdsr = here->BSIM4V4gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4V4cddb;
                  Cdgr = here->BSIM4V4cdgb;
                  Cdsr = here->BSIM4V4cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4V4cgdb;
                  Cggr = here->BSIM4V4cggb;
                  Cgsr = here->BSIM4V4cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4V4mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4V4rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4V4cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4V4cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4V4cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4V4cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4V4cbgb - pParam->BSIM4V4cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4V4capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4V4capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4V4rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4V4cbdb - here->BSIM4V4capbd) * omega;
                      xcbsb = (here->BSIM4V4cbsb - here->BSIM4V4capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4V4capbs * omega;

                      xcbdb = here->BSIM4V4cbdb * omega;
                      xcbsb = here->BSIM4V4cbsb * omega;
    
                      xcdbdb = -here->BSIM4V4capbd * omega;
                      xcsbsb = -here->BSIM4V4capbs * omega;
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

                  if (here->BSIM4V4rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4V4cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4V4cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4V4cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4V4cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4V4cbgb - pParam->BSIM4V4cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4V4capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4V4capbs + cgso) * omega;
    
                  if (!here->BSIM4V4rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4V4cbsb - here->BSIM4V4capbd) * omega;
                      xcbsb = (here->BSIM4V4cbdb - here->BSIM4V4capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4V4capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4V4cbsb * omega;
                      xcbsb = here->BSIM4V4cbdb * omega;
                      xcdbdb = -here->BSIM4V4capbd * omega;
                      xcsbsb = -here->BSIM4V4capbs * omega;
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


              /*
               * Loading AC matrix
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

                  *(here->BSIM4V4GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4V4GPgpPtr) += m * (geltd + xcggbi + gIgtotg);
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4V4GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4V4GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4V4GPbpPtr) += m * (xcgbbi + gIgtotb);
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4V4rgateMod == 2)
              {   *(here->BSIM4V4GEgePtr) += m * gcrg;
                  *(here->BSIM4V4GEgpPtr) += m * gcrgg;
                  *(here->BSIM4V4GEdpPtr) += m * gcrgd;
                  *(here->BSIM4V4GEspPtr) += m * gcrgs;
                  *(here->BSIM4V4GEbpPtr) += m * gcrgb;

                  *(here->BSIM4V4GPgePtr) -= m * gcrg;
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4V4GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdbr;
		  *(here->BSIM4V4GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsbr;
		  *(here->BSIM4V4GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbbr;
		  *(here->BSIM4V4GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
              }
              else if (here->BSIM4V4rgateMod == 3)
              {   *(here->BSIM4V4GEgePtr) += m * geltd;
                  *(here->BSIM4V4GEgmPtr) -= m * geltd;
                  *(here->BSIM4V4GMgePtr) -= m * geltd;
                  *(here->BSIM4V4GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4V4GMgmPtr +1) += m * xcgmgmb;
   
                  *(here->BSIM4V4GMdpPtr) += m * gcrgd;
                  *(here->BSIM4V4GMdpPtr +1) += m * xcgmdb;
                  *(here->BSIM4V4GMgpPtr) += m * gcrgg;
                  *(here->BSIM4V4GMspPtr) += m * gcrgs;
                  *(here->BSIM4V4GMspPtr +1) += m * xcgmsb;
                  *(here->BSIM4V4GMbpPtr) += m * gcrgb;
                  *(here->BSIM4V4GMbpPtr +1) += m * xcgmbb;
   
                  *(here->BSIM4V4DPgmPtr +1) += m * xcdgmb;
                  *(here->BSIM4V4GPgmPtr) -= m * gcrg;
                  *(here->BSIM4V4SPgmPtr +1) += m * xcsgmb;
                  *(here->BSIM4V4BPgmPtr +1) += m * xcbgmb;
   
                  *(here->BSIM4V4GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4V4GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4V4GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4V4GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4V4GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbbr;
              }
              else
              {   *(here->BSIM4V4GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4V4GPgpPtr) += m * (xcggbi + gIgtotg);
                  *(here->BSIM4V4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4V4GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4V4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4V4GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4V4GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4V4GPbpPtr) += m * (xcgbbi + gIgtotb);
              }

              if (model->BSIM4V4rdsMod)
              {   (*(here->BSIM4V4DgpPtr) += m * gdtotg);
                  (*(here->BSIM4V4DspPtr) += m * gdtots);
                  (*(here->BSIM4V4DbpPtr) += m * gdtotb);
                  (*(here->BSIM4V4SdpPtr) += m * gstotd);
                  (*(here->BSIM4V4SgpPtr) += m * gstotg);
                  (*(here->BSIM4V4SbpPtr) += m * gstotb);
              }

              *(here->BSIM4V4DPdpPtr +1) += m * (xcddbr + gdsi + RevSumi);
              *(here->BSIM4V4DPdpPtr) += m * (gdpr + xcddbi + gdsr + here->BSIM4V4gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd);
              *(here->BSIM4V4DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4V4DPgpPtr +1) += m * (xcdgbr + Gmi);
              *(here->BSIM4V4DPgpPtr) += m * (Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg);
              *(here->BSIM4V4DPspPtr +1) += m * (xcdsbr - gdsi - FwdSumi);
              *(here->BSIM4V4DPspPtr) -= m * (gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots);
              *(here->BSIM4V4DPbpPtr +1) += m * (xcdbbr + Gmbsi);
              *(here->BSIM4V4DPbpPtr) -= m * (gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb);

              *(here->BSIM4V4DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4V4DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4V4SPdpPtr +1) += m * (xcsdbr - gdsi - RevSumi);
              *(here->BSIM4V4SPdpPtr) -= m * (gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd);
              *(here->BSIM4V4SPgpPtr +1) += m * (xcsgbr - Gmi);
              *(here->BSIM4V4SPgpPtr) -= m * (Gmr - xcsgbi + gstotg - gbspg + gIstotg);
              *(here->BSIM4V4SPspPtr +1) += m * (xcssbr + gdsi + FwdSumi);
              *(here->BSIM4V4SPspPtr) += m * (gspr + xcssbi + gdsr + here->BSIM4V4gbs
				     - gstots + FwdSumr + gbspsp - gIstots);
              *(here->BSIM4V4SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4V4SPbpPtr +1) += m * (xcsbbr - Gmbsi);
              *(here->BSIM4V4SPbpPtr) -= m * (gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb);

              *(here->BSIM4V4SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4V4SsPtr) += m * (gspr + gstot);

              *(here->BSIM4V4BPdpPtr +1) += m * xcbdb;
              *(here->BSIM4V4BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4V4BPgpPtr +1) += m * xcbgb;
              *(here->BSIM4V4BPgpPtr) -= m * (here->BSIM4V4gbgs + gIbtotg);
              *(here->BSIM4V4BPspPtr +1) += m * xcbsb;
              *(here->BSIM4V4BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4V4BPbpPtr +1) += m * xcbbb;
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
              {   (*(here->BSIM4V4DPdbPtr +1) += m * xcdbdb);
                  (*(here->BSIM4V4DPdbPtr) -= m * here->BSIM4V4gbd);
                  (*(here->BSIM4V4SPsbPtr +1) += m * xcsbsb);
                  (*(here->BSIM4V4SPsbPtr) -= m * here->BSIM4V4gbs);

                  (*(here->BSIM4V4DBdpPtr +1) += m * xcdbdb);
                  (*(here->BSIM4V4DBdpPtr) -= m * here->BSIM4V4gbd);
                  (*(here->BSIM4V4DBdbPtr +1) -= m * xcdbdb);
                  (*(here->BSIM4V4DBdbPtr) += m * (here->BSIM4V4gbd + here->BSIM4V4grbpd 
                                          + here->BSIM4V4grbdb));
                  (*(here->BSIM4V4DBbpPtr) -= m * here->BSIM4V4grbpd);
                  (*(here->BSIM4V4DBbPtr) -= m * here->BSIM4V4grbdb);

                  (*(here->BSIM4V4BPdbPtr) -= m * here->BSIM4V4grbpd);
                  (*(here->BSIM4V4BPbPtr) -= m * here->BSIM4V4grbpb);
                  (*(here->BSIM4V4BPsbPtr) -= m * here->BSIM4V4grbps);
                  (*(here->BSIM4V4BPbpPtr) += m * (here->BSIM4V4grbpd + here->BSIM4V4grbps 
					  + here->BSIM4V4grbpb));
		  /* WDLiu: (-here->BSIM4V4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4V4SBspPtr +1) += m * xcsbsb);
                  (*(here->BSIM4V4SBspPtr) -= m * here->BSIM4V4gbs);
                  (*(here->BSIM4V4SBbpPtr) -= m * here->BSIM4V4grbps);
                  (*(here->BSIM4V4SBbPtr) -= m * here->BSIM4V4grbsb);
                  (*(here->BSIM4V4SBsbPtr +1) -= m * xcsbsb);
                  (*(here->BSIM4V4SBsbPtr) += m * (here->BSIM4V4gbs
					  + here->BSIM4V4grbps + here->BSIM4V4grbsb));

                  (*(here->BSIM4V4BdbPtr) -= m * here->BSIM4V4grbdb);
                  (*(here->BSIM4V4BbpPtr) -= m * here->BSIM4V4grbpb);
                  (*(here->BSIM4V4BsbPtr) -= m * here->BSIM4V4grbsb);
                  (*(here->BSIM4V4BbPtr) += m * (here->BSIM4V4grbsb + here->BSIM4V4grbdb
                                        + here->BSIM4V4grbpb));
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4V4trnqsMod)
           {   (*(here->BSIM4V4QqPtr) += m * 1.0);
               (*(here->BSIM4V4QgpPtr) += 0.0);
               (*(here->BSIM4V4QdpPtr) += 0.0);
               (*(here->BSIM4V4QspPtr) += 0.0);
               (*(here->BSIM4V4QbpPtr) += 0.0);

               (*(here->BSIM4V4DPqPtr) += 0.0);
               (*(here->BSIM4V4SPqPtr) += 0.0);
               (*(here->BSIM4V4GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
