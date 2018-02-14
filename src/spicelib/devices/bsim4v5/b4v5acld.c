/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v5acLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v5model *model = (BSIM4v5model*)inModel;
BSIM4v5instance *here;

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
struct bsim4v5SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    omega = ckt->CKTomega;
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

              Csd = -(here->BSIM4v5cddb + here->BSIM4v5cgdb + here->BSIM4v5cbdb);
              Csg = -(here->BSIM4v5cdgb + here->BSIM4v5cggb + here->BSIM4v5cbgb);
              Css = -(here->BSIM4v5cdsb + here->BSIM4v5cgsb + here->BSIM4v5cbsb);

              if (here->BSIM4v5acnqsMod)
              {   T0 = omega * here->BSIM4v5taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v5gm * T2;
                  gmbsr = here->BSIM4v5gmbs * T2;
                  gdsr = here->BSIM4v5gds * T2;

                  gmi = -here->BSIM4v5gm * T3;
                  gmbsi = -here->BSIM4v5gmbs * T3;
                  gdsi = -here->BSIM4v5gds * T3;

                  Cddr = here->BSIM4v5cddb * T2;
                  Cdgr = here->BSIM4v5cdgb * T2;
                  Cdsr = here->BSIM4v5cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v5cddb * T3 * omega;
                  Cdgi = here->BSIM4v5cdgb * T3 * omega;
                  Cdsi = here->BSIM4v5cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4v5cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4v5cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4v5cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v5gm;
                  gmbsr = here->BSIM4v5gmbs;
                  gdsr = here->BSIM4v5gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v5cddb;
                  Cdgr = here->BSIM4v5cdgb;
                  Cdsr = here->BSIM4v5cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v5cgdb;
                  Cggr = here->BSIM4v5cggb;
                  Cgsr = here->BSIM4v5cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v5mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4v5rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v5cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v5cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v5cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v5cbgb - pParam->BSIM4v5cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v5capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v5capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v5rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v5cbdb - here->BSIM4v5capbd) * omega;
                      xcbsb = (here->BSIM4v5cbsb - here->BSIM4v5capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4v5capbs * omega;

                      xcbdb = here->BSIM4v5cbdb * omega;
                      xcbsb = here->BSIM4v5cbsb * omega;
    
                      xcdbdb = -here->BSIM4v5capbd * omega;
                      xcsbsb = -here->BSIM4v5capbs * omega;
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

                  if (here->BSIM4v5rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v5cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v5cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v5cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v5cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v5cbgb - pParam->BSIM4v5cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v5capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v5capbs + cgso) * omega;
    
                  if (!here->BSIM4v5rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v5cbsb - here->BSIM4v5capbd) * omega;
                      xcbsb = (here->BSIM4v5cbdb - here->BSIM4v5capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v5capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v5cbsb * omega;
                      xcbsb = here->BSIM4v5cbdb * omega;
                      xcdbdb = -here->BSIM4v5capbd * omega;
                      xcsbsb = -here->BSIM4v5capbs * omega;
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


              /*
               * Loading AC matrix
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

                  *(here->BSIM4v5GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v5GPgpPtr) += m * (geltd + xcggbi + gIgtotg);
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v5GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v5GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v5GPbpPtr) += m * (xcgbbi + gIgtotb);
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v5rgateMod == 2)
              {   *(here->BSIM4v5GEgePtr) += m * gcrg;
                  *(here->BSIM4v5GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v5GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v5GEspPtr) += m * gcrgs;
                  *(here->BSIM4v5GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v5GPgePtr) -= m * gcrg;
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v5GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdbr;
		  *(here->BSIM4v5GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsbr;
		  *(here->BSIM4v5GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbbr;
		  *(here->BSIM4v5GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
              }
              else if (here->BSIM4v5rgateMod == 3)
              {   *(here->BSIM4v5GEgePtr) += m * geltd;
                  *(here->BSIM4v5GEgmPtr) -= m * geltd;
                  *(here->BSIM4v5GMgePtr) -= m * geltd;
                  *(here->BSIM4v5GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v5GMgmPtr +1) += m * xcgmgmb;
   
                  *(here->BSIM4v5GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v5GMdpPtr +1) += m * xcgmdb;
                  *(here->BSIM4v5GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v5GMspPtr) += m * gcrgs;
                  *(here->BSIM4v5GMspPtr +1) += m * xcgmsb;
                  *(here->BSIM4v5GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v5GMbpPtr +1) += m * xcgmbb;
   
                  *(here->BSIM4v5DPgmPtr +1) += m * xcdgmb;
                  *(here->BSIM4v5GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v5SPgmPtr +1) += m * xcsgmb;
                  *(here->BSIM4v5BPgmPtr +1) += m * xcbgmb;
   
                  *(here->BSIM4v5GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v5GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v5GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v5GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v5GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbbr;
              }
              else
              {   *(here->BSIM4v5GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v5GPgpPtr) += m * (xcggbi + gIgtotg);
                  *(here->BSIM4v5GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v5GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v5GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v5GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v5GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v5GPbpPtr) += m * (xcgbbi + gIgtotb);
              }

              if (model->BSIM4v5rdsMod)
              {   (*(here->BSIM4v5DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v5DspPtr) += m * gdtots);
                  (*(here->BSIM4v5DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v5SdpPtr) += m * gstotd);
                  (*(here->BSIM4v5SgpPtr) += m * gstotg);
                  (*(here->BSIM4v5SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v5DPdpPtr +1) += m * (xcddbr + gdsi + RevSumi);
              *(here->BSIM4v5DPdpPtr) += m * (gdpr + xcddbi + gdsr + here->BSIM4v5gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd);
              *(here->BSIM4v5DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v5DPgpPtr +1) += m * (xcdgbr + Gmi);
              *(here->BSIM4v5DPgpPtr) += m * (Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg);
              *(here->BSIM4v5DPspPtr +1) += m * (xcdsbr - gdsi - FwdSumi);
              *(here->BSIM4v5DPspPtr) -= m * (gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots);
              *(here->BSIM4v5DPbpPtr +1) += m * (xcdbbr + Gmbsi);
              *(here->BSIM4v5DPbpPtr) -= m * (gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb);

              *(here->BSIM4v5DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v5DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v5SPdpPtr +1) += m * (xcsdbr - gdsi - RevSumi);
              *(here->BSIM4v5SPdpPtr) -= m * (gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd);
              *(here->BSIM4v5SPgpPtr +1) += m * (xcsgbr - Gmi);
              *(here->BSIM4v5SPgpPtr) -= m * (Gmr - xcsgbi + gstotg - gbspg + gIstotg);
              *(here->BSIM4v5SPspPtr +1) += m * (xcssbr + gdsi + FwdSumi);
              *(here->BSIM4v5SPspPtr) += m * (gspr + xcssbi + gdsr + here->BSIM4v5gbs
				     - gstots + FwdSumr + gbspsp - gIstots);
              *(here->BSIM4v5SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v5SPbpPtr +1) += m * (xcsbbr - Gmbsi);
              *(here->BSIM4v5SPbpPtr) -= m * (gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb);

              *(here->BSIM4v5SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v5SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v5BPdpPtr +1) += m * xcbdb;
              *(here->BSIM4v5BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v5BPgpPtr +1) += m * xcbgb;
              *(here->BSIM4v5BPgpPtr) -= m * (here->BSIM4v5gbgs + gIbtotg);
              *(here->BSIM4v5BPspPtr +1) += m * xcbsb;
              *(here->BSIM4v5BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v5BPbpPtr +1) += m * xcbbb;
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
              {   (*(here->BSIM4v5DPdbPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v5DPdbPtr) -= m * here->BSIM4v5gbd);
                  (*(here->BSIM4v5SPsbPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v5SPsbPtr) -= m * here->BSIM4v5gbs);

                  (*(here->BSIM4v5DBdpPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v5DBdpPtr) -= m * here->BSIM4v5gbd);
                  (*(here->BSIM4v5DBdbPtr +1) -= m * xcdbdb);
                  (*(here->BSIM4v5DBdbPtr) += m * (here->BSIM4v5gbd + here->BSIM4v5grbpd 
                                          + here->BSIM4v5grbdb));
                  (*(here->BSIM4v5DBbpPtr) -= m * here->BSIM4v5grbpd);
                  (*(here->BSIM4v5DBbPtr) -= m * here->BSIM4v5grbdb);

                  (*(here->BSIM4v5BPdbPtr) -= m * here->BSIM4v5grbpd);
                  (*(here->BSIM4v5BPbPtr) -= m * here->BSIM4v5grbpb);
                  (*(here->BSIM4v5BPsbPtr) -= m * here->BSIM4v5grbps);
                  (*(here->BSIM4v5BPbpPtr) += m * (here->BSIM4v5grbpd + here->BSIM4v5grbps 
					  + here->BSIM4v5grbpb));
		  /* WDLiu: (-here->BSIM4v5gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v5SBspPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v5SBspPtr) -= m * here->BSIM4v5gbs);
                  (*(here->BSIM4v5SBbpPtr) -= m * here->BSIM4v5grbps);
                  (*(here->BSIM4v5SBbPtr) -= m * here->BSIM4v5grbsb);
                  (*(here->BSIM4v5SBsbPtr +1) -= m * xcsbsb);
                  (*(here->BSIM4v5SBsbPtr) += m * (here->BSIM4v5gbs
					  + here->BSIM4v5grbps + here->BSIM4v5grbsb));

                  (*(here->BSIM4v5BdbPtr) -= m * here->BSIM4v5grbdb);
                  (*(here->BSIM4v5BbpPtr) -= m * here->BSIM4v5grbpb);
                  (*(here->BSIM4v5BsbPtr) -= m * here->BSIM4v5grbsb);
                  (*(here->BSIM4v5BbPtr) += m * (here->BSIM4v5grbsb + here->BSIM4v5grbdb
                                        + here->BSIM4v5grbpb));
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4v5trnqsMod)
           {   (*(here->BSIM4v5QqPtr) += m * 1.0);
               (*(here->BSIM4v5QgpPtr) += 0.0);
               (*(here->BSIM4v5QdpPtr) += 0.0);
               (*(here->BSIM4v5QspPtr) += 0.0);
               (*(here->BSIM4v5QbpPtr) += 0.0);

               (*(here->BSIM4v5DPqPtr) += 0.0);
               (*(here->BSIM4v5SPqPtr) += 0.0);
               (*(here->BSIM4v5GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
