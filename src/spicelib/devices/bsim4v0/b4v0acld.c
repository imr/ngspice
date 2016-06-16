/**** BSIM4v0.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4v0.0.0.
 * Authors: Weidong Liu, Xiaodong Jin, Kanyu M. Cao, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v0acLoad(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register BSIM4v0model *model = (BSIM4v0model*)inModel;
register BSIM4v0instance *here;

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
struct BSIM4v0SizeDependParam *pParam;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->BSIM4v0nextModel) 
    {    for (here = model->BSIM4v0instances; here!= NULL;
              here = here->BSIM4v0nextInstance) 
	 {    pParam = here->pParam;
              capbd = here->BSIM4v0capbd;
              capbs = here->BSIM4v0capbs;
              cgso = here->BSIM4v0cgso;
              cgdo = here->BSIM4v0cgdo;
              cgbo = pParam->BSIM4v0cgbo;

              Csd = -(here->BSIM4v0cddb + here->BSIM4v0cgdb + here->BSIM4v0cbdb);
              Csg = -(here->BSIM4v0cdgb + here->BSIM4v0cggb + here->BSIM4v0cbgb);
              Css = -(here->BSIM4v0cdsb + here->BSIM4v0cgsb + here->BSIM4v0cbsb);

              if (here->BSIM4v0acnqsMod)
              {   T0 = omega * here->BSIM4v0taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v0gm * T2;
                  gmbsr = here->BSIM4v0gmbs * T2;
                  gdsr = here->BSIM4v0gds * T2;

                  gmi = -here->BSIM4v0gm * T3;
                  gmbsi = -here->BSIM4v0gmbs * T3;
                  gdsi = -here->BSIM4v0gds * T3;

                  Cddr = here->BSIM4v0cddb * T2;
                  Cdgr = here->BSIM4v0cdgb * T2;
                  Cdsr = here->BSIM4v0cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v0cddb * T3 * omega;
                  Cdgi = here->BSIM4v0cdgb * T3 * omega;
                  Cdsi = here->BSIM4v0cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4v0cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4v0cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4v0cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v0gm;
                  gmbsr = here->BSIM4v0gmbs;
                  gdsr = here->BSIM4v0gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v0cddb;
                  Cdgr = here->BSIM4v0cdgb;
                  Cdsr = here->BSIM4v0cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v0cgdb;
                  Cggr = here->BSIM4v0cggb;
                  Cgsr = here->BSIM4v0cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v0mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

                  gbbdp = -(here->BSIM4v0gbds + here->BSIM4v0ggidld);
                  gbbsp = here->BSIM4v0gbds + here->BSIM4v0gbgs + here->BSIM4v0gbbs
                        - here->BSIM4v0ggidls;
                  gbdpg = here->BSIM4v0gbgs + here->BSIM4v0ggidlg;
                  gbdpdp = here->BSIM4v0gbds + here->BSIM4v0ggidld;
                  gbdpb = here->BSIM4v0gbbs + here->BSIM4v0ggidlb;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb) + here->BSIM4v0ggidls;

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v0igcMod)
                  {   gIstotg = here->BSIM4v0gIgsg + here->BSIM4v0gIgcsg;
                      gIstotd = here->BSIM4v0gIgcsd;
                      gIstots = here->BSIM4v0gIgss + here->BSIM4v0gIgcss;
                      gIstotb = here->BSIM4v0gIgcsb;

                      gIdtotg = here->BSIM4v0gIgdg + here->BSIM4v0gIgcdg;
                      gIdtotd = here->BSIM4v0gIgdd + here->BSIM4v0gIgcdd;
                      gIdtots = here->BSIM4v0gIgcds;
                      gIdtotb = here->BSIM4v0gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v0igbMod)
                  {   gIbtotg = here->BSIM4v0gIgbg;
                      gIbtotd = here->BSIM4v0gIgbd;
                      gIbtots = here->BSIM4v0gIgbs;
                      gIbtotb = here->BSIM4v0gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v0igcMod != 0) || (model->BSIM4v0igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v0rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  else if (here->BSIM4v0rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  if (here->BSIM4v0rgateMod > 1)
                  {   gcrgd = here->BSIM4v0gcrgd * T0;
                      gcrgg = here->BSIM4v0gcrgg * T0;
                      gcrgs = here->BSIM4v0gcrgs * T0;
                      gcrgb = here->BSIM4v0gcrgb * T0;
                      gcrgg -= here->BSIM4v0gcrg;
                      gcrg = here->BSIM4v0gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v0rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v0cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v0cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v0cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v0cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v0cbgb - pParam->BSIM4v0cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v0capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v0capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v0rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v0cbdb - here->BSIM4v0capbd) * omega;
                      xcbsb = (here->BSIM4v0cbsb - here->BSIM4v0capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4v0capbs * omega;

                      xcbdb = here->BSIM4v0cbdb * omega;
                      xcbsb = here->BSIM4v0cbsb * omega;
    
                      xcdbdb = -here->BSIM4v0capbd * omega;
                      xcsbsb = -here->BSIM4v0capbs * omega;
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

                  gbbsp = -(here->BSIM4v0gbds + here->BSIM4v0ggidld);
                  gbbdp = here->BSIM4v0gbds + here->BSIM4v0gbgs + here->BSIM4v0gbbs
                        - here->BSIM4v0ggidls;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v0gbgs + here->BSIM4v0ggidlg;
                  gbspsp = here->BSIM4v0gbds + here->BSIM4v0ggidld;
                  gbspb = here->BSIM4v0gbbs + here->BSIM4v0ggidlb;
                  gbspdp = -(gbspg + gbspsp + gbspb) + here->BSIM4v0ggidls;

                  if (model->BSIM4v0igcMod)
                  {   gIstotg = here->BSIM4v0gIgsg + here->BSIM4v0gIgcdg;
                      gIstotd = here->BSIM4v0gIgcds;
                      gIstots = here->BSIM4v0gIgss + here->BSIM4v0gIgcdd;
                      gIstotb = here->BSIM4v0gIgcdb;

                      gIdtotg = here->BSIM4v0gIgdg + here->BSIM4v0gIgcsg;
                      gIdtotd = here->BSIM4v0gIgdd + here->BSIM4v0gIgcss;
                      gIdtots = here->BSIM4v0gIgcsd;
                      gIdtotb = here->BSIM4v0gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v0igbMod)
                  {   gIbtotg = here->BSIM4v0gIgbg;
                      gIbtotd = here->BSIM4v0gIgbs;
                      gIbtots = here->BSIM4v0gIgbd;
                      gIbtotb = here->BSIM4v0gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v0igcMod != 0) || (model->BSIM4v0igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v0rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  else if (here->BSIM4v0rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  if (here->BSIM4v0rgateMod > 1)
                  {   gcrgd = here->BSIM4v0gcrgs * T0;
                      gcrgg = here->BSIM4v0gcrgg * T0;
                      gcrgs = here->BSIM4v0gcrgd * T0;
                      gcrgb = here->BSIM4v0gcrgb * T0;
                      gcrgg -= here->BSIM4v0gcrg;
                      gcrg = here->BSIM4v0gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v0rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v0cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v0cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v0cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v0cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v0cbgb - pParam->BSIM4v0cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v0capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v0capbs + cgso) * omega;
    
                  if (!here->BSIM4v0rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v0cbsb - here->BSIM4v0capbd) * omega;
                      xcbsb = (here->BSIM4v0cbdb - here->BSIM4v0capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v0capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v0cbsb * omega;
                      xcbsb = here->BSIM4v0cbdb * omega;
                      xcdbdb = -here->BSIM4v0capbd * omega;
                      xcsbsb = -here->BSIM4v0capbs * omega;
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

              if (model->BSIM4v0rdsMod == 1)
              {   gstot = here->BSIM4v0gstot;
                  gstotd = here->BSIM4v0gstotd;
                  gstotg = here->BSIM4v0gstotg;
                  gstots = here->BSIM4v0gstots - gstot;
                  gstotb = here->BSIM4v0gstotb;

                  gdtot = here->BSIM4v0gdtot;
                  gdtotd = here->BSIM4v0gdtotd - gdtot;
                  gdtotg = here->BSIM4v0gdtotg;
                  gdtots = here->BSIM4v0gdtots;
                  gdtotb = here->BSIM4v0gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


              /*
               * Loading AC matrix
               */

              if (!model->BSIM4v0rdsMod)
              {   gdpr = here->BSIM4v0drainConductance;
                  gspr = here->BSIM4v0sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v0rbodyMod)
              {   gjbd = here->BSIM4v0gbd;
                  gjbs = here->BSIM4v0gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v0grgeltd;

              if (here->BSIM4v0rgateMod == 1)
              {   *(here->BSIM4v0GEgePtr) += geltd;
                  *(here->BSIM4v0GPgePtr) -= geltd;
                  *(here->BSIM4v0GEgpPtr) -= geltd;

                  *(here->BSIM4v0GPgpPtr +1) += xcggbr;
		  *(here->BSIM4v0GPgpPtr) += geltd + xcggbi + gIgtotg;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v0GPdpPtr) += xcgdbi + gIgtotd;
                  *(here->BSIM4v0GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v0GPspPtr) += xcgsbi + gIgtots;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbbr;
                  *(here->BSIM4v0GPbpPtr) += xcgbbi + gIgtotb;
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v0rgateMod == 2)
              {   *(here->BSIM4v0GEgePtr) += gcrg;
                  *(here->BSIM4v0GEgpPtr) += gcrgg;
                  *(here->BSIM4v0GEdpPtr) += gcrgd;
                  *(here->BSIM4v0GEspPtr) += gcrgs;
                  *(here->BSIM4v0GEbpPtr) += gcrgb;

                  *(here->BSIM4v0GPgePtr) -= gcrg;
                  *(here->BSIM4v0GPgpPtr +1) += xcggbr;
		  *(here->BSIM4v0GPgpPtr) -= gcrgg - xcggbi - gIgtotg;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdbr;
		  *(here->BSIM4v0GPdpPtr) -= gcrgd - xcgdbi - gIgtotd;
                  *(here->BSIM4v0GPspPtr +1) += xcgsbr;
		  *(here->BSIM4v0GPspPtr) -= gcrgs - xcgsbi - gIgtots;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbbr;
		  *(here->BSIM4v0GPbpPtr) -= gcrgb - xcgbbi - gIgtotb;
              }
              else if (here->BSIM4v0rgateMod == 3)
              {   *(here->BSIM4v0GEgePtr) += geltd;
                  *(here->BSIM4v0GEgmPtr) -= geltd;
                  *(here->BSIM4v0GMgePtr) -= geltd;
                  *(here->BSIM4v0GMgmPtr) += geltd + gcrg;
                  *(here->BSIM4v0GMgmPtr +1) += xcgmgmb;
   
                  *(here->BSIM4v0GMdpPtr) += gcrgd;
                  *(here->BSIM4v0GMdpPtr +1) += xcgmdb;
                  *(here->BSIM4v0GMgpPtr) += gcrgg;
                  *(here->BSIM4v0GMspPtr) += gcrgs;
                  *(here->BSIM4v0GMspPtr +1) += xcgmsb;
                  *(here->BSIM4v0GMbpPtr) += gcrgb;
                  *(here->BSIM4v0GMbpPtr +1) += xcgmbb;
   
                  *(here->BSIM4v0DPgmPtr +1) += xcdgmb;
                  *(here->BSIM4v0GPgmPtr) -= gcrg;
                  *(here->BSIM4v0SPgmPtr +1) += xcsgmb;
                  *(here->BSIM4v0BPgmPtr +1) += xcbgmb;
   
                  *(here->BSIM4v0GPgpPtr) -= gcrgg - xcggbi - gIgtotg;
                  *(here->BSIM4v0GPgpPtr +1) += xcggbr;
                  *(here->BSIM4v0GPdpPtr) -= gcrgd - xcgdbi - gIgtotd;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v0GPspPtr) -= gcrgs - xcgsbi - gIgtots;
                  *(here->BSIM4v0GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v0GPbpPtr) -= gcrgb - xcgbbi - gIgtotb;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbbr;
              }
              else
              {   *(here->BSIM4v0GPgpPtr +1) += xcggbr;
                  *(here->BSIM4v0GPgpPtr) += xcggbi + gIgtotg;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdbr;
                  *(here->BSIM4v0GPdpPtr) += xcgdbi + gIgtotd;
                  *(here->BSIM4v0GPspPtr +1) += xcgsbr;
                  *(here->BSIM4v0GPspPtr) += xcgsbi + gIgtots;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbbr;
                  *(here->BSIM4v0GPbpPtr) += xcgbbi + gIgtotb;
              }

              if (model->BSIM4v0rdsMod)
              {   (*(here->BSIM4v0DgpPtr) += gdtotg);
                  (*(here->BSIM4v0DspPtr) += gdtots);
                  (*(here->BSIM4v0DbpPtr) += gdtotb);
                  (*(here->BSIM4v0SdpPtr) += gstotd);
                  (*(here->BSIM4v0SgpPtr) += gstotg);
                  (*(here->BSIM4v0SbpPtr) += gstotb);
              }

              *(here->BSIM4v0DPdpPtr +1) += xcddbr + gdsi + RevSumi;
              *(here->BSIM4v0DPdpPtr) += gdpr + xcddbi + gdsr + here->BSIM4v0gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd;
              *(here->BSIM4v0DPdPtr) -= gdpr + gdtot;
              *(here->BSIM4v0DPgpPtr +1) += xcdgbr + Gmi;
              *(here->BSIM4v0DPgpPtr) += Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg;
              *(here->BSIM4v0DPspPtr +1) += xcdsbr - gdsi - FwdSumi;
              *(here->BSIM4v0DPspPtr) -= gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots;
              *(here->BSIM4v0DPbpPtr +1) += xcdbbr + Gmbsi;
              *(here->BSIM4v0DPbpPtr) -= gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb;

              *(here->BSIM4v0DdpPtr) -= gdpr - gdtotd;
              *(here->BSIM4v0DdPtr) += gdpr + gdtot;

              *(here->BSIM4v0SPdpPtr +1) += xcsdbr - gdsi - RevSumi;
              *(here->BSIM4v0SPdpPtr) -= gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd;
              *(here->BSIM4v0SPgpPtr +1) += xcsgbr - Gmi;
              *(here->BSIM4v0SPgpPtr) -= Gmr - xcsgbi + gstotg - gbspg + gIstotg;
              *(here->BSIM4v0SPspPtr +1) += xcssbr + gdsi + FwdSumi;
              *(here->BSIM4v0SPspPtr) += gspr + xcssbi + gdsr + here->BSIM4v0gbs
				     - gstots + FwdSumr + gbspsp - gIstots;
              *(here->BSIM4v0SPsPtr) -= gspr + gstot;
              *(here->BSIM4v0SPbpPtr +1) += xcsbbr - Gmbsi;
              *(here->BSIM4v0SPbpPtr) -= gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb;

              *(here->BSIM4v0SspPtr) -= gspr - gstots;
              *(here->BSIM4v0SsPtr) += gspr + gstot;

              *(here->BSIM4v0BPdpPtr +1) += xcbdb;
              *(here->BSIM4v0BPdpPtr) -= gjbd - gbbdp + gIbtotd;
              *(here->BSIM4v0BPgpPtr +1) += xcbgb;
              *(here->BSIM4v0BPgpPtr) -= here->BSIM4v0gbgs + here->BSIM4v0ggidlg + gIbtotg;
              *(here->BSIM4v0BPspPtr +1) += xcbsb;
              *(here->BSIM4v0BPspPtr) -= gjbs - gbbsp + gIbtots;
              *(here->BSIM4v0BPbpPtr +1) += xcbbb;
              *(here->BSIM4v0BPbpPtr) += gjbd + gjbs - here->BSIM4v0gbbs
				     - here->BSIM4v0ggidlb - gIbtotb;

              if (here->BSIM4v0rbodyMod)
              {   (*(here->BSIM4v0DPdbPtr +1) += xcdbdb);
                  (*(here->BSIM4v0DPdbPtr) -= here->BSIM4v0gbd);
                  (*(here->BSIM4v0SPsbPtr +1) += xcsbsb);
                  (*(here->BSIM4v0SPsbPtr) -= here->BSIM4v0gbs);

                  (*(here->BSIM4v0DBdpPtr +1) += xcdbdb);
                  (*(here->BSIM4v0DBdpPtr) -= here->BSIM4v0gbd);
                  (*(here->BSIM4v0DBdbPtr +1) -= xcdbdb);
                  (*(here->BSIM4v0DBdbPtr) += here->BSIM4v0gbd + here->BSIM4v0grbpd 
                                          + here->BSIM4v0grbdb);
                  (*(here->BSIM4v0DBbpPtr) -= here->BSIM4v0grbpd);
                  (*(here->BSIM4v0DBbPtr) -= here->BSIM4v0grbdb);

                  (*(here->BSIM4v0BPdbPtr) -= here->BSIM4v0grbpd);
                  (*(here->BSIM4v0BPbPtr) -= here->BSIM4v0grbpb);
                  (*(here->BSIM4v0BPsbPtr) -= here->BSIM4v0grbps);
                  (*(here->BSIM4v0BPbpPtr) += here->BSIM4v0grbpd + here->BSIM4v0grbps 
					  + here->BSIM4v0grbpb);
		  /* WDLiu: (-here->BSIM4v0gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v0SBspPtr +1) += xcsbsb);
                  (*(here->BSIM4v0SBspPtr) -= here->BSIM4v0gbs);
                  (*(here->BSIM4v0SBbpPtr) -= here->BSIM4v0grbps);
                  (*(here->BSIM4v0SBbPtr) -= here->BSIM4v0grbsb);
                  (*(here->BSIM4v0SBsbPtr +1) -= xcsbsb);
                  (*(here->BSIM4v0SBsbPtr) += here->BSIM4v0gbs
					  + here->BSIM4v0grbps + here->BSIM4v0grbsb);

                  (*(here->BSIM4v0BdbPtr) -= here->BSIM4v0grbdb);
                  (*(here->BSIM4v0BbpPtr) -= here->BSIM4v0grbpb);
                  (*(here->BSIM4v0BsbPtr) -= here->BSIM4v0grbsb);
                  (*(here->BSIM4v0BbPtr) += here->BSIM4v0grbsb + here->BSIM4v0grbdb
                                        + here->BSIM4v0grbpb);
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4v0trnqsMod)
           {   (*(here->BSIM4v0QqPtr) += 1.0);
               (*(here->BSIM4v0QgpPtr) += 0.0);
               (*(here->BSIM4v0QdpPtr) += 0.0);
               (*(here->BSIM4v0QspPtr) += 0.0);
               (*(here->BSIM4v0QbpPtr) += 0.0);

               (*(here->BSIM4v0DPqPtr) += 0.0);
               (*(here->BSIM4v0SPqPtr) += 0.0);
               (*(here->BSIM4v0GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
