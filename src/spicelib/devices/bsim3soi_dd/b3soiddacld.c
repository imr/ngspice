/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddacld.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/* 
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;
int selfheat;
double xcggb, xcgdb, xcgsb, xcgeb, xcgT;
double xcdgb, xcddb, xcdsb, xcdeb, xcdT;
double xcsgb, xcsdb, xcssb, xcseb, xcsT;
double xcbgb, xcbdb, xcbsb, xcbeb, xcbT;
double xcegb, xcedb, xcesb, xceeb, xceT;
double gdpr, gspr, gds;
double cggb, cgdb, cgsb, cgeb, cgT;
double cdgb, cddb, cdsb, cdeb, cdT;
double cbgb, cbdb, cbsb, cbeb, cbT;
double cegb, cedb, cesb, ceeb, ceT;
double GSoverlapCap, GDoverlapCap, GEoverlapCap, FwdSum, RevSum, Gm, Gmbs, Gme, GmT;
double omega;
double dxpart, sxpart;
double gbbg, gbbdp, gbbb, gbbe, gbbp, gbbsp, gbbT;
double gddpg, gddpdp, gddpsp, gddpb, gddpe, gddpT;
double gsspg, gsspdp, gsspsp, gsspb, gsspe, gsspT;
double gppg, gppdp, gppb, gppe, gppp, gppsp, gppT;
double xcTt, cTt, gTtt, gTtg, gTtb, gTte, gTtdp, gTtsp;

FILE *fpdebug = NULL;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = B3SOIDDnextModel(model)) 
    {    

      for (here = B3SOIDDinstances(model); here!= NULL;
              here = B3SOIDDnextInstance(here)) 
	 {    
	      selfheat = (model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0);
              if (here->B3SOIDDdebugMod > 2)
              {
                 fpdebug = fopen("b3soiddac.log", "a");
                 fprintf(fpdebug, ".......omega=%.5e\n", omega);
              }
              if (here->B3SOIDDmode >= 0) 
	      {   Gm = here->B3SOIDDgm;
		  Gmbs = here->B3SOIDDgmbs;
	          Gme = here->B3SOIDDgme;
                  GmT = model->B3SOIDDtype * here->B3SOIDDgmT;
		  FwdSum = Gm + Gmbs + Gme;
		  RevSum = 0.0;

                  cbgb = here->B3SOIDDcbgb;
                  cbsb = here->B3SOIDDcbsb;
                  cbdb = here->B3SOIDDcbdb;
                  cbeb = here->B3SOIDDcbeb;
                  cbT  = model->B3SOIDDtype * here->B3SOIDDcbT;

                  cegb = here->B3SOIDDcegb;
                  cesb = here->B3SOIDDcesb;
                  cedb = here->B3SOIDDcedb;
                  ceeb = here->B3SOIDDceeb;
                  ceT  = model->B3SOIDDtype * here->B3SOIDDceT;

                  cggb = here->B3SOIDDcggb;
                  cgsb = here->B3SOIDDcgsb;
                  cgdb = here->B3SOIDDcgdb;
                  cgeb = here->B3SOIDDcgeb;
                  cgT  = model->B3SOIDDtype * here->B3SOIDDcgT;

                  cdgb = here->B3SOIDDcdgb;
                  cdsb = here->B3SOIDDcdsb;
                  cddb = here->B3SOIDDcddb;
                  cdeb = here->B3SOIDDcdeb;
                  cdT  = model->B3SOIDDtype * here->B3SOIDDcdT;

                  cTt = here->pParam->B3SOIDDcth;

                  gbbg  = -here->B3SOIDDgbgs;
                  gbbdp = -here->B3SOIDDgbds;
                  gbbb  = -here->B3SOIDDgbbs;
                  gbbe  = -here->B3SOIDDgbes;
                  gbbp  = -here->B3SOIDDgbps;
                  gbbT  = -model->B3SOIDDtype * here->B3SOIDDgbT;
                  gbbsp = - ( gbbg + gbbdp + gbbb + gbbe + gbbp);

                  gddpg  = -here->B3SOIDDgjdg;
                  gddpdp = -here->B3SOIDDgjdd;
                  gddpb  = -here->B3SOIDDgjdb;
                  gddpe  = -here->B3SOIDDgjde;
                  gddpT  = -model->B3SOIDDtype * here->B3SOIDDgjdT;
                  gddpsp = - ( gddpg + gddpdp + gddpb + gddpe);

                  gsspg  = -here->B3SOIDDgjsg;
                  gsspdp = -here->B3SOIDDgjsd;
                  gsspb  = -here->B3SOIDDgjsb;
                  gsspe  = 0.0;
                  gsspT  = -model->B3SOIDDtype * here->B3SOIDDgjsT;
                  gsspsp = - (gsspg + gsspdp + gsspb + gsspe);

                  gppg = -here->B3SOIDDgbpgs;
                  gppdp = -here->B3SOIDDgbpds;
             	  gppb = -here->B3SOIDDgbpbs;
                  gppe = -here->B3SOIDDgbpes;
                  gppp = -here->B3SOIDDgbpps;
                  gppT = -model->B3SOIDDtype * here->B3SOIDDgbpT;
                  gppsp = - (gppg + gppdp + gppb + gppe + gppp);

                  gTtg  = here->B3SOIDDgtempg;
                  gTtb  = here->B3SOIDDgtempb;
                  gTte  = here->B3SOIDDgtempe;
                  gTtdp = here->B3SOIDDgtempd;
                  gTtt  = here->B3SOIDDgtempT;
                  gTtsp = - (gTtg + gTtb + gTte + gTtdp);

                  sxpart = 0.6;
                  dxpart = 0.4;

              } 
	      else
	      {   Gm = -here->B3SOIDDgm;
		  Gmbs = -here->B3SOIDDgmbs;
                  Gme = -here->B3SOIDDgme;
                  GmT = -model->B3SOIDDtype * here->B3SOIDDgmT;
		  FwdSum = 0.0;
		  RevSum = -Gm - Gmbs - Gme;

                  cdgb = - (here->B3SOIDDcdgb + here->B3SOIDDcggb + here->B3SOIDDcbgb
                          + here->B3SOIDDcegb);
                  cdsb = - (here->B3SOIDDcddb + here->B3SOIDDcgdb + here->B3SOIDDcbdb
                          + here->B3SOIDDcedb);
                  cddb = - (here->B3SOIDDcdsb + here->B3SOIDDcgsb + here->B3SOIDDcbsb
                          + here->B3SOIDDcesb);
                  cdeb = - (here->B3SOIDDcdeb + here->B3SOIDDcgeb + here->B3SOIDDcbeb
                          + here->B3SOIDDceeb);
                  cdT  = - model->B3SOIDDtype * (here->B3SOIDDcgT + here->B3SOIDDcbT
                         + here->B3SOIDDcdT + here->B3SOIDDceT);

                  cegb = here->B3SOIDDcegb;
                  cesb = here->B3SOIDDcedb;
                  cedb = here->B3SOIDDcesb;
                  ceeb = here->B3SOIDDceeb;
                  ceT  = model->B3SOIDDtype * here->B3SOIDDceT;

                  cggb = here->B3SOIDDcggb;
                  cgsb = here->B3SOIDDcgdb;
                  cgdb = here->B3SOIDDcgsb;
                  cgeb = here->B3SOIDDcgeb;
                  cgT  = model->B3SOIDDtype * here->B3SOIDDcgT;

                  cbgb = here->B3SOIDDcbgb;
                  cbsb = here->B3SOIDDcbdb;
                  cbdb = here->B3SOIDDcbsb;
                  cbeb = here->B3SOIDDcbeb;
                  cbT  = model->B3SOIDDtype * here->B3SOIDDcbT;

                  cTt = here->pParam->B3SOIDDcth;

                  gbbg  = -here->B3SOIDDgbgs;
                  gbbb  = -here->B3SOIDDgbbs;
                  gbbe  = -here->B3SOIDDgbes;
                  gbbp  = -here->B3SOIDDgbps;
                  gbbsp = -here->B3SOIDDgbds;
                  gbbT  = -model->B3SOIDDtype * here->B3SOIDDgbT;
                  gbbdp = - ( gbbg + gbbsp + gbbb + gbbe + gbbp);

                  gddpg  = -here->B3SOIDDgjsg;
                  gddpsp = -here->B3SOIDDgjsd;
                  gddpb  = -here->B3SOIDDgjsb;
                  gddpe  = 0.0;
                  gddpT  = -model->B3SOIDDtype * here->B3SOIDDgjsT;
                  gddpdp = - (gddpg + gddpsp + gddpb + gddpe);

                  gsspg  = -here->B3SOIDDgjdg;
                  gsspsp = -here->B3SOIDDgjdd;
                  gsspb  = -here->B3SOIDDgjdb;
                  gsspe  = -here->B3SOIDDgjde;
                  gsspT  = -model->B3SOIDDtype * here->B3SOIDDgjdT;
                  gsspdp = - ( gsspg + gsspsp + gsspb + gsspe);

                  gppg = -here->B3SOIDDgbpgs;
                  gppsp = -here->B3SOIDDgbpds;
                  gppb = -here->B3SOIDDgbpbs;
                  gppe = -here->B3SOIDDgbpes;
                  gppp = -here->B3SOIDDgbpps;
                  gppT = -model->B3SOIDDtype * here->B3SOIDDgbpT;
                  gppdp = - (gppg + gppsp + gppb + gppe + gppp);

                  gTtt = here->B3SOIDDgtempT;
                  gTtg = here->B3SOIDDgtempg;
                  gTtb = here->B3SOIDDgtempb;
                  gTte = here->B3SOIDDgtempe;
                  gTtdp = here->B3SOIDDgtempd;
                  gTtsp = - (gTtt + gTtg + gTtb + gTte + gTtdp);

                  gTtg  = here->B3SOIDDgtempg;
                  gTtb  = here->B3SOIDDgtempb;
                  gTte  = here->B3SOIDDgtempe;
                  gTtsp = here->B3SOIDDgtempd;
                  gTtt  = here->B3SOIDDgtempT;
                  gTtdp = - (gTtg + gTtb + gTte + gTtsp);

                  sxpart = 0.6;
                  sxpart = 0.4;
                  dxpart = 0.6;
              }

              gdpr=here->B3SOIDDdrainConductance;
              gspr=here->B3SOIDDsourceConductance;
              gds= here->B3SOIDDgds;

	      GSoverlapCap = here->B3SOIDDcgso;
	      GDoverlapCap = here->B3SOIDDcgdo;
	      GEoverlapCap = here->pParam->B3SOIDDcgeo;

              xcegb = (cegb - GEoverlapCap) * omega;
              xcedb = cedb * omega;
              xcesb = cesb * omega;
              xceeb = (ceeb + GEoverlapCap) * omega;
              xceT  =  ceT * omega;

              xcggb = (cggb + GDoverlapCap + GSoverlapCap + GEoverlapCap)
		    * omega;
              xcgdb = (cgdb - GDoverlapCap ) * omega;
              xcgsb = (cgsb - GSoverlapCap) * omega;
              xcgeb = (cgeb - GEoverlapCap) * omega;
              xcgT  = cgT * omega;

              xcdgb = (cdgb - GDoverlapCap) * omega;
              xcddb = (cddb + GDoverlapCap) * omega;
              xcdsb = cdsb * omega;
              xcdeb = cdeb * omega;
              xcdT  = cdT * omega;

              xcsgb = -(cggb + cbgb + cdgb + cegb + GSoverlapCap) * omega;
              xcsdb = -(cgdb + cbdb + cddb + cedb) * omega;
              xcssb = (GSoverlapCap - (cgsb + cbsb + cdsb + cesb)) * omega;
              xcseb = -(cgeb + cbeb + cdeb + ceeb) * omega;
              xcsT  = -(cgT + cbT + cdT + ceT) * omega;

              xcbgb = cbgb * omega;
              xcbdb = cbdb * omega;
              xcbsb = cbsb * omega;
              xcbeb = cbeb * omega;
              xcbT  = cbT * omega;
	      
              xcTt = cTt * omega;

              m = here->B3SOIDDm;

              *(here->B3SOIDDEgPtr  +1) += m * xcegb;
              *(here->B3SOIDDEdpPtr  +1) += m * xcedb;
              *(here->B3SOIDDEspPtr  +1) += m * xcesb;
              *(here->B3SOIDDGePtr +1)  += m * xcgeb;
              *(here->B3SOIDDDPePtr +1) += m * xcdeb;
              *(here->B3SOIDDSPePtr +1) += m * xcseb;
              
              *(here->B3SOIDDEePtr  +1) += m * xceeb;

              *(here->B3SOIDDGgPtr  +1) += m * xcggb;
              *(here->B3SOIDDGdpPtr +1) += m * xcgdb;
              *(here->B3SOIDDGspPtr +1) += m * xcgsb;

              *(here->B3SOIDDDPgPtr +1) += m * xcdgb;
              *(here->B3SOIDDDPdpPtr +1) += m * xcddb;
              *(here->B3SOIDDDPspPtr +1) += m * xcdsb;

              *(here->B3SOIDDSPgPtr +1) += m * xcsgb;
              *(here->B3SOIDDSPdpPtr +1) += m * xcsdb;
              *(here->B3SOIDDSPspPtr +1) += m * xcssb;

              *(here->B3SOIDDBePtr +1) += m * xcbeb;
              *(here->B3SOIDDBgPtr +1) += m * xcbgb;
              *(here->B3SOIDDBdpPtr +1) += m * xcbdb;
              *(here->B3SOIDDBspPtr +1) += m * xcbsb;

              *(here->B3SOIDDEbPtr  +1) -= m * (xcegb + xcedb + xcesb + xceeb);
              *(here->B3SOIDDGbPtr +1) -= m * (xcggb + xcgdb + xcgsb + xcgeb);
              *(here->B3SOIDDDPbPtr +1) -= m * (xcdgb + xcddb + xcdsb + xcdeb);
              *(here->B3SOIDDSPbPtr +1) -= m * (xcsgb + xcsdb + xcssb + xcseb);
              *(here->B3SOIDDBbPtr +1) -= m * (xcbgb + xcbdb + xcbsb + xcbeb);

              if (selfheat)
              {
                 *(here->B3SOIDDTemptempPtr + 1) += m * xcTt;
                 *(here->B3SOIDDDPtempPtr + 1) += m * xcdT;
                 *(here->B3SOIDDSPtempPtr + 1) += m * xcsT;
                 *(here->B3SOIDDBtempPtr + 1) += m * xcbT;
                 *(here->B3SOIDDEtempPtr + 1) += m * xceT;
                 *(here->B3SOIDDGtempPtr + 1) += m * xcgT;
              }
                                                               
 
if (here->B3SOIDDdebugMod > 3)
{
fprintf(fpdebug, "Cbg+Cbs+Cbe = %.5e; Cbd = %.5e;\n",
(xcbgb+xcbsb+xcbeb)/omega, xcbdb/omega);
fprintf(fpdebug, "gbs = %.5e; gbd = %.5e\n", gbbsp, gbbdp);


   fprintf(fpdebug, "AC condunctance...\n");
   fprintf(fpdebug, "Eg=%.5e; Edp=%.5e; Esp=%.5e;\nEb=%.5e; Ee=%.5e\n",
xcegb, xcedb, xcesb, -(xcegb+xcedb+xcesb+xceeb), xceeb);
   fprintf(fpdebug, "Gg=%.5e; Gdp=%.5e; Gsp=%.5e;\nGb=%.5e; Ge=%.5e\n",
xcggb, xcgdb, xcgsb, -(xcggb+xcgdb+xcgsb+xcgeb), xcgeb);
   fprintf(fpdebug, "Bg=%.5e; Bdp=%.5e; Bsp=%.5e;\nBb=%.5e; Be=%.5e\n",
xcbgb, xcbdb, xcbsb, -(xcbgb+xcbdb+xcbsb+xcbeb), xcbeb);
   fprintf(fpdebug, "DPg=%.5e; DPdp=%.5e; DPsp=%.5e;\nDPb=%.5e; DPe=%.5e\n",
xcdgb, xcddb, xcdsb, -(xcdgb+xcddb+xcdsb+xcdeb), xcdeb);
   fprintf(fpdebug, "SPg=%.5e; SPdp=%.5e; SPsp=%.5e;\nSPb=%.5e; SPe=%.5e\n",
xcsgb, xcsdb, xcssb, -(xcsgb+xcsdb+xcssb+xcseb), xcseb);
}



              *(here->B3SOIDDEgPtr) += 0.0;
              *(here->B3SOIDDEdpPtr) += 0.0;
              *(here->B3SOIDDEspPtr) += 0.0;
              *(here->B3SOIDDGePtr) -=  0.0;
              *(here->B3SOIDDDPePtr) += m * (Gme + gddpe);
              *(here->B3SOIDDSPePtr) += m * (gsspe - Gme);
             
              *(here->B3SOIDDEePtr) += 0.0;

              *(here->B3SOIDDDPgPtr) += m * (Gm + gddpg);
              *(here->B3SOIDDDPdpPtr) += m * (gdpr + gds + gddpdp + RevSum);
              *(here->B3SOIDDDPspPtr) -= m * (gds + FwdSum - gddpsp);
              *(here->B3SOIDDDPdPtr) -= m * gdpr;

              *(here->B3SOIDDSPgPtr) -= m * (Gm - gsspg);
              *(here->B3SOIDDSPdpPtr) -= m * (gds + RevSum - gsspdp);
              *(here->B3SOIDDSPspPtr) += m * (gspr + gds + FwdSum + gsspsp);
              *(here->B3SOIDDSPsPtr) -= m * gspr;

              *(here->B3SOIDDBePtr) += m * gbbe;
              *(here->B3SOIDDBgPtr)  += m * gbbg;
              *(here->B3SOIDDBdpPtr) += m * gbbdp;
              *(here->B3SOIDDBspPtr) += m * gbbsp;
	      *(here->B3SOIDDBbPtr) += m * gbbb;
              *(here->B3SOIDDEbPtr) += 0.0;
              *(here->B3SOIDDSPbPtr) -= m * (Gmbs - gsspb); 
              *(here->B3SOIDDDPbPtr) -= m * (-gddpb - Gmbs);

              if (selfheat)
              {
                 *(here->B3SOIDDDPtempPtr) += m * (GmT + gddpT);
                 *(here->B3SOIDDSPtempPtr) += m * (-GmT + gsspT);
                 *(here->B3SOIDDBtempPtr) += m * gbbT;
                 if (here->B3SOIDDbodyMod == 1) {
                    (*(here->B3SOIDDPtempPtr) += m * gppT);
                 }

                 *(here->B3SOIDDTemptempPtr) += m * (gTtt + 1/here->pParam->B3SOIDDrth);
                 *(here->B3SOIDDTempgPtr) += m * gTtg;
                 *(here->B3SOIDDTempbPtr) += m * gTtb;
                 *(here->B3SOIDDTempePtr) += m * gTte;
                 *(here->B3SOIDDTempdpPtr) += m * gTtdp;
                 *(here->B3SOIDDTempspPtr) += m * gTtsp;
              }

if (here->B3SOIDDdebugMod > 3)
{
   fprintf(fpdebug, "Static condunctance...\n");
   fprintf(fpdebug, "Gg=%.5e; Gdp=%.5e; Gsp=%.5e;\nGb=%.5e; Ge=%.5e\n",
   *(here->B3SOIDDGgPtr), *(here->B3SOIDDGdpPtr),
   *(here->B3SOIDDGspPtr), *(here->B3SOIDDGbPtr),
   *(here->B3SOIDDGePtr));
   fprintf(fpdebug, "DPg=%.5e; DPdp=%.5e; DPsp=%.5e;\nDPb=%.5e; DPe=%.5e\n",
   *(here->B3SOIDDDPgPtr), *(here->B3SOIDDDPdpPtr),
   *(here->B3SOIDDDPspPtr), *(here->B3SOIDDDPbPtr),
   *(here->B3SOIDDDPePtr));
   fprintf(fpdebug, "SPg=%.5e; SPdp=%.5e; SPsp=%.5e;\nSPb=%.5e; SPe=%.5e\n",
   *(here->B3SOIDDSPgPtr), *(here->B3SOIDDSPdpPtr),
   *(here->B3SOIDDSPspPtr), *(here->B3SOIDDSPbPtr),
   *(here->B3SOIDDSPePtr));
   fprintf(fpdebug, "Bg=%.5e; Bdp=%.5e; Bsp=%.5e;\nBb=%.5e; Be=%.5e\n",
gbbg, gbbdp, gbbsp, gbbb, gbbe);
}

              *(here->B3SOIDDDdPtr) += m * gdpr;
              *(here->B3SOIDDDdpPtr) -= m * gdpr;
              *(here->B3SOIDDSsPtr) += m * gspr;
              *(here->B3SOIDDSspPtr) -= m * gspr;


              if (here->B3SOIDDbodyMod == 1) {
                 (*(here->B3SOIDDBpPtr) -= m * gppp);
                 (*(here->B3SOIDDPbPtr) += m * gppb);
                 (*(here->B3SOIDDPpPtr) += m * gppp);
                 (*(here->B3SOIDDPgPtr) += m * gppg);
                 (*(here->B3SOIDDPdpPtr) += m * gppdp);
                 (*(here->B3SOIDDPspPtr) += m * gppsp);
                 (*(here->B3SOIDDPePtr) += m * gppe);
              }
              if (here->B3SOIDDdebugMod > 1)
              {
                      *(here->B3SOIDDVbsPtr) += m * 1;
                      *(here->B3SOIDDIdsPtr) += m * 1;
                      *(here->B3SOIDDIcPtr) += m * 1;
                      *(here->B3SOIDDIbsPtr) += m * 1;
                      *(here->B3SOIDDIbdPtr) += m * 1;
                      *(here->B3SOIDDIiiPtr) += m * 1;
                      *(here->B3SOIDDIgidlPtr) += m * 1;
                      *(here->B3SOIDDItunPtr) += m * 1;
                      *(here->B3SOIDDIbpPtr) += m * 1;
                      *(here->B3SOIDDAbeffPtr) += m * 1;
                      *(here->B3SOIDDVbs0effPtr) += m * 1;
                      *(here->B3SOIDDVbseffPtr) += 1;
                      *(here->B3SOIDDXcPtr) += m * 1;
                      *(here->B3SOIDDCbgPtr) += m * 1;
                      *(here->B3SOIDDCbbPtr) += m * 1;
                      *(here->B3SOIDDCbdPtr) += m * 1;
                      *(here->B3SOIDDqbPtr) += m * 1;
                      *(here->B3SOIDDQbfPtr) += m * 1;
                      *(here->B3SOIDDQjsPtr) += m * 1;
                      *(here->B3SOIDDQjdPtr) += m * 1;

                      /* clean up last */
                      *(here->B3SOIDDGmPtr) += m * 1;
                      *(here->B3SOIDDGmbsPtr) += m * 1;
                      *(here->B3SOIDDGdsPtr) += m * 1;
                      *(here->B3SOIDDGmePtr) += m * 1;
                      *(here->B3SOIDDVbs0teffPtr) += m * 1;
                      *(here->B3SOIDDVgsteffPtr) += m * 1;
                      *(here->B3SOIDDCbePtr) += m * 1;
                      *(here->B3SOIDDVthPtr) += m * 1;
                      *(here->B3SOIDDXcsatPtr) += m * 1;
                      *(here->B3SOIDDVdscvPtr) += m * 1;
                      *(here->B3SOIDDVcscvPtr) += m * 1;
                      *(here->B3SOIDDQaccPtr) += m * 1;
                      *(here->B3SOIDDQsub0Ptr) += m * 1;
                      *(here->B3SOIDDQsubs1Ptr) += m * 1;
                      *(here->B3SOIDDQsubs2Ptr) += m * 1;
                      *(here->B3SOIDDqgPtr) += m * 1;
                      *(here->B3SOIDDqdPtr) += m * 1;
                      *(here->B3SOIDDqePtr) += m * 1;
                      *(here->B3SOIDDDum1Ptr) += m * 1;
                      *(here->B3SOIDDDum2Ptr) += m * 1;
                      *(here->B3SOIDDDum3Ptr) += m * 1;
                      *(here->B3SOIDDDum4Ptr) += m * 1;
                      *(here->B3SOIDDDum5Ptr) += m * 1;
              }

           if (here->B3SOIDDdebugMod > 2)
              fclose(fpdebug);
        }
    }
    return(OK);
}

