/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Pin Su 99/9/27
Modified by Paolo Nenzi 2002
File: b3soifdacld.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIFDacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;
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
    for (; model != NULL; model = B3SOIFDnextModel(model)) 
    {    

      for (here = B3SOIFDinstances(model); here!= NULL;
              here = B3SOIFDnextInstance(here)) 
	 {	      
	      selfheat = (model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0 != 0.0);
              if (here->B3SOIFDdebugMod > 2)
              {
                 fpdebug = fopen("b3soifdac.log", "a");
                 fprintf(fpdebug, ".......omega=%.5e\n", omega);
              }
              if (here->B3SOIFDmode >= 0) 
	      {   Gm = here->B3SOIFDgm;
		  Gmbs = here->B3SOIFDgmbs;
	          Gme = here->B3SOIFDgme;
                  GmT = model->B3SOIFDtype * here->B3SOIFDgmT;
		  FwdSum = Gm + Gmbs + Gme;
		  RevSum = 0.0;

                  cbgb = here->B3SOIFDcbgb;
                  cbsb = here->B3SOIFDcbsb;
                  cbdb = here->B3SOIFDcbdb;
                  cbeb = here->B3SOIFDcbeb;
                  cbT  = model->B3SOIFDtype * here->B3SOIFDcbT;

                  cegb = here->B3SOIFDcegb;
                  cesb = here->B3SOIFDcesb;
                  cedb = here->B3SOIFDcedb;
                  ceeb = here->B3SOIFDceeb;
                  ceT  = model->B3SOIFDtype * here->B3SOIFDceT;

                  cggb = here->B3SOIFDcggb;
                  cgsb = here->B3SOIFDcgsb;
                  cgdb = here->B3SOIFDcgdb;
                  cgeb = here->B3SOIFDcgeb;
                  cgT  = model->B3SOIFDtype * here->B3SOIFDcgT;

                  cdgb = here->B3SOIFDcdgb;
                  cdsb = here->B3SOIFDcdsb;
                  cddb = here->B3SOIFDcddb;
                  cdeb = here->B3SOIFDcdeb;
                  cdT  = model->B3SOIFDtype * here->B3SOIFDcdT;

                  cTt = here->pParam->B3SOIFDcth;

                  gbbg  = -here->B3SOIFDgbgs;
                  gbbdp = -here->B3SOIFDgbds;
                  gbbb  = -here->B3SOIFDgbbs;
                  gbbe  = -here->B3SOIFDgbes;
                  gbbp  = -here->B3SOIFDgbps;
                  gbbT  = -model->B3SOIFDtype * here->B3SOIFDgbT;
                  gbbsp = - ( gbbg + gbbdp + gbbb + gbbe + gbbp);

                  gddpg  = -here->B3SOIFDgjdg;
                  gddpdp = -here->B3SOIFDgjdd;
                  gddpb  = -here->B3SOIFDgjdb;
                  gddpe  = -here->B3SOIFDgjde;
                  gddpT  = -model->B3SOIFDtype * here->B3SOIFDgjdT;
                  gddpsp = - ( gddpg + gddpdp + gddpb + gddpe);

                  gsspg  = -here->B3SOIFDgjsg;
                  gsspdp = -here->B3SOIFDgjsd;
                  gsspb  = -here->B3SOIFDgjsb;
                  gsspe  = 0.0;
                  gsspT  = -model->B3SOIFDtype * here->B3SOIFDgjsT;
                  gsspsp = - (gsspg + gsspdp + gsspb + gsspe);

                  gppg = -here->B3SOIFDgbpgs;
                  gppdp = -here->B3SOIFDgbpds;
             	  gppb = -here->B3SOIFDgbpbs;
                  gppe = -here->B3SOIFDgbpes;
                  gppp = -here->B3SOIFDgbpps;
                  gppT = -model->B3SOIFDtype * here->B3SOIFDgbpT;
                  gppsp = - (gppg + gppdp + gppb + gppe + gppp);

                  gTtg  = here->B3SOIFDgtempg;
                  gTtb  = here->B3SOIFDgtempb;
                  gTte  = here->B3SOIFDgtempe;
                  gTtdp = here->B3SOIFDgtempd;
                  gTtt  = here->B3SOIFDgtempT;
                  gTtsp = - (gTtg + gTtb + gTte + gTtdp);

                  sxpart = 0.6;
                  dxpart = 0.4;

              } 
	      else
	      {   Gm = -here->B3SOIFDgm;
		  Gmbs = -here->B3SOIFDgmbs;
                  Gme = -here->B3SOIFDgme;
                  GmT = -model->B3SOIFDtype * here->B3SOIFDgmT;
		  FwdSum = 0.0;
		  RevSum = -Gm - Gmbs - Gme;

                  cdgb = - (here->B3SOIFDcdgb + here->B3SOIFDcggb + here->B3SOIFDcbgb
                          + here->B3SOIFDcegb);
                  cdsb = - (here->B3SOIFDcddb + here->B3SOIFDcgdb + here->B3SOIFDcbdb
                          + here->B3SOIFDcedb);
                  cddb = - (here->B3SOIFDcdsb + here->B3SOIFDcgsb + here->B3SOIFDcbsb
                          + here->B3SOIFDcesb);
                  cdeb = - (here->B3SOIFDcdeb + here->B3SOIFDcgeb + here->B3SOIFDcbeb
                          + here->B3SOIFDceeb);
                  cdT  = - model->B3SOIFDtype * (here->B3SOIFDcgT + here->B3SOIFDcbT
                         + here->B3SOIFDcdT + here->B3SOIFDceT);

                  cegb = here->B3SOIFDcegb;
                  cesb = here->B3SOIFDcedb;
                  cedb = here->B3SOIFDcesb;
                  ceeb = here->B3SOIFDceeb;
                  ceT  = model->B3SOIFDtype * here->B3SOIFDceT;

                  cggb = here->B3SOIFDcggb;
                  cgsb = here->B3SOIFDcgdb;
                  cgdb = here->B3SOIFDcgsb;
                  cgeb = here->B3SOIFDcgeb;
                  cgT  = model->B3SOIFDtype * here->B3SOIFDcgT;

                  cbgb = here->B3SOIFDcbgb;
                  cbsb = here->B3SOIFDcbdb;
                  cbdb = here->B3SOIFDcbsb;
                  cbeb = here->B3SOIFDcbeb;
                  cbT  = model->B3SOIFDtype * here->B3SOIFDcbT;

                  cTt = here->pParam->B3SOIFDcth;

                  gbbg  = -here->B3SOIFDgbgs;
                  gbbb  = -here->B3SOIFDgbbs;
                  gbbe  = -here->B3SOIFDgbes;
                  gbbp  = -here->B3SOIFDgbps;
                  gbbsp = -here->B3SOIFDgbds;
                  gbbT  = -model->B3SOIFDtype * here->B3SOIFDgbT;
                  gbbdp = - ( gbbg + gbbsp + gbbb + gbbe + gbbp);

                  gddpg  = -here->B3SOIFDgjsg;
                  gddpsp = -here->B3SOIFDgjsd;
                  gddpb  = -here->B3SOIFDgjsb;
                  gddpe  = 0.0;
                  gddpT  = -model->B3SOIFDtype * here->B3SOIFDgjsT;
                  gddpdp = - (gddpg + gddpsp + gddpb + gddpe);

                  gsspg  = -here->B3SOIFDgjdg;
                  gsspsp = -here->B3SOIFDgjdd;
                  gsspb  = -here->B3SOIFDgjdb;
                  gsspe  = -here->B3SOIFDgjde;
                  gsspT  = -model->B3SOIFDtype * here->B3SOIFDgjdT;
                  gsspdp = - ( gsspg + gsspsp + gsspb + gsspe);

                  gppg = -here->B3SOIFDgbpgs;
                  gppsp = -here->B3SOIFDgbpds;
                  gppb = -here->B3SOIFDgbpbs;
                  gppe = -here->B3SOIFDgbpes;
                  gppp = -here->B3SOIFDgbpps;
                  gppT = -model->B3SOIFDtype * here->B3SOIFDgbpT;
                  gppdp = - (gppg + gppsp + gppb + gppe + gppp);

                  gTtt = here->B3SOIFDgtempT;
                  gTtg = here->B3SOIFDgtempg;
                  gTtb = here->B3SOIFDgtempb;
                  gTte = here->B3SOIFDgtempe;
                  gTtdp = here->B3SOIFDgtempd;
                  gTtsp = - (gTtt + gTtg + gTtb + gTte + gTtdp);

                  gTtg  = here->B3SOIFDgtempg;
                  gTtb  = here->B3SOIFDgtempb;
                  gTte  = here->B3SOIFDgtempe;
                  gTtsp = here->B3SOIFDgtempd;
                  gTtt  = here->B3SOIFDgtempT;
                  gTtdp = - (gTtg + gTtb + gTte + gTtsp);

                  sxpart = 0.6;
                  sxpart = 0.4;
                  dxpart = 0.6;
              }

              gdpr=here->B3SOIFDdrainConductance;
              gspr=here->B3SOIFDsourceConductance;
              gds= here->B3SOIFDgds;

	      GSoverlapCap = here->B3SOIFDcgso;
	      GDoverlapCap = here->B3SOIFDcgdo;
	      GEoverlapCap = here->pParam->B3SOIFDcgeo;

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

              m = here->B3SOIFDm;

                 *(here->B3SOIFDEgPtr  +1) += m * xcegb;
                 *(here->B3SOIFDEdpPtr  +1) += m * xcedb;
                 *(here->B3SOIFDEspPtr  +1) += m * xcesb;
                 *(here->B3SOIFDGePtr +1)  += m * xcgeb;
                 *(here->B3SOIFDDPePtr +1) += m * xcdeb;
                 *(here->B3SOIFDSPePtr +1) += m * xcseb;

              *(here->B3SOIFDEePtr  +1) += m * xceeb;

              *(here->B3SOIFDGgPtr  +1) += m * xcggb;
              *(here->B3SOIFDGdpPtr +1) += m * xcgdb;
              *(here->B3SOIFDGspPtr +1) += m * xcgsb;

              *(here->B3SOIFDDPgPtr +1) += m * xcdgb;
              *(here->B3SOIFDDPdpPtr +1) += m * xcddb;
              *(here->B3SOIFDDPspPtr +1) += m * xcdsb;

              *(here->B3SOIFDSPgPtr +1) += m * xcsgb;
              *(here->B3SOIFDSPdpPtr +1) += m * xcsdb;
              *(here->B3SOIFDSPspPtr +1) += m * xcssb;

              if (selfheat)
              {
                 *(here->B3SOIFDTemptempPtr + 1) += m * xcTt;
                 *(here->B3SOIFDDPtempPtr + 1) += m * xcdT;
                 *(here->B3SOIFDSPtempPtr + 1) += m * xcsT;
                 *(here->B3SOIFDBtempPtr + 1) += m * xcbT;
                 *(here->B3SOIFDEtempPtr + 1) += m * xceT;
                 *(here->B3SOIFDGtempPtr + 1) += m * xcgT;
              }
                                                               
 
if (here->B3SOIFDdebugMod > 3)
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



                 *(here->B3SOIFDEgPtr) += 0.0;
                 *(here->B3SOIFDEdpPtr) += 0.0;
                 *(here->B3SOIFDEspPtr) += 0.0;
                 *(here->B3SOIFDGePtr) -=  0.0;
                 *(here->B3SOIFDDPePtr) += m * (Gme + gddpe);
                 *(here->B3SOIFDSPePtr) += m * (gsspe - Gme);

              *(here->B3SOIFDEePtr) += 0.0;

              *(here->B3SOIFDDPgPtr) += m * (Gm + gddpg);
              *(here->B3SOIFDDPdpPtr) += m * (gdpr + gds + gddpdp + RevSum);
              *(here->B3SOIFDDPspPtr) -= m * (gds + FwdSum - gddpsp);
              *(here->B3SOIFDDPdPtr) -= m * gdpr;

              *(here->B3SOIFDSPgPtr) -= m * (Gm - gsspg);
              *(here->B3SOIFDSPdpPtr) -= m * (gds + RevSum - gsspdp);
              *(here->B3SOIFDSPspPtr) += m * (gspr + gds + FwdSum + gsspsp);
              *(here->B3SOIFDSPsPtr) -= m * gspr;

              if (selfheat)
              {
                 *(here->B3SOIFDDPtempPtr) += m * (GmT + gddpT);
                 *(here->B3SOIFDSPtempPtr) += m * (-GmT + gsspT);
                 *(here->B3SOIFDBtempPtr) += m * gbbT;
                 if (here->B3SOIFDbodyMod == 1) {
                    (*(here->B3SOIFDPtempPtr) += m * gppT);
                 }

                 *(here->B3SOIFDTemptempPtr) += m * (gTtt + 1/here->pParam->B3SOIFDrth);
                 *(here->B3SOIFDTempgPtr) += m * gTtg;
                 *(here->B3SOIFDTempbPtr) += m * gTtb;
                 *(here->B3SOIFDTempePtr) += m * gTte;
                 *(here->B3SOIFDTempdpPtr) += m * gTtdp;
                 *(here->B3SOIFDTempspPtr) += m * gTtsp;
              }

if (here->B3SOIFDdebugMod > 3)
{
   fprintf(fpdebug, "Static condunctance...\n");
   fprintf(fpdebug, "Gg=%.5e; Gdp=%.5e; Gsp=%.5e;\nGb=%.5e; Ge=%.5e\n",
   *(here->B3SOIFDGgPtr), *(here->B3SOIFDGdpPtr),
   *(here->B3SOIFDGspPtr), *(here->B3SOIFDGbPtr),
   *(here->B3SOIFDGePtr));
   fprintf(fpdebug, "DPg=%.5e; DPdp=%.5e; DPsp=%.5e;\nDPb=%.5e; DPe=%.5e\n",
   *(here->B3SOIFDDPgPtr), *(here->B3SOIFDDPdpPtr),
   *(here->B3SOIFDDPspPtr), *(here->B3SOIFDDPbPtr),
   *(here->B3SOIFDDPePtr));
   fprintf(fpdebug, "SPg=%.5e; SPdp=%.5e; SPsp=%.5e;\nSPb=%.5e; SPe=%.5e\n",
   *(here->B3SOIFDSPgPtr), *(here->B3SOIFDSPdpPtr),
   *(here->B3SOIFDSPspPtr), *(here->B3SOIFDSPbPtr),
   *(here->B3SOIFDSPePtr));
   fprintf(fpdebug, "Bg=%.5e; Bdp=%.5e; Bsp=%.5e;\nBb=%.5e; Be=%.5e\n",
gbbg, gbbdp, gbbsp, gbbb, gbbe);
}

              *(here->B3SOIFDDdPtr) += m * gdpr;
              *(here->B3SOIFDDdpPtr) -= m * gdpr;
              *(here->B3SOIFDSsPtr) += m * gspr;
              *(here->B3SOIFDSspPtr) -= m * gspr;


              if (here->B3SOIFDbodyMod == 1) {
                 (*(here->B3SOIFDBpPtr) -= m * gppp);
                 (*(here->B3SOIFDPbPtr) += m * gppb);
                 (*(here->B3SOIFDPpPtr) += m * gppp);
                    (*(here->B3SOIFDPgPtr) += m * gppg);
                    (*(here->B3SOIFDPdpPtr) += m * gppdp);
                    (*(here->B3SOIFDPspPtr) += m * gppsp);
                    (*(here->B3SOIFDPePtr) += m * gppe);
              }
              if (here->B3SOIFDdebugMod > 1)
              {
                      *(here->B3SOIFDVbsPtr) += m * 1;
                      *(here->B3SOIFDIdsPtr) += m * 1;
                      *(here->B3SOIFDIcPtr) += m * 1;
                      *(here->B3SOIFDIbsPtr) += m * 1;
                      *(here->B3SOIFDIbdPtr) += m * 1;
                      *(here->B3SOIFDIiiPtr) += m * 1;
                      *(here->B3SOIFDIgidlPtr) += m * 1;
                      *(here->B3SOIFDItunPtr) += m * 1;
                      *(here->B3SOIFDIbpPtr) += m * 1;
                      *(here->B3SOIFDAbeffPtr) += m * 1;
                      *(here->B3SOIFDVbs0effPtr) += m * 1;
                      *(here->B3SOIFDVbseffPtr) += m * 1;
                      *(here->B3SOIFDXcPtr) += m * 1;
                      *(here->B3SOIFDCbgPtr) += m * 1;
                      *(here->B3SOIFDCbbPtr) += m * 1;
                      *(here->B3SOIFDCbdPtr) += m * 1;
                      *(here->B3SOIFDqbPtr) += m * 1;
                      *(here->B3SOIFDQbfPtr) += m * 1;
                      *(here->B3SOIFDQjsPtr) += m * 1;
                      *(here->B3SOIFDQjdPtr) += m * 1;

                      /* clean up last */
                      *(here->B3SOIFDGmPtr) += m * 1;
                      *(here->B3SOIFDGmbsPtr) += m * 1;
                      *(here->B3SOIFDGdsPtr) += m * 1;
                      *(here->B3SOIFDGmePtr) += m * 1;
                      *(here->B3SOIFDVbs0teffPtr) += m * 1;
                      *(here->B3SOIFDVgsteffPtr) += m * 1;
                      *(here->B3SOIFDCbePtr) += m * 1;
                      *(here->B3SOIFDVthPtr) += m * 1;
                      *(here->B3SOIFDXcsatPtr) += m * 1;
                      *(here->B3SOIFDVdscvPtr) += m * 1;
                      *(here->B3SOIFDVcscvPtr) += m * 1;
                      *(here->B3SOIFDQaccPtr) += m * 1;
                      *(here->B3SOIFDQsub0Ptr) += m * 1;
                      *(here->B3SOIFDQsubs1Ptr) += m * 1;
                      *(here->B3SOIFDQsubs2Ptr) += m * 1;
                      *(here->B3SOIFDqgPtr) += m * 1;
                      *(here->B3SOIFDqdPtr) += m * 1;
                      *(here->B3SOIFDqePtr) += m * 1;
                      *(here->B3SOIFDDum1Ptr) += m * 1;
                      *(here->B3SOIFDDum2Ptr) += m * 1;
                      *(here->B3SOIFDDum3Ptr) += m * 1;
                      *(here->B3SOIFDDum4Ptr) += m * 1;
                      *(here->B3SOIFDDum5Ptr) += m * 1;
              }

           if (here->B3SOIFDdebugMod > 2)
              fclose(fpdebug);
        }
    }
    return(OK);
}

