/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddacld.c          98/5/01
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "b3soidddef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIDDacLoad(inModel,ckt)
GENmodel *inModel;
register CKTcircuit *ckt;
{
register B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
register B3SOIDDinstance *here;
register int selfheat;
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
double xcTt, cTt, gcTt, gTtt, gTtg, gTtb, gTte, gTtdp, gTtsp;

double Dum1, Dum2, Dum3, Dum4, Dum5;
FILE *fpdebug;

    omega = ckt->CKTomega;
    for (; model != NULL; model = model->B3SOIDDnextModel) 
    {    

      for (here = model->B3SOIDDinstances; here!= NULL;
              here = here->B3SOIDDnextInstance) 
	 {    
              selfheat = (model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0 != 0.0);
              if (here->B3SOIDDdebugMod > 2)
              {
                 fpdebug = fopen("b3soiDDac.log", "a");
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

              *(here->B3SOIDDEgPtr  +1) += xcegb;
              *(here->B3SOIDDEdpPtr  +1) += xcedb;
              *(here->B3SOIDDEspPtr  +1) += xcesb;
              *(here->B3SOIDDGePtr +1)  += xcgeb;
              *(here->B3SOIDDDPePtr +1) += xcdeb;
              *(here->B3SOIDDSPePtr +1) += xcseb;
              
              *(here->B3SOIDDEePtr  +1) += xceeb;

              *(here->B3SOIDDGgPtr  +1) += xcggb;
              *(here->B3SOIDDGdpPtr +1) += xcgdb;
              *(here->B3SOIDDGspPtr +1) += xcgsb;

              *(here->B3SOIDDDPgPtr +1) += xcdgb;
              *(here->B3SOIDDDPdpPtr +1) += xcddb;
              *(here->B3SOIDDDPspPtr +1) += xcdsb;

              *(here->B3SOIDDSPgPtr +1) += xcsgb;
              *(here->B3SOIDDSPdpPtr +1) += xcsdb;
              *(here->B3SOIDDSPspPtr +1) += xcssb;

              *(here->B3SOIDDBePtr +1) += xcbeb;
              *(here->B3SOIDDBgPtr +1) += xcbgb;
              *(here->B3SOIDDBdpPtr +1) += xcbdb;
              *(here->B3SOIDDBspPtr +1) += xcbsb;

              *(here->B3SOIDDEbPtr  +1) -= xcegb + xcedb + xcesb + xceeb;
              *(here->B3SOIDDGbPtr +1) -= xcggb + xcgdb + xcgsb + xcgeb;
              *(here->B3SOIDDDPbPtr +1) -= xcdgb + xcddb + xcdsb + xcdeb;
              *(here->B3SOIDDSPbPtr +1) -= xcsgb + xcsdb + xcssb + xcseb;
              *(here->B3SOIDDBbPtr +1) -= xcbgb + xcbdb + xcbsb + xcbeb;

              if (selfheat)
              {
                 *(here->B3SOIDDTemptempPtr + 1) += xcTt;
                 *(here->B3SOIDDDPtempPtr + 1) += xcdT;
                 *(here->B3SOIDDSPtempPtr + 1) += xcsT;
                 *(here->B3SOIDDBtempPtr + 1) += xcbT;
                 *(here->B3SOIDDEtempPtr + 1) += xceT;
                 *(here->B3SOIDDGtempPtr + 1) += xcgT;
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
              *(here->B3SOIDDDPePtr) += Gme + gddpe;
              *(here->B3SOIDDSPePtr) += gsspe - Gme;
             
              *(here->B3SOIDDEePtr) += 0.0;

              *(here->B3SOIDDDPgPtr) += Gm + gddpg;
              *(here->B3SOIDDDPdpPtr) += gdpr + gds + gddpdp + RevSum ;
              *(here->B3SOIDDDPspPtr) -= gds + FwdSum - gddpsp;
              *(here->B3SOIDDDPdPtr) -= gdpr;

              *(here->B3SOIDDSPgPtr) -= Gm - gsspg;
              *(here->B3SOIDDSPdpPtr) -= gds + RevSum - gsspdp;
              *(here->B3SOIDDSPspPtr) += gspr + gds + FwdSum + gsspsp;
              *(here->B3SOIDDSPsPtr) -= gspr;

              *(here->B3SOIDDBePtr) += gbbe;
              *(here->B3SOIDDBgPtr)  += gbbg;
              *(here->B3SOIDDBdpPtr) += gbbdp;
              *(here->B3SOIDDBspPtr) += gbbsp;
	      *(here->B3SOIDDBbPtr) += gbbb;
              *(here->B3SOIDDEbPtr) += 0.0;
              *(here->B3SOIDDSPbPtr) -= Gmbs - gsspb; 
              *(here->B3SOIDDDPbPtr) -= (-gddpb - Gmbs);

              if (selfheat)
              {
                 *(here->B3SOIDDDPtempPtr) += GmT + gddpT;
                 *(here->B3SOIDDSPtempPtr) += -GmT + gsspT;
                 *(here->B3SOIDDBtempPtr) += gbbT;
                 if (here->B3SOIDDbodyMod == 1) {
                    (*(here->B3SOIDDPtempPtr) += gppT);
                 }

                 *(here->B3SOIDDTemptempPtr) += gTtt + 1/here->pParam->B3SOIDDrth;
                 *(here->B3SOIDDTempgPtr) += gTtg;
                 *(here->B3SOIDDTempbPtr) += gTtb;
                 *(here->B3SOIDDTempePtr) += gTte;
                 *(here->B3SOIDDTempdpPtr) += gTtdp;
                 *(here->B3SOIDDTempspPtr) += gTtsp;
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

              *(here->B3SOIDDDdPtr) += gdpr;
              *(here->B3SOIDDDdpPtr) -= gdpr;
              *(here->B3SOIDDSsPtr) += gspr;
              *(here->B3SOIDDSspPtr) -= gspr;


              if (here->B3SOIDDbodyMod == 1) {
                 (*(here->B3SOIDDBpPtr) -= gppp);
                 (*(here->B3SOIDDPbPtr) += gppb);
                 (*(here->B3SOIDDPpPtr) += gppp);
                 (*(here->B3SOIDDPgPtr) += gppg);
                 (*(here->B3SOIDDPdpPtr) += gppdp);
                 (*(here->B3SOIDDPspPtr) += gppsp);
                 (*(here->B3SOIDDPePtr) += gppe);
              }
              if (here->B3SOIDDdebugMod > 1)
              {
                      *(here->B3SOIDDVbsPtr) += 1;
                      *(here->B3SOIDDIdsPtr) += 1;
                      *(here->B3SOIDDIcPtr) += 1;
                      *(here->B3SOIDDIbsPtr) += 1;
                      *(here->B3SOIDDIbdPtr) += 1;
                      *(here->B3SOIDDIiiPtr) += 1;
                      *(here->B3SOIDDIgidlPtr) += 1;
                      *(here->B3SOIDDItunPtr) += 1;
                      *(here->B3SOIDDIbpPtr) += 1;
                      *(here->B3SOIDDAbeffPtr) += 1;
                      *(here->B3SOIDDVbs0effPtr) += 1;
                      *(here->B3SOIDDVbseffPtr) += 1;
                      *(here->B3SOIDDXcPtr) += 1;
                      *(here->B3SOIDDCbgPtr) += 1;
                      *(here->B3SOIDDCbbPtr) += 1;
                      *(here->B3SOIDDCbdPtr) += 1;
                      *(here->B3SOIDDqbPtr) += 1;
                      *(here->B3SOIDDQbfPtr) += 1;
                      *(here->B3SOIDDQjsPtr) += 1;
                      *(here->B3SOIDDQjdPtr) += 1;

                      /* clean up last */
                      *(here->B3SOIDDGmPtr) += 1;
                      *(here->B3SOIDDGmbsPtr) += 1;
                      *(here->B3SOIDDGdsPtr) += 1;
                      *(here->B3SOIDDGmePtr) += 1;
                      *(here->B3SOIDDVbs0teffPtr) += 1;
                      *(here->B3SOIDDVgsteffPtr) += 1;
                      *(here->B3SOIDDCbePtr) += 1;
                      *(here->B3SOIDDVthPtr) += 1;
                      *(here->B3SOIDDXcsatPtr) += 1;
                      *(here->B3SOIDDVdscvPtr) += 1;
                      *(here->B3SOIDDVcscvPtr) += 1;
                      *(here->B3SOIDDQaccPtr) += 1;
                      *(here->B3SOIDDQsub0Ptr) += 1;
                      *(here->B3SOIDDQsubs1Ptr) += 1;
                      *(here->B3SOIDDQsubs2Ptr) += 1;
                      *(here->B3SOIDDqgPtr) += 1;
                      *(here->B3SOIDDqdPtr) += 1;
                      *(here->B3SOIDDqePtr) += 1;
                      *(here->B3SOIDDDum1Ptr) += 1;
                      *(here->B3SOIDDDum2Ptr) += 1;
                      *(here->B3SOIDDDum3Ptr) += 1;
                      *(here->B3SOIDDDum4Ptr) += 1;
                      *(here->B3SOIDDDum5Ptr) += 1;
              }

           if (here->B3SOIDDdebugMod > 2)
              fclose(fpdebug);
        }
    }
    return(OK);
}

