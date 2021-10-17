/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* actually load the current ac sensitivity 
 * information into the  array previously provided 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "bjtdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


int
BJTsAcLoad(GENmodel *inModel, CKTcircuit *ckt)
{

    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;
    double SaveState[25];
    int    error;
    int    flag;
    double vbeOp;
    double vbcOp;
    double A0;
    double DELA = 0.0;
    double Apert;
    double DELAinv;
    double vte = 0.0;
    double gcpr;
    double gepr;
    double gpi;
    double gmu;
    double go;
    double xgm;
    double td;
    double arg;
    double gm;
    double gx;
    double xcpi;
    double xcmu;
    double xcbx;
    double xccs;
    double xcmcb;
    double cx,icx;
    double cbx,icbx;
    double ccs,iccs;
    double cbc,icbc;
    double cbe,icbe;
    double cce,icce;
    double cb,icb;
    double cbprm,icbprm;
    double cc,icc;
    double ccprm,iccprm;
    double ce,ice;
    double ceprm,iceprm;
    double cs,ics;
    double vcpr,ivcpr;
    double vepr,ivepr;
    double vx,ivx;
    double vbx,ivbx;
    double vcs,ivcs;
    double vbc,ivbc;
    double vbe,ivbe;
    double vce,ivce;
    double cb0,icb0;
    double cbprm0,icbprm0;
    double cc0,icc0;
    double ccprm0,iccprm0;
    double ce0,ice0;
    double ceprm0,iceprm0;
    double cs0,ics0;
    double DvDp = 0.0;
    int iparmno,i;
    SENstruct *info;


#ifdef SENSDEBUG
    printf("BJTsenacload \n");
    printf("BJTsenacload \n");
#endif /* SENSDEBUG */

    info = ckt->CKTsenInfo;
    info->SENstatus = PERTURBATION;

    /*  loop through all the models */
    for( ; model != NULL; model = BJTnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = BJTinstances(model); here != NULL ;
                here=BJTnextInstance(here)) {


            /* save the unperturbed values in the state vector */
            for(i=0; i <= 20; i++) {
                *(SaveState + i) = *(ckt->CKTstate0 + here->BJTstate + i);
            }

            vcpr = *(ckt->CKTrhsOld + here->BJTcolNode) 
                - *(ckt->CKTrhsOld + here->BJTcolPrimeNode) ;
            ivcpr = *(ckt->CKTirhsOld + here->BJTcolNode) 
                - *(ckt->CKTirhsOld + here->BJTcolPrimeNode) ;
            vepr = *(ckt->CKTrhsOld + here->BJTemitNode) 
                - *(ckt->CKTrhsOld + here->BJTemitPrimeNode) ;
            ivepr = *(ckt->CKTirhsOld + here->BJTemitNode) 
                - *(ckt->CKTirhsOld + here->BJTemitPrimeNode) ;
            vx = *(ckt->CKTrhsOld + here->BJTbaseNode) 
                - *(ckt->CKTrhsOld + here->BJTbasePrimeNode) ;/* vb_bprm */
            ivx = *(ckt->CKTirhsOld + here->BJTbaseNode)
                - *(ckt->CKTirhsOld + here->BJTbasePrimeNode) ;/* ivb_bprm */
            vcs = *(ckt->CKTrhsOld + here->BJTcolPrimeNode) 
                - *(ckt->CKTrhsOld + here->BJTsubstNode) ;
            ivcs = *(ckt->CKTirhsOld + here->BJTcolPrimeNode) 
                - *(ckt->CKTirhsOld + here->BJTsubstNode) ;
            vbc = *(ckt->CKTrhsOld + here->BJTbasePrimeNode) 
                - *(ckt->CKTrhsOld + here->BJTcolPrimeNode) ;/* vbprm_cprm */
            ivbc = *(ckt->CKTirhsOld + here->BJTbasePrimeNode)
                - *(ckt->CKTirhsOld + here->BJTcolPrimeNode) ;/* ivbprm_cprm */
            vbe = *(ckt->CKTrhsOld + here->BJTbasePrimeNode) 
                - *(ckt->CKTrhsOld + here->BJTemitPrimeNode) ;/* vbprm_eprm */
            ivbe = *(ckt->CKTirhsOld + here->BJTbasePrimeNode)
                - *(ckt->CKTirhsOld + here->BJTemitPrimeNode) ;/* ivbprm_eprm */
            vce = vbe - vbc ;
            ivce = ivbe - ivbc ;
            vbx = vx + vbc ;
            ivbx = ivx + ivbc ;



            vbeOp =model->BJTtype * ( *(ckt->CKTrhsOp +  here->BJTbasePrimeNode)
                -  *(ckt->CKTrhsOp +  here->BJTemitPrimeNode));
            vbcOp =model->BJTtype * ( *(ckt->CKTrhsOp +  here->BJTbasePrimeNode)
                -  *(ckt->CKTrhsOp +  here->BJTcolPrimeNode));

#ifdef SENSDEBUG
            printf("\n without perturbation\n");
#endif /* SENSDEBUG */
            /* without perturbation */
            A0 = here->BJTarea;
            here->BJTsenPertFlag = ON;
            *(ckt->CKTstate0 + here->BJTvbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJTvbc) = vbcOp;
            /* info->SENacpertflag == 1 only for first frequency */

            if(info->SENacpertflag == 1){

                /* store the  unperturbed values of small signal parameters */

	      if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		     return(error);

                *(here->BJTsenGpi)= *(ckt->CKTstate0 + here->BJTgpi);
                *(here->BJTsenGmu)= *(ckt->CKTstate0 + here->BJTgmu);
                *(here->BJTsenGm)= *(ckt->CKTstate0 + here->BJTgm);
                *(here->BJTsenGo)= *(ckt->CKTstate0 + here->BJTgo);
                *(here->BJTsenGx)= *(ckt->CKTstate0 + here->BJTgx);
                *(here->BJTsenCpi)= *(ckt->CKTstate0 + here->BJTcqbe);
                *(here->BJTsenCmu)= *(ckt->CKTstate0 + here->BJTcqbc);
                *(here->BJTsenCbx)= *(ckt->CKTstate0 + here->BJTcqbx);
                *(here->BJTsenCsub)= *(ckt->CKTstate0 + here->BJTcqsub);
                *(here->BJTsenCmcb)= *(ckt->CKTstate0 + here->BJTcexbc);
            }
            gcpr = here->BJTtcollectorConduct;
            gepr = here->BJTtemitterConduct;
            gpi= *(here->BJTsenGpi);
            gmu= *(here->BJTsenGmu);
            gm= *(here->BJTsenGm);
            go= *(here->BJTsenGo);
            gx= *(here->BJTsenGx);
            xgm=0;
            td=model->BJTexcessPhase;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            xcpi= *(here->BJTsenCpi) * ckt->CKTomega;
            xcmu= *(here->BJTsenCmu) * ckt->CKTomega;
            xcbx= *(here->BJTsenCbx) * ckt->CKTomega;
            xccs= *(here->BJTsenCsub) * ckt->CKTomega;
            xcmcb= *(here->BJTsenCmcb) * ckt->CKTomega;


            cx=gx * vx ;
            icx=gx * ivx;
            cbx=( -xcbx * ivbx) ;
            icbx= xcbx * vbx ;
            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cbc=(gmu * vbc -xcmu * ivbc) ;
            icbc=xcmu * vbc +  gmu * ivbc ;
            cbe=gpi * vbe -xcpi * ivbe - xcmcb * ivbc ;
            icbe=xcpi * vbe +  gpi * ivbe + xcmcb * vbc;
            cce= go * vce + gm * vbe - xgm * ivbe;
            icce=go * ivce + gm * ivbe + xgm * vbe ;

            cc0=gcpr * vcpr ;
            icc0=gcpr * ivcpr ;
            ce0=gepr * vepr;
            ice0=gepr * ivepr ;
            cb0 = cx + cbx;
            icb0 = icx + icbx;
            if(here->BJTbaseNode != here->BJTbasePrimeNode){
                cbprm0 = (- cx + cbe + cbc);
                icbprm0 = (- icx + icbe + icbc);
            }
            else{
                cbprm0 = ( cbx + cbe + cbc);
                icbprm0 = (icbx + icbe + icbc);
            }
            ccprm0 = (- cbx - cc0 + ccs + cce - cbc);
            iccprm0 = (- icbx - icc0 + iccs + icce - icbc);
            ceprm0 = (- cbe - cce - ce0);
            iceprm0 = (- icbe - icce - ice0);
            cs0 = (- ccs) ;
            ics0 = (- iccs) ;

#ifdef SENSDEBUG
            printf("gepr0 = %.7e , gcpr0 = %.7e , gmu0 = %.7e, gpi0 = %.7e\n",
                    gepr,gcpr,gmu,gpi);
            printf("gm0 = %.7e , go0 = %.7e , gx0 = %.7e, xcpi0 = %.7e\n",
                    gm,go,gx,xcpi);
            printf("xcmu0 = %.7e , xcbx0 = %.7e , xccs0 = %.7e, xcmcb0 = %.7e\n"
                    ,xcmu,xcbx,xccs,xcmcb);
            printf("vepr = %.7e + j%.7e , vcpr = %.7e + j%.7e\n",
                    vepr,ivepr,vcpr,ivcpr);
            printf("vbx = %.7e + j%.7e , vx = %.7e + j%.7e\n",
                    vbx,ivbx,vx,ivx);
            printf("vbc = %.7e + j%.7e , vbe = %.7e + j%.7e\n",
                    vbc,ivbc,vbe,ivbe);
            printf("vce = %.7e + j%.7e , vcs = %.7e + j%.7e\n",
                    vce,ivce,vcs,ivcs);
            printf("cce0 = %.7e + j%.7e , cbe0 = %.7e + j%.7e\n",
                    cce,icce,cbe,icbe);
            printf("cbc0 = %.7e + j%.7e\n",
                    cbc,icbc);
            printf("cc0 = %.7e + j%.7e , ce0 = %.7e + j%.7e\n",
                    cc0,icc0,ce0,ice0);
            printf("cb0 = %.7e + j%.7e , cs0 = %.7e + j%.7e\n",
                    cb0,icb0,cs0,ics0);
            printf("cbprm0 = %.7e + j%.7e , ceprm0 = %.7e + j%.7e\n",
                    cbprm0,icbprm0,ceprm0,iceprm0);
            printf("ccprm0 = %.7e + j%.7e \n",
                    ccprm0,iccprm0);
            printf("\nPerturbation of Area\n");
#endif /* SENSDEBUG */
            /* Perturbation of Area */
            if(here->BJTsenParmNo == 0){
                flag = 0;
                goto next1;
            }

            DELA = info->SENpertfac * A0;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            here->BJTarea = Apert;

            *(ckt->CKTstate0 + here->BJTvbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJTvbc) = vbcOp;
            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed area 
                 */
                if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->BJTsenGpi + 1)= *(ckt->CKTstate0 + here->BJTgpi);
                *(here->BJTsenGmu + 1)= *(ckt->CKTstate0 + here->BJTgmu);
                *(here->BJTsenGm + 1)= *(ckt->CKTstate0 + here->BJTgm);
                *(here->BJTsenGo + 1)= *(ckt->CKTstate0 + here->BJTgo);
                *(here->BJTsenGx + 1)= *(ckt->CKTstate0 + here->BJTgx);
                *(here->BJTsenCpi + 1)= *(ckt->CKTstate0 + here->BJTcqbe);
                *(here->BJTsenCmu + 1)= *(ckt->CKTstate0 + here->BJTcqbc);
                *(here->BJTsenCbx + 1)= *(ckt->CKTstate0 + here->BJTcqbx);
                *(here->BJTsenCsub + 1)= *(ckt->CKTstate0 + here->BJTcqsub);
                *(here->BJTsenCmcb + 1)= *(ckt->CKTstate0 + here->BJTcexbc);
            }


            flag = 0;
            goto load;



pertvbx:    /* Perturbation of vbx */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbx\n");
#endif /* SENSDEBUG */
            here->BJTarea = A0;
            A0 = model->BJTtype * (*(ckt->CKTrhsOp + here->BJTbaseNode)
                - *(ckt->CKTrhsOp + here->BJTcolPrimeNode));
            DELA = info->SENpertfac * A0 + 1e-8; 
            Apert = A0 + DELA;
            DELAinv = model->BJTtype * 1.0/DELA;
            *(ckt->CKTrhsOp + here->BJTbaseNode) += DELA;
            *(ckt->CKTstate0 + here->BJTvbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJTvbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbx 
                 */
                if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->BJTsenGpi + 2)= *(ckt->CKTstate0 + here->BJTgpi);
                *(here->BJTsenGmu + 2)= *(ckt->CKTstate0 + here->BJTgmu);
                *(here->BJTsenGm + 2)= *(ckt->CKTstate0 + here->BJTgm);
                *(here->BJTsenGo + 2)= *(ckt->CKTstate0 + here->BJTgo);
                *(here->BJTsenGx + 2)= *(ckt->CKTstate0 + here->BJTgx);
                *(here->BJTsenCpi + 2)= *(ckt->CKTstate0 + here->BJTcqbe);
                *(here->BJTsenCmu + 2)= *(ckt->CKTstate0 + here->BJTcqbc);
                *(here->BJTsenCbx + 2)= *(ckt->CKTstate0 + here->BJTcqbx);
                *(here->BJTsenCsub + 2)= *(ckt->CKTstate0 + here->BJTcqsub);
                *(here->BJTsenCmcb + 2)= *(ckt->CKTstate0 + here->BJTcexbc);
            }


            flag = 1;
            goto load;


pertvbe:    /* Perturbation of vbe */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbe\n");
#endif /* SENSDEBUG */
            if (*(here->BJTsenCbx) != 0){ 
                *(ckt->CKTrhsOp + here ->BJTbaseNode) -= DELA;
            }
            vte=model->BJTleakBEemissionCoeff*CONSTvt0;
            A0 = vbeOp;
            DELA = info->SENpertfac * vte ;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;
            *(ckt->CKTstate0 + here->BJTvbe) = Apert;
            *(ckt->CKTstate0 + here->BJTvbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbe 
                 */
                if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		  return(error);

                *(here->BJTsenGpi + 3)= *(ckt->CKTstate0 + here->BJTgpi);
                *(here->BJTsenGmu + 3)= *(ckt->CKTstate0 + here->BJTgmu);
                *(here->BJTsenGm + 3)= *(ckt->CKTstate0 + here->BJTgm);
                *(here->BJTsenGo + 3)= *(ckt->CKTstate0 + here->BJTgo);
                *(here->BJTsenGx + 3)= *(ckt->CKTstate0 + here->BJTgx);
                *(here->BJTsenCpi + 3)= *(ckt->CKTstate0 + here->BJTcqbe);
                *(here->BJTsenCmu + 3)= *(ckt->CKTstate0 + here->BJTcqbc);
                *(here->BJTsenCbx + 3)= *(ckt->CKTstate0 + here->BJTcqbx);
                *(here->BJTsenCsub + 3)= *(ckt->CKTstate0 + here->BJTcqsub);
                *(here->BJTsenCmcb + 3)= *(ckt->CKTstate0 + here->BJTcexbc);
            }


            flag = 2;
            goto load;


pertvbc:    /* Perturbation of vbc */
#ifdef SENSDEBUG
            printf("\nPerturbation of vbc\n");
#endif /* SENSDEBUG */
            *(ckt->CKTstate0 + here->BJTvbe) = A0;

            A0 = vbcOp;
            DELA = info->SENpertfac * vte ;
            Apert = A0 + DELA;
            DELAinv = 1.0/DELA;


            *(ckt->CKTstate0 + here->BJTvbc) = Apert;

            *(ckt->CKTstate0 + here->BJTvbe) = vbeOp;



            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vbc 
                 */
                if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->BJTsenGpi + 4)= *(ckt->CKTstate0 + here->BJTgpi);
                *(here->BJTsenGmu + 4)= *(ckt->CKTstate0 + here->BJTgmu);
                *(here->BJTsenGm + 4)= *(ckt->CKTstate0 + here->BJTgm);
                *(here->BJTsenGo + 4)= *(ckt->CKTstate0 + here->BJTgo);
                *(here->BJTsenGx + 4)= *(ckt->CKTstate0 + here->BJTgx);
                *(here->BJTsenCpi + 4)= *(ckt->CKTstate0 + here->BJTcqbe);
                *(here->BJTsenCmu + 4)= *(ckt->CKTstate0 + here->BJTcqbc);
                *(here->BJTsenCbx + 4)= *(ckt->CKTstate0 + here->BJTcqbx);
                *(here->BJTsenCsub + 4)= *(ckt->CKTstate0 + here->BJTcqsub);
                *(here->BJTsenCmcb + 4)= *(ckt->CKTstate0 + here->BJTcexbc);

            }


            flag = 3;
            goto load;



pertvcs:    /* Perturbation of vcs */
#ifdef SENSDEBUG
            printf("\nPerturbation of vcs\n");
#endif /* SENSDEBUG */
            *(ckt->CKTstate0 + here->BJTvbc) = A0;
            A0 = model->BJTtype * (*(ckt->CKTrhsOp + here->BJTsubstNode)
                -  *(ckt->CKTrhsOp + here->BJTcolPrimeNode));
            DELA = info->SENpertfac * A0 + 1e-8; 
            Apert = A0 + DELA;
            DELAinv = model->BJTtype * 1.0/DELA;
            *(ckt->CKTrhsOp + here->BJTsubstNode) += DELA;
            *(ckt->CKTstate0 + here->BJTvbe) = vbeOp;
            *(ckt->CKTstate0 + here->BJTvbc) = vbcOp;

            if(info->SENacpertflag == 1){

                /* store the  small signal parameters 
                 * corresponding to perturbed vcs 
                 */
                if ((error = BJTload((GENmodel*)model,ckt)) != 0)
		  return(error);
                *(here->BJTsenCsub + 5)= *(ckt->CKTstate0 + here->BJTcqsub);

            }


            flag = 4;

            *(ckt->CKTrhsOp + here->BJTsubstNode) -= DELA;
            xccs= *(here->BJTsenCsub + 5) * ckt->CKTomega;

            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cs = -ccs;
            ics = -iccs;
            ccprm = ccprm0 +  cs0 - cs;
            iccprm = iccprm0 + ics0 - ics;
            cbprm = cbprm0;
            icbprm = icbprm0;
            ceprm = ceprm0;
            iceprm = iceprm0;
            cc = cc0;
            icc = icc0;
            ce = ce0;
            ice = ice0;
            cb = cb0;
            icb = icb0;
            goto next2;

load:
            gcpr=here->BJTtcollectorConduct;
            gepr=here->BJTtemitterConduct;
            gpi= *(here->BJTsenGpi + flag+1);
            gmu= *(here->BJTsenGmu + flag+1);
            gm= *(here->BJTsenGm + flag+1);
            go= *(here->BJTsenGo + flag+1);
            gx= *(here->BJTsenGx + flag+1);
            xgm=0;
            td=model->BJTexcessPhase;
            if(td != 0) {
                arg = td*ckt->CKTomega;
                gm = gm+go;
                xgm = -gm * sin(arg);
                gm = gm * cos(arg)-go;
            }
            xcpi= *(here->BJTsenCpi + flag+1) * ckt->CKTomega;
            xcmu= *(here->BJTsenCmu + flag+1) * ckt->CKTomega;
            xcbx= *(here->BJTsenCbx + flag+1) * ckt->CKTomega;
            xccs= *(here->BJTsenCsub + flag+1) * ckt->CKTomega;
            xcmcb= *(here->BJTsenCmcb + flag+1) * ckt->CKTomega;


            cc=gcpr * vcpr ;
            icc=gcpr * ivcpr ;
            ce=gepr * vepr;
            ice=gepr * ivepr ;
            cx=gx * vx ;
            icx=gx * ivx;
            cbx=( -xcbx * ivbx) ;
            icbx= xcbx * vbx ;
            ccs=( -xccs * ivcs) ;
            iccs= xccs * vcs ;
            cbc=(gmu * vbc -xcmu * ivbc) ;
            icbc=xcmu * vbc +  gmu * ivbc ;
            cbe=gpi * vbe -xcpi * ivbe - xcmcb * ivbc ;
            icbe=xcpi * vbe +  gpi * ivbe + xcmcb * vbc;
            cce= go * vce + gm * vbe - xgm * ivbe;
            icce=go * ivce + gm * ivbe + xgm * vbe ;


            cb= cx + cbx;
            icb= icx + icbx;
            if(here->BJTbaseNode != here->BJTbasePrimeNode){
                cbprm=(- cx + cbe + cbc);
                icbprm=(- icx + icbe + icbc);
            }
            else{
                cbprm=( cbx + cbe + cbc);
                icbprm=(icbx + icbe + icbc);
            }
            ccprm=(- cbx - cc + ccs + cce - cbc);
            iccprm=(- icbx - icc + iccs + icce - icbc);
            ceprm=(- cbe - cce - ce);
            iceprm=(- icbe - icce - ice);
            cs= (- ccs) ;
            ics= (- iccs) ;

#ifdef SENSDEBUG
            printf("A0 = %.7e , Apert = %.7e , DELA = %.7e\n"
                    ,A0,Apert,DELA);
            printf("gepr = %.7e , gcpr = %.7e , gmu = %.7e, gpi = %.7e\n"
                    ,gepr,gcpr,gmu,gpi);
            printf("gm = %.7e , go = %.7e , gx = %.7e, xcpi = %.7e\n"
                    ,gm,go,gx,xcpi);
            printf("xcmu = %.7e , xcbx = %.7e , xccs = %.7e, xcmcb = %.7e\n"
                    ,xcmu,xcbx,xccs,xcmcb);

            printf("cx = %.7e + j%.7e , cbx = %.7e + j%.7e\n"
                    ,cx,icx,cbx,icbx);
            printf("ccs %.7e + j%.7e , cbc = %.7e + j%.7e"
                    ,ccs,iccs,cbc,icbc);
            printf("cbe %.7e + j%.7e , cce = %.7e + j%.7e\n"
                    ,cbe,icbe,cce,icce);

            printf("cc = %.7e + j%.7e , ce = %.7e + j%.7e,",
                    cc,icc,ce,ice);
            printf("ccprm = %.7e + j%.7e , ceprm = %.7e + j%.7e",
                    ccprm,iccprm,ceprm,iceprm);
            printf("cb = %.7e + j%.7e , cbprm = %.7e + j%.7e , ",
                    cb,icb,cbprm,icbprm);
            printf("cs = %.7e + j%.7e\n",
                    cs,ics);
#endif /* SENSDEBUG */


            /* load the RHS matrix */
next2:
            for(iparmno = 1;iparmno<=info->SENparms;iparmno++){
                if( (!flag) && (iparmno != here->BJTsenParmNo) ) continue;
                switch(flag){

                case 0: 
                    /* area : so no DC sensitivity term involved */
                    DvDp = 1.0; 

                    break;
                    /* calculate the DC sensitivities of operating points */
                case 1: 
                    DvDp = model->BJTtype * 
                            (info->SEN_Sap[here->BJTbaseNode][iparmno]
                            -  info->SEN_Sap[here->BJTcolPrimeNode][iparmno]);
                    break;
                case 2: 
                    DvDp = model->BJTtype * 
                            (info->SEN_Sap[here->BJTbasePrimeNode][iparmno]
                            -  info->SEN_Sap[here->BJTemitPrimeNode][iparmno]);
                    break;
                case 3: 
                    DvDp = model->BJTtype * 
                            (info->SEN_Sap[here->BJTbasePrimeNode][iparmno]
                            -  info->SEN_Sap[here->BJTcolPrimeNode][iparmno]);
                    break;
                case 4: 
                    DvDp = model->BJTtype * 
                            (info->SEN_Sap[here->BJTsubstNode][iparmno]
                            -  info->SEN_Sap[here->BJTcolPrimeNode][iparmno]);
                    break;
                }
#ifdef SENSDEBUG
                printf("before loading\n");  
                printf("BJTtype = %d\n",model->BJTtype);  
                printf("DvDp = %.7e , flag = %d , iparmno = %d,senparmno = %d\n"
                        ,DvDp,flag,iparmno,here->BJTsenParmNo);
                printf("senb = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTbaseNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTbaseNode] + iparmno));
                printf("senbrm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTbasePrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTbasePrimeNode] + iparmno));
                printf("senc = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTcolNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTcolNode] + iparmno));
                printf("sencprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTcolPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTcolPrimeNode] + iparmno));
                printf("sene = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTemitNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTemitNode] + iparmno));
                printf("seneprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTemitPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTemitPrimeNode] + iparmno));
                printf("sens = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTsubstNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTsubstNode] + iparmno));
#endif /* SENSDEBUG */


                if(here->BJTbaseNode != here->BJTbasePrimeNode){
                    *(info->SEN_RHS[here->BJTbaseNode] + iparmno) -=  
                            ( cb  - cb0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJTbaseNode] + iparmno) -=  
                            ( icb  - icb0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJTbasePrimeNode] + iparmno) -=  
                        ( cbprm  - cbprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJTbasePrimeNode] + iparmno) -=  
                        ( icbprm  - icbprm0) * DELAinv * DvDp;

                if(here->BJTcolNode != here->BJTcolPrimeNode){
                    *(info->SEN_RHS[here->BJTcolNode] + iparmno) -=  
                            ( cc  - cc0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJTcolNode] + iparmno) -=  
                            ( icc  - icc0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJTcolPrimeNode] + iparmno) -=  
                        ( ccprm  - ccprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJTcolPrimeNode] + iparmno) -=  
                        ( iccprm  - iccprm0) * DELAinv * DvDp;

                if(here->BJTemitNode != here->BJTemitPrimeNode){
                    *(info->SEN_RHS[here->BJTemitNode] + iparmno) -= 
                            ( ce  - ce0) * DELAinv * DvDp;
                    *(info->SEN_iRHS[here->BJTemitNode] + iparmno) -= 
                            ( ice  - ice0) * DELAinv * DvDp;
                }

                *(info->SEN_RHS[here->BJTemitPrimeNode] + iparmno) -= 
                        ( ceprm  - ceprm0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJTemitPrimeNode] + iparmno) -= 
                        ( iceprm  - iceprm0) * DELAinv * DvDp;
                *(info->SEN_RHS[here->BJTsubstNode] + iparmno) -=  
                        ( cs  - cs0) * DELAinv * DvDp;
                *(info->SEN_iRHS[here->BJTsubstNode] + iparmno) -=  
                        ( ics  - ics0) * DELAinv * DvDp;
#ifdef SENSDEBUG
                printf("after loading\n");                  

                printf("senb = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTbaseNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTbaseNode] + iparmno));
                printf("senbrm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTbasePrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTbasePrimeNode] + iparmno));
                printf("senc = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTcolNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTcolNode] + iparmno));
                printf("sencprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTcolPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTcolPrimeNode] + iparmno));
                printf("sene = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTemitNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTemitNode] + iparmno));
                printf("seneprm = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTemitPrimeNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTemitPrimeNode] + iparmno));
                printf("sens = %.7e + j%.7e\n "
                        ,*(info->SEN_RHS[here->BJTsubstNode] + iparmno),
                        *(info->SEN_iRHS[here->BJTsubstNode] + iparmno));
#endif /* SENSDEBUG */

            }


next1:          
            switch(flag){
            case 0: 
                if (*(here->BJTsenCbx) == 0){ 
                    here->BJTarea = A0;
                    goto pertvbe ;
                }
                else{
                    goto pertvbx;
                }
            case 1: 
                goto pertvbe ; 
            case 2: 
                goto pertvbc ;
            case 3: 
                goto pertvcs ;
            case 4: 
                break; 
            }

            /* put the unperturbed values back into the state vector */
            for(i=0; i <= 20; i++) {
                *(ckt->CKTstate0 + here->BJTstate + i) = *(SaveState + i);
            }
            here->BJTsenPertFlag = OFF;
        }

    }
    info->SENstatus = NORMAL;
#ifdef SENSDEBUG
    printf("BJTsenacload end\n");
#endif /* SENSDEBUG */
    return(OK);
}

